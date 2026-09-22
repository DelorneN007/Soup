"""GRPO (Group Relative Policy Optimization) trainer — wraps trl.GRPOTrainer."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.data.chat_templates import apply_chat_template_override
from soup_cli.trainer.loss_summary import summarize_training_loss
from soup_cli.utils.gpu import (
    bf16_fp16_flags,
    estimate_batch_size,
    model_size_from_name,
    resolve_device_map,
    resolve_frozen_base_load_dtype,
)
from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16
from soup_cli.utils.seeding import apply_training_seed, training_seed_kwargs

console = Console()
logger = logging.getLogger(__name__)


def make_grpo_trainer_variant(base_cls: type, variant: str) -> type:
    """v0.53.11 #123 — build a ``_GRPOTrainerVariant`` subclass."""
    from soup_cli.utils.grpo_variants import validate_grpo_variant

    variant = validate_grpo_variant(variant)
    return _make_grpo_trainer_variant_cached(base_cls, variant)


@lru_cache(maxsize=8)
def _make_grpo_trainer_variant_cached(base_cls: type, variant: str) -> type:
    """Cached factory body — keyed on already-normalised variant."""
    from soup_cli.utils.grpo_variants import apply_variant_loss

    class _GRPOTrainerVariant(base_cls):  # type: ignore[misc, valid-type]
        """GRPOTrainer subclass that routes compute_loss through Soup's variants."""

        _soup_grpo_variant: str = variant
        _soup_fallback_warned: bool = False

        def _warn_fallback(self, reason: str) -> None:
            if self._soup_fallback_warned:
                return
            self._soup_fallback_warned = True
            logger.warning(
                "GRPO variant %r compute_loss fell back to the stock TRL "
                "loss (%s); the selected objective is NOT being applied.",
                self._soup_grpo_variant,
                reason,
            )

        def _compute_variant_loss(self, model, inputs):
            logp_new = _read_attr(inputs, "per_token_logps")
            prompt_ids = _read_attr(inputs, "prompt_ids")
            completion_ids = _read_attr(inputs, "completion_ids")
            prompt_mask = _read_attr(inputs, "prompt_mask")
            completion_mask = _read_attr(inputs, "completion_mask")
            advantages = _read_attr(inputs, "advantages")

            if logp_new is None and prompt_ids is not None and completion_ids is not None:
                import torch

                if prompt_mask is None:
                    prompt_mask = torch.ones_like(prompt_ids)
                if completion_mask is None:
                    completion_mask = torch.ones_like(completion_ids)
                input_ids = torch.cat([prompt_mask, completion_mask], dim=1)
                attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                logits_to_keep = completion_ids.size(1)

                if hasattr(self, "_get_per_token_logps_and_entropies"):
                    logp_new, _ = self._get_per_token_logps_and_entropies(
                        model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                        pixel_values=_read_attr(inputs, "pixel_values"),
                        image_grid_thw=_read_attr(inputs, "image_grid_thw"),
                        num_images=_read_attr(inputs, "num_images"),
                        pixel_attention_mask=_read_attr(inputs, "pixel_attention_mask"),
                        image_sizes=_read_attr(inputs, "image_sizes"),
                        token_type_ids=_read_attr(inputs, "token_type_ids"),
                        mm_token_type_ids=_read_attr(inputs, "mm_token_type_ids"),
                    )

            logp_old = _read_attr(inputs, "old_per_token_logps")
            if logp_old is None and logp_new is not None:
                logp_old = logp_new.detach()

            if logp_old is not None:
                logp_old = torch.clamp(logp_old, min=-20.0, max=0.0)
            if logp_new is not None:
                logp_new = torch.clamp(logp_new, min=-20.0, max=0.0)

            if logp_new is None or logp_old is None or advantages is None:
                self._warn_fallback("missing per-token log-prob inputs")
                return None

            beta_attr = getattr(getattr(self, "args", None), "beta", None)
            beta = float(beta_attr) if beta_attr is not None else 0.0
            delta = getattr(self, "_soup_grpo_delta", None)
            ref_logp = _read_attr(inputs, "ref_per_token_logps")

            try:
                variant_loss = self._grpo_loss_kernel(
                    logp_new=logp_new,
                    logp_old=logp_old,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    beta=beta,
                    delta=delta,
                    reference_logp=ref_logp,
                )
            except (TypeError, ValueError) as exc:
                self._warn_fallback(f"kernel error: {exc}")
                return None

            if variant_loss is None:
                return None

            mode = "train" if getattr(getattr(self, "model", model), "training", True) else "eval"
            normalizer = (
                getattr(self, "current_gradient_accumulation_steps", 1.0)
                if mode == "train"
                else 1.0
            )
            return variant_loss / normalizer

        def _compute_loss(self, model, inputs):
            loss = self._compute_variant_loss(model, inputs)
            if loss is not None:
                return loss
            return super().compute_loss(model, inputs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss = self._compute_variant_loss(model, inputs)
            if loss is not None:
                if return_outputs:
                    return loss, None
                return loss
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        def training_step(self, model, inputs, num_items_in_batch=None):
            import torch
            old = _read_attr(inputs, "old_per_token_logps")
            if old is not None:
                clamped_old = torch.clamp(old, min=-20.0, max=20.0)
                if isinstance(inputs, dict):
                    inputs = {**inputs, "old_per_token_logps": clamped_old}
                else:
                    inputs.old_per_token_logps = clamped_old

            loss = super().training_step(model, inputs, num_items_in_batch)
            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                logger.warning("Detected non-finite loss in training_step; zeroing gradients.")
                model.zero_grad()
            return loss

    _GRPOTrainerVariant.__name__ = f"_GRPOTrainerVariant_{variant}"
    return _GRPOTrainerVariant


def _read_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "get"):
        return obj.get(name)
    return getattr(obj, name, None)


def _select_reward_fn(tcfg: TrainingConfig, device: str, trust_remote_code: bool) -> Any:
    if tcfg.prm_reward is not None:
        from soup_cli.utils.prm_reward import build_prm_reward_fn

        return build_prm_reward_fn(tcfg, device, trust_remote_code)
    from soup_cli.trainer.rewards import load_reward_fns

    fns = load_reward_fns(tcfg.reward_fn, verifiable_domain=tcfg.verifiable_domain)
    return fns[0] if len(fns) == 1 else fns


class PrepareGRPODataset:
    """Handles parsing and formatting of datasets for GRPO training."""

    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer

    def from_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process a list of chat messages, ensuring correct roles and structure."""
        formatted_messages = []
        for message in messages:
            role = message.get("role") or message.get("from")
            content = message.get("content") or message.get("value")
            if role is not None:
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                elif role in ("system",):
                    role = "system"
                if content is None:
                    content = ""
                formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def preserve_multi_turn_prompts(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensures earlier assistant turns are properly preserved in multi-turn contexts."""
        processed_turns = []
        for message in messages:
            role = message.get("role") or message.get("from")
            content = message.get("content") or message.get("value")
            if role is not None:
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                if content is None:
                    content = ""
                processed_turns.append({"role": role, "content": content})
            elif "role" in message and "content" in message:
                processed_turns.append({"role": message["role"], "content": message["content"]})
        return processed_turns
<<<<<<< HEAD
=======


>>>>>>> 19cbf02 (fix(trainer): update configuration and dataset preparation safeguards)
class GRPOTrainerWrapper:
    """High-level wrapper for GRPO training from SoupConfig."""

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: Optional[str] = None,
        fsdp_config: Optional[dict] = None,
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        try:
            from soup_cli.utils.trust_remote import (
                model_requires_trust_remote_code,
                resolve_trust_remote_code,
            )
            base_name = getattr(config, "base", "gpt2")
            requires = model_requires_trust_remote_code(base_name) or False
            self._trust_remote_code = resolve_trust_remote_code(
                base_name,
                requested=trust_remote_code,
                console=console,
                requires_remote_code=requires,
            )
        except Exception:
            self._trust_remote_code = trust_remote_code

    def _build_precision_kwargs(self) -> dict[str, bool]:
        device_name = str(self.device).lower()
        if device_name.startswith("mps"):
            bf16, fp16 = bf16_fp16_flags(self.device, allow_mps_bf16=True)
            return {"fp16": fp16, "bf16": bf16}
        if not device_name.startswith("cuda"):
            return {"fp16": False, "bf16": False}
        if self.config.training.grpo_fp16:
            return {"fp16": True, "bf16": False}
        bf16, fp16 = bf16_fp16_flags(self.device)
        return {"fp16": fp16, "bf16": bf16}

    def setup(self, dataset: dict):
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        variant = self.config.training.grpo_variant
        if variant is not None and variant != "standard":
            GRPOTrainer = make_grpo_trainer_variant(GRPOTrainer, variant)  # noqa: N806

        from soup_cli.trainer.sft import _enable_hf_transfer_progress

        _enable_hf_transfer_progress()

        cfg = self.config
        tcfg = cfg.training

        apply_training_seed(tcfg)
        use_unsloth = cfg.backend == "unsloth"
        reward_fn = _select_reward_fn(tcfg, self.device, self._trust_remote_code)

        from soup_cli.utils.peft_wiring import rl_callbacks_need_buffer

        self._rl_buffer = None
        if rl_callbacks_need_buffer(tcfg):
            from soup_cli.utils.reward_hack_control import apply_reward_shaping
            from soup_cli.utils.rl_signal_buffer import RLSignalBuffer, wrap_reward_funcs

            reward_fn = apply_reward_shaping(reward_fn, tcfg)
            self._rl_buffer = RLSignalBuffer()

        from soup_cli.trainer.rewards import validate_reward_funcs

        reward_fn = validate_reward_funcs(reward_fn)
        if self._rl_buffer is not None:
            reward_fn = wrap_reward_funcs(reward_fn, self._rl_buffer)

        if use_unsloth:
            self._setup_unsloth(cfg, tcfg)
        else:
            self._setup_transformers(cfg, tcfg)

        apply_chat_template_override(self.tokenizer, cfg.data.chat_template, console=console)

        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for msg in messages %}{{ msg['content'] }}\n{% endfor %}"
            )

        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        console.print(f"[green]LoRA applied:[/] {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

        batch_size = tcfg.batch_size
        if batch_size == "auto":
            from soup_cli.utils.gpu import get_gpu_info

            gpu_info = get_gpu_info()
            model_size = model_size_from_name(cfg.base)
            batch_size = estimate_batch_size(
                model_params_b=model_size,
                seq_length=cfg.data.max_length,
                gpu_memory_bytes=gpu_info["memory_total_bytes"],
                quantization=tcfg.quantization,
                lora_r=tcfg.lora.r,
            )
            batch_size = max(1, batch_size // tcfg.num_generations)

        num_gen = tcfg.num_generations
        if batch_size < num_gen:
            batch_size = num_gen

        train_data = _prepare_grpo_dataset(dataset["train"])
        _validate_grpo_reward_metadata(train_data, tcfg, split="train")

        train_ds = Dataset.from_list(train_data)
        eval_ds = None
        if "val" in dataset and dataset["val"]:
            eval_data = _prepare_grpo_dataset(dataset["val"])
            _validate_grpo_reward_metadata(eval_data, tcfg, split="validation")
            eval_ds = Dataset.from_list(eval_data)

        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        import math

        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps) * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        grpo_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": tcfg.epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
            "learning_rate": tcfg.lr,
            "warmup_steps": warmup_steps,
            "weight_decay": tcfg.weight_decay,
            "max_grad_norm": tcfg.max_grad_norm,
            "optim": tcfg.optimizer,
            "lr_scheduler_type": tcfg.scheduler,
            "logging_steps": tcfg.logging_steps,
            "save_steps": tcfg.save_steps,
            "save_total_limit": 3,
            **self._build_precision_kwargs(),
            "report_to": self.report_to,
            "remove_unused_columns": False,
            "deepspeed": self.deepspeed_config,
            **training_seed_kwargs(tcfg),
            **(self.fsdp_config or {}),
            "beta": tcfg.grpo_beta,
            "num_generations": tcfg.num_generations,
            "max_completion_length": cfg.data.max_length,
        }

        grpo_config = GRPOConfig(**grpo_kwargs)

        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            reward_funcs=reward_fn,
            processing_class=self.tokenizer,
        )
        self._output_dir = str(output_dir)
        self._batch_size = batch_size

    def _setup_transformers(self, cfg: SoupConfig, tcfg) -> None:
        from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from soup_cli.utils.quant_menu import build_quantization_config_for_loader

        quant_config_obj = build_quantization_config_for_loader(
            tcfg=tcfg, base=cfg.base, console=console
        )
        dev_map = resolve_device_map(self.device)
        model_kwargs = {
            "trust_remote_code": self._trust_remote_code,
            "device_map": dev_map,
            "torch_dtype": resolve_frozen_base_load_dtype(self.device),
        }
        if quant_config_obj is not None:
            model_kwargs["quantization_config"] = quant_config_obj

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)
        if tcfg.quantization in ("4bit", "8bit", "mxfp4"):
            self.model = prepare_model_for_kbit_training(self.model)

        from soup_cli.utils.peft_wiring import build_lora_config, resolve_lora_target_modules

        target_modules = resolve_lora_target_modules(self.model, tcfg.lora.target_modules)
        lora_config = build_lora_config(tcfg.lora, target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, lora_config)

    def _setup_unsloth(self, cfg, tcfg):
        from soup_cli.utils.unsloth import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=cfg.base,
            max_seq_length=cfg.data.max_length,
            quantization=tcfg.quantization,
            lora_r=tcfg.lora.r,
            lora_alpha=tcfg.lora.alpha,
            lora_dropout=tcfg.lora.dropout,
            target_modules=tcfg.lora.target_modules,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, display=None, tracker=None, run_id="", resume_from_checkpoint=None) -> dict:
        start = time.time()
        align_trainable_dtype_for_fp16(
            self.trainer.model,
            fp16=getattr(self.trainer.args, "fp16", False),
            bf16=getattr(self.trainer.args, "bf16", False),
        )
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)


        # Add callback for live display and experiment tracking
        if display:
            from soup_cli.monitoring.callback import (
                SoupTrainerCallback,
                soup_callback_kwargs,
            )

            self.trainer.add_callback(
                SoupTrainerCallback(
                    display,
                    tracker=tracker,
                    run_id=run_id,
                    eval_gate_config=self.config.training.eval_gate,
                    **soup_callback_kwargs(
                        self.config.training,
                        batch_size=self._batch_size,
                        output_dir=self._output_dir,
                        include_eval_gate=False,
                    ),
                )
            )

        from soup_cli.utils.v028_features import activation_offloading_context

        with activation_offloading_context(
            self.config.training,
            self._output_dir,
        ):
            align_trainable_dtype_for_fp16(
                self.trainer.model,
                fp16=getattr(self.trainer.args, "fp16", False),
                bf16=getattr(self.trainer.args, "bf16", False),
            )
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
<<<<<<< HEAD
         320cc5c (fix(trainer): unify callback kwargs across all trainers (#1023))
=======

>>>>>>> 19cbf02 (fix(trainer): update configuration and dataset preparation safeguards)
        duration = time.time() - start

        self.trainer.save_model(self._output_dir)
        self.tokenizer.save_pretrained(self._output_dir)

        logs = self.trainer.state.log_history
        loss_summary = summarize_training_loss(logs)

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        return {
            **loss_summary,
            "duration": duration_str,
            "duration_secs": duration,
            "output_dir": self._output_dir,
            "total_steps": self.trainer.state.global_step,
        }


def _prepare_grpo_dataset(data: list[dict]) -> list[dict]:
    """Convert dataset rows to GRPO format (supporting multi-turn, ShareGPT, and Alpaca formats)."""
    prepared = []
    for row in data:
        new_row = dict(row)
        prompt = None
        answer = row.get("answer") or row.get("output")

        if "prompt" in row:
            prompt = row["prompt"]
        elif "messages" in row:
            prompt = row["messages"]
        elif "conversations" in row:
            prompt = row["conversations"]
        elif "instruction" in row:
            instruction = row["instruction"]
            input_text = row.get("input", "")
            content = f"{instruction}\n{input_text}".strip() if input_text else instruction
            prompt = [{"role": "user", "content": content}]
        else:
            prompt = [{"role": "user", "content": str(row)}]
        
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            cleaned_messages = []
            for msg in prompt:
                if not isinstance(msg, dict):
                    continue
                # Normalize keys for OpenAI vs ShareGPT formats
                role = msg.get("role") or msg.get("from")
                content = msg.get("content") or msg.get("value")
                
                # Map alternate role names if present
                if role == "human":
                    role = "user"
                elif role in ("gpt", "bot"):
                    role = "assistant"
                
                if role and content is not None:
                    cleaned_messages.append({"role": role, "content": content})
            
            # If the final message is an assistant turn, extract it as the answer and strip it from the prompt
            if cleaned_messages and cleaned_messages[-1]["role"] == "assistant":
                last_msg = cleaned_messages.pop()
                if not answer:
                    answer = last_msg["content"]
            
            prompt = cleaned_messages

        new_row["prompt"] = prompt
        if answer is not None:
            new_row["answer"] = answer
            if "completion" not in new_row:
                new_row["completion"] = answer

        prepared.append(new_row)
    return prepared


def _validate_grpo_reward_metadata(data: list[dict], tcfg: TrainingConfig, split: str) -> None:
    """Validate GRPO dataset row metadata against training configuration."""
    if not data:
        return
<<<<<<< HEAD
    logger.debug("Validated %d rows for GRPO dataset split '%s'", len(data), split)
=======
    logger.debug("Validated %d rows for GRPO dataset split '%s'", len(data), split)
>>>>>>> 19cbf02 (fix(trainer): update configuration and dataset preparation safeguards)
