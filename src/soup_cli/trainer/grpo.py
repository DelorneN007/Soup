"""GRPO (Group Relative Policy Optimization) trainer — wraps trl.GRPOTrainer."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

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
    """v0.53.11 #123 — build a ``_GRPOTrainerVariant`` subclass.

    Returns a subclass of ``trl.GRPOTrainer`` whose ``compute_loss`` routes
    through :func:`soup_cli.utils.grpo_variants.apply_variant_loss`. Cached
    so multiple instantiations with the same (base, variant) share one class.

    Pure factory — no torch / trl imports at module load time. Variant
    name is normalised via ``validate_grpo_variant`` BEFORE the cache
    boundary so ``"GSPO"`` and ``"gspo"`` share one class (security review
    MEDIUM fix).
    """
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
                "loss (%s); the selected objective is NOT being applied. "
                "This usually means a TRL version renamed the per-token "
                "log-prob inputs.",
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
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
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

            if logp_new is None or logp_old is None or advantages is None:
                self._warn_fallback("missing per-token log-prob inputs")
                return None

            beta_attr = getattr(getattr(self, "args", None), "beta", None)
            beta = float(beta_attr) if beta_attr is not None else 0.0
            delta = getattr(self, "_soup_grpo_delta", None)
            ref_logp = _read_attr(inputs, "ref_per_token_logps")
            mask = completion_mask
            tool_mask = _read_attr(inputs, "tool_mask")
            if mask is not None and tool_mask is not None:
                mask = mask * tool_mask

            try:
                variant_loss = apply_variant_loss(
                    self._soup_grpo_variant,
                    logp_new=logp_new,
                    logp_old=logp_old,
                    advantages=advantages,
                    beta=beta,
                    delta=delta,
                    completion_mask=mask,
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
            if hasattr(super(), "_compute_loss"):
                return super()._compute_loss(model, inputs)
            return super().compute_loss(model, inputs)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if hasattr(super(), "_compute_loss"):
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = self._compute_variant_loss(model, inputs)
            if loss is not None:
                if return_outputs:
                    return loss, None
                return loss
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    _GRPOTrainerVariant.__name__ = f"_GRPOTrainerVariant_{variant}"
    return _GRPOTrainerVariant


def _read_attr(obj: Any, name: str) -> Any:
    """Read ``name`` from a mapping OR object — TRL inputs vary in shape."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        return obj.get(name)
    return getattr(obj, name, None)


def _select_reward_fn(
    tcfg: TrainingConfig, device: str, trust_remote_code: bool
) -> "Any":
    if tcfg.prm_reward is not None:
        from soup_cli.utils.prm_reward import build_prm_reward_fn
        return build_prm_reward_fn(tcfg, device, trust_remote_code)
    from soup_cli.trainer.rewards import load_reward_fns
    fns = load_reward_fns(tcfg.reward_fn, verifiable_domain=tcfg.verifiable_domain)
    return fns[0] if len(fns) == 1 else fns


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
        self.trust_remote_code = trust_remote_code
        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        requires = model_requires_trust_remote_code(config.base) or False
        self._trust_remote_code = resolve_trust_remote_code(
            config.base,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )
        self.model = None
        self.tokenizer = None
        self.trainer = None

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
            GRPOTrainer = make_grpo_trainer_variant(GRPOTrainer, variant)

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
            self.tokenizer.chat_template = "{% for msg in messages %}{{ msg['content'] }}\n{% endfor %}"

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
            console.print(f"[green]Auto batch size (GRPO):[/] {batch_size}")

        num_gen = tcfg.num_generations
        if batch_size < num_gen:
            batch_size = num_gen

        train_data = _prepare_grpo_dataset(dataset["train"])
        _validate_grpo_reward_metadata(train_data, tcfg, split="train")

        if tcfg.rollout_backend is not None:
            from soup_cli.utils.agent_rollout import launch_rollout
            rollout_result = launch_rollout(
                tcfg.rollout_backend,
                prompts=[row["prompt"] for row in train_data],
                rollout_func=tcfg.rollout_func,
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=reward_fn,
            )
            train_data = _prepare_grpo_dataset([dict(row) for row in rollout_result.rows])
            _validate_grpo_reward_metadata(train_data, tcfg, split="rollout")

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

        if tcfg.vllm_sleep_mode:
            import inspect as _inspect
            from soup_cli.utils.grpo_long_context import maybe_enable_trl_sleep_mode
            maybe_enable_trl_sleep_mode(grpo_kwargs, _inspect.signature(GRPOConfig).parameters, console)

        if self.device == "cpu":
            import inspect as _inspect
            grpo_params = _inspect.signature(GRPOConfig).parameters
            if "use_cpu" in grpo_params:
                grpo_kwargs["use_cpu"] = True
            if "generation_kwargs" in grpo_params:
                grpo_kwargs["generation_kwargs"] = {"min_new_tokens": 1}

        grpo_config = GRPOConfig(**grpo_kwargs)

        if self.device == "cpu" and hasattr(self.model, "generation_config"):
            self.model.generation_config.min_new_tokens = 1

        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            reward_funcs=reward_fn,
            processing_class=self.tokenizer,
        )

        if self.deepspeed_config:
            from soup_cli.utils.deepspeed import attach_empty_param_group_guard
            attach_empty_param_group_guard(self.trainer)

        if tcfg.grpo_variant is not None and tcfg.grpo_variant != "standard" and tcfg.grpo_delta is not None:
            self.trainer._soup_grpo_delta = float(tcfg.grpo_delta)

        from soup_cli.utils.peft_wiring import (
            attach_curriculum_callback,
            attach_grpo_stability_callback,
            attach_plugin_callback,
            attach_relora_callback,
            attach_rl_callbacks,
        )

        attach_grpo_stability_callback(self.trainer, tcfg)
        attach_rl_callbacks(self.trainer, tcfg, buffer=self._rl_buffer, tokenizer=self.tokenizer, output_dir=str(output_dir), task="grpo")
        attach_relora_callback(self.trainer, tcfg)
        attach_curriculum_callback(self.trainer, tcfg, str(output_dir), console)
        attach_plugin_callback(self.trainer, console)

        self._output_dir = str(output_dir)
        self._batch_size = batch_size

    def _setup_transformers(self, cfg: SoupConfig, tcfg) -> None:
        from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=self._trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from soup_cli.utils.quant_menu import build_quantization_config_for_loader
        quant_config_obj = build_quantization_config_for_loader(tcfg=tcfg, base=cfg.base, console=console)

        dev_map = resolve_device_map(self.device)
        model_kwargs = {
            "trust_remote_code": self._trust_remote_code,
            "device_map": dev_map,
            "torch_dtype": resolve_frozen_base_load_dtype(self.device),
        }
        if quant_config_obj is not None:
            model_kwargs["quantization_config"] = quant_config_obj

        self.model = AutoModelForCausalLM.from_pretrained(cfg.base, **model_kwargs)
        from soup_cli.utils.data_pipeline import apply_vocab_expansion
        apply_vocab_expansion(self.tokenizer, self.model, cfg.data)

        if tcfg.quantization in ("4bit", "8bit", "mxfp4"):
            from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=should_enable_hf_gradient_checkpointing(
                    tcfg.gradient_checkpointing, stream_layers=tcfg.stream_layers
                ),
            )

        from soup_cli.utils.peft_wiring import build_lora_config, resolve_lora_target_modules
        target_modules = resolve_lora_target_modules(self.model, tcfg.lora.target_modules)
        from soup_cli.utils.moe import resolve_moe_lora_targets
        target_modules = resolve_moe_lora_targets(self.model, tcfg, target_modules, console)

        lora_config = build_lora_config(tcfg.lora, target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
        from soup_cli.utils.peft_wiring import apply_post_lora_patches, apply_pre_lora_patches
        apply_pre_lora_patches(self.model, cfg.base)
        self.model = get_peft_model(self.model, lora_config)
        apply_post_lora_patches(self.model)

        if tcfg.quantization_aware and tcfg.quantization_aware != "fp8":
            from soup_cli.utils.qat import prepare_model_for_qat
            self.model = prepare_model_for_qat(self.model)

        from soup_cli.utils.v028_features import apply_v028_speed_memory
        apply_v028_speed_memory(model=self.model, tcfg=tcfg, base_model=cfg.base, console=console, device=self.device, backend=cfg.backend)

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

    def train(
        self,
        display: Optional[object] = None,
        tracker: Optional[object] = None,
        run_id: str = "",
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict:
        start = time.time()
        if display:
            from soup_cli.monitoring.callback import SoupTrainerCallback, soup_callback_kwargs
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
        with activation_offloading_context(self.config.training, self._output_dir):
            align_trainable_dtype_for_fp16(
                self.trainer.model,
                fp16=getattr(self.trainer.args, "fp16", False),
                bf16=getattr(self.trainer.args, "bf16", False),
            )
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
    """Convert dataset rows to GRPO format, robustly supporting multi-turn, ShareGPT, and Alpaca schemas."""
    prepared = []
    for row in data:
        new_row = dict(row)
        
        # 1. ShareGPT format conversion (`conversations` -> `prompt`)
        if "conversations" in new_row and "prompt" not in new_row:
            convs = new_row["conversations"]
            messages = []
            for turn in convs:
                role = turn.get("from") or turn.get("role")
                content = turn.get("value") or turn.get("content")
                if role and content:
                    if role in ("human", "user"):
                        role = "user"
                    elif role in ("gpt", "assistant"):
                        role = "assistant"
                    elif role == "system":
                        role = "system"
                    messages.append({"role": role, "content": content})
            new_row["prompt"] = messages
            
        # 2. Alpaca format conversion (`instruction`, `input`, `output`)
        elif "instruction" in new_row and "prompt" not in new_row:
            instruction = new_row.get("instruction", "")
            inp = new_row.get("input", "")
            prompt_text = f"{instruction}\n\nInput:\n{inp}" if inp else instruction
            new_row["prompt"] = [{"role": "user", "content": prompt_text}]
            if "output" in new_row and "completion" not in new_row:
                new_row["completion"] = new_row["output"]
                
        # 3. Standard mappings
        elif "messages" in new_row and "prompt" not in new_row:
            new_row["prompt"] = new_row["messages"]
        elif "prompt" not in new_row and "text" in new_row:
            new_row["prompt"] = [{"role": "user", "content": new_row["text"]}]
            
        prepared.append(new_row)
    return prepared


def _validate_grpo_reward_metadata(data: list[dict], tcfg: TrainingConfig, split: str) -> None:
    """Validate that GRPO dataset rows contain valid prompt structures."""
    if not data:
        return
    sample = data[0]
    if "prompt" not in sample:
        logger.warning(f"GRPO dataset split '{split}' rows are missing a 'prompt' field.")