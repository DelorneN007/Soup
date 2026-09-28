"""Tests for GRPO trainer, dataset preparation, and variant factory."""

from __future__ import annotations

import pytest
from soup_cli.trainer.grpo import (
    _prepare_grpo_dataset,
    _validate_grpo_reward_metadata,
    make_grpo_trainer_variant,
)


def test_prepare_grpo_dataset_instruction():
    """Verify conversion of Alpaca-style instruction/input fields into prompts."""
    data = [{"instruction": "Write a python function", "input": "to sort a list"}]
    prepared = _prepare_grpo_dataset(data)
    assert len(prepared) == 1
    assert prepared[0]["prompt"] == "Write a python function\n\nto sort a list"


def test_prepare_grpo_dataset_text():
    """Verify conversion of raw text fields into prompts."""
    data = [{"text": "Explain quantum computing."}]
    prepared = _prepare_grpo_dataset(data)
    assert len(prepared) == 1
    assert prepared[0]["prompt"] == "Explain quantum computing."


def test_prepare_grpo_dataset_messages():
    """Verify chat message lists are kept intact for native chat templates."""
    data = [{"messages": [{"role": "user", "content": "Hello!"}]}]
    prepared = _prepare_grpo_dataset(data)
    assert len(prepared) == 1
    assert "messages" in prepared[0]


def test_validate_grpo_reward_metadata():
    """Verify reward metadata validation helper executes without errors."""
    # Valid rows
    _validate_grpo_reward_metadata([{"prompt": "test prompt"}], None, split="train")
    # Empty dataset case
    _validate_grpo_reward_metadata([], None, split="train")


def test_make_grpo_trainer_variant_factory():
    """Verify the variant trainer subclass factory normalizes names and builds classes."""
    class DummyBaseTrainer:
        def __init__(self, *args, **kwargs):
            pass

    variant_cls = make_grpo_trainer_variant(DummyBaseTrainer, "gspo")
    assert variant_cls is not None
    assert variant_cls._soup_grpo_variant == "gspo"