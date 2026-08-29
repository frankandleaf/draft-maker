"""Tests for the retained distillation utility configuration."""

import pytest

from draft_adapter.config import DistillConfig


def test_distill_defaults():
    cfg = DistillConfig()
    assert cfg.kl_mode == "reverse"
    assert cfg.top_k == 10


def test_invalid_kl_mode():
    with pytest.raises(ValueError):
        DistillConfig(kl_mode="invalid")


def test_invalid_top_k():
    with pytest.raises(ValueError):
        DistillConfig(top_k=0)


def test_negative_hard_label_weight():
    with pytest.raises(ValueError, match="hard_label_weight"):
        DistillConfig(hard_label_weight=-0.1)
