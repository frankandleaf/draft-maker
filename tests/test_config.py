"""Tests for config validation."""

import pytest
from draft_adapter.config import DepthConfig, DistillConfig, PipelineConfig, WidthConfig


class TestWidthConfig:
    def test_defaults(self):
        cfg = WidthConfig()
        assert cfg.head_dim_factor == 0.5
        assert cfg.head_size_factor == 0.5
        assert cfg.embed_size_factor == 0.5

    def test_valid_factors(self):
        cfg = WidthConfig(head_dim_factor=0.5, head_size_factor=0.3, embed_size_factor=0.1)
        assert cfg.head_dim_factor == 0.5

    def test_zero_factor_raises(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            WidthConfig(head_dim_factor=0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            WidthConfig(embed_size_factor=0.0)

    def test_negative_factor_raises(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            WidthConfig(head_size_factor=-0.1)

    def test_factor_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
            WidthConfig(embed_size_factor=1.5)


class TestDepthConfig:
    def test_defaults(self):
        cfg = DepthConfig()
        assert cfg.layer_factor == 0.75
        assert cfg.protect_first == 1
        assert cfg.protect_last == 1

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            DepthConfig(layer_factor=0)
        with pytest.raises(ValueError):
            DepthConfig(layer_factor=2.0)


class TestDistillConfig:
    def test_defaults(self):
        cfg = DistillConfig()
        assert cfg.kl_mode == "reverse"
        assert cfg.top_k == 10

    def test_invalid_kl_mode(self):
        with pytest.raises(ValueError):
            DistillConfig(kl_mode="invalid")

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            DistillConfig(top_k=0)


class TestPipelineConfig:
    def test_tokenizer_defaults_to_model(self):
        cfg = PipelineConfig(model="test-model")
        assert cfg.tokenizer == "test-model"

    def test_explicit_tokenizer(self):
        cfg = PipelineConfig(model="test-model", tokenizer="custom-tok")
        assert cfg.tokenizer == "custom-tok"
