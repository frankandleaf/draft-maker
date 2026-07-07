"""Tests for architecture inspection."""

import pytest
from draft_adapter.inspect import (
    ModelArchitecture,
    UnsupportedArchitectureError,
    SUPPORTED_ARCHITECTURES,
    compute_targets,
)
from draft_adapter.config import WidthConfig, DepthConfig


class TestModelArchitecture:
    def test_kv_groups(self):
        arch = ModelArchitecture(
            model_type="llama",
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )
        assert arch.num_kv_groups == 4

    def test_kv_groups_no_gqa(self):
        """MHA (num_heads == num_kv_heads)."""
        arch = ModelArchitecture(
            model_type="llama",
            num_layers=12,
            num_attention_heads=12,
            num_kv_heads=12,
            head_dim=64,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )
        assert arch.num_kv_groups == 1


class TestComputeTargets:
    def test_basic_scaling(self):
        """head_dim is FROZEN — only num_heads reduces."""
        arch = ModelArchitecture(
            model_type="qwen3",
            num_layers=24,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
            hidden_size=2048,
            intermediate_size=5632,
            vocab_size=151936,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )

        width = WidthConfig(
            embed_size_factor=0.5, # 2048 -> 1024
        )
        depth = DepthConfig(layer_factor=0.5)  # 24 -> 12

        result = compute_targets(arch, width, depth)

        # head_dim frozen at 128
        assert result.target_head_dim == 128
        # embed_dim = 2048 * 0.5 = 1024, rounded to multiple of 128
        assert result.target_embed_dim == 1024
        # num_heads = embed_dim / head_dim = 1024 / 128 = 8
        assert result.target_num_heads == 8
        # num_kv_heads = num_heads / kv_groups = 8 / 4 = 2
        assert result.target_num_kv_heads == 2
        assert result.target_num_layers == 12

    def test_gqa_invariant_maintained(self):
        """Verify num_heads % num_kv_heads == 0 after compression."""
        arch = ModelArchitecture(
            model_type="llama",
            num_layers=32,
            num_attention_heads=32,
            num_kv_heads=8,
            head_dim=128,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            max_position_embeddings=4096,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )

        width = WidthConfig(embed_size_factor=0.5)
        depth = DepthConfig(layer_factor=0.75)

        result = compute_targets(arch, width, depth)
        assert result.target_num_heads % result.target_num_kv_heads == 0

    def test_embed_dim_matches_heads_times_head_dim(self):
        """head_dim frozen, embed_dim = num_heads * head_dim."""
        arch = ModelArchitecture(
            model_type="qwen2",
            num_layers=28,
            num_attention_heads=28,
            num_kv_heads=4,
            head_dim=128,
            hidden_size=3584,
            intermediate_size=18944,
            vocab_size=152064,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )

        width = WidthConfig(embed_size_factor=0.5)
        depth = DepthConfig(layer_factor=0.5)

        result = compute_targets(arch, width, depth)
        # head_dim frozen at 128
        assert result.target_head_dim == 128
        # embed_dim = 3584 * 0.5 = 1792, rounded to multiple of 128 = 1792
        # num_heads = 1792 / 128 = 14
        assert result.target_num_heads * result.target_head_dim == result.target_embed_dim

    def test_min_layers_protected(self):
        arch = ModelArchitecture(
            model_type="qwen3",
            num_layers=8,
            num_attention_heads=12,
            num_kv_heads=3,
            head_dim=64,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=151936,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )

        width = WidthConfig(embed_size_factor=0.5)
        depth = DepthConfig(layer_factor=0.1, protect_first=1, protect_last=1)  # 8*0.1=0.8->1

        result = compute_targets(arch, width, depth)
        # Should be at least protect_first + protect_last = 2
        assert result.target_num_layers >= 2


class TestUnsupportedArchitecture:
    def test_supported_architectures(self):
        assert "llama" in SUPPORTED_ARCHITECTURES
        assert "qwen2" in SUPPORTED_ARCHITECTURES
        assert "qwen3" in SUPPORTED_ARCHITECTURES
        assert "mistral" in SUPPORTED_ARCHITECTURES
        assert "gemma2" in SUPPORTED_ARCHITECTURES
        assert "stablelm" in SUPPORTED_ARCHITECTURES
