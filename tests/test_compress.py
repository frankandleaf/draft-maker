"""Smoke test for compress.py: verify projection + slicing logic without GPU."""

import torch
import pytest
from draft_adapter.compress import (
    WidthCompressor,
    _get_projection_rule,
)
from draft_adapter.inspect import ModelArchitecture


def create_dummy_arch(hidden_size=1024, num_heads=16, num_kv=4, head_dim=64,
                      intermediate=4096, vocab=32000, num_layers=8):
    """head_dim is FROZEN: target_head_dim == head_dim."""
    return ModelArchitecture(
        model_type="llama",
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        target_embed_dim=512,
        target_head_dim=head_dim,     # FROZEN
        target_num_heads=8,           # 512 // 64 = 8
        target_num_kv_heads=2,
        target_intermediate_size=2048,
        target_num_layers=6,
    )


class TestProjectionRules:
    """Verify state dict keys map to correct projection rules."""

    def test_q_proj_is_input(self):
        assert _get_projection_rule("model.layers.0.self_attn.q_proj.weight") == "input"

    def test_k_proj_is_input(self):
        assert _get_projection_rule("model.layers.0.self_attn.k_proj.weight") == "input"

    def test_v_proj_is_input(self):
        assert _get_projection_rule("model.layers.0.self_attn.v_proj.weight") == "input"

    def test_o_proj_is_output(self):
        assert _get_projection_rule("model.layers.0.self_attn.o_proj.weight") == "output"

    def test_gate_proj_is_input(self):
        assert _get_projection_rule("model.layers.0.mlp.gate_proj.weight") == "input"

    def test_up_proj_is_input(self):
        assert _get_projection_rule("model.layers.0.mlp.up_proj.weight") == "input"

    def test_down_proj_is_output(self):
        assert _get_projection_rule("model.layers.0.mlp.down_proj.weight") == "output"

    def test_embed_is_embed(self):
        assert _get_projection_rule("model.embed_tokens.weight") == "embed"

    def test_lm_head_is_embed(self):
        assert _get_projection_rule("lm_head.weight") == "embed"

    def test_norm_is_norm(self):
        assert _get_projection_rule("model.norm.weight") == "norm"

    def test_input_layernorm_is_norm(self):
        assert _get_projection_rule("model.layers.0.input_layernorm.weight") == "norm"

    def test_unknown_key_is_skip(self):
        assert _get_projection_rule("some.random.key") == "skip"


class TestWidthCompressorProjection:
    """Test weight projection with dummy weights."""

    @pytest.fixture
    def arch(self):
        return create_dummy_arch()

    @pytest.fixture
    def compressor(self, arch):
        # Create a dummy orthogonal Q_top
        d, d_prime = arch.hidden_size, arch.target_embed_dim  # 1024, 512
        # Create random orthogonal matrix
        Q = torch.randn(d, d)
        Q, _ = torch.linalg.qr(Q)  # orthogonal
        Q_top = Q[:, :d_prime]  # [d, d']
        Q_top = Q_top.float()

        comp = WidthCompressor(arch)
        comp.Q_top = Q_top
        return comp

    def test_q_proj_input_side_with_slice(self, compressor):
        """q_proj: head_dim frozen, only num_heads reduced."""
        d, d_prime = 1024, 512
        nh, hd = compressor.arch.num_attention_heads, compressor.arch.head_dim  # 16, 64
        target_nh = compressor.arch.target_num_heads  # 8
        # head_dim frozen: 64
        W_q = torch.randn(nh * hd, d)
        result = compressor._project_weight(W_q, "model.layers.0.self_attn.q_proj.weight")
        assert result.shape == (target_nh * hd, d_prime)  # 8*64=512

    def test_o_proj_output_side_with_slice(self, compressor):
        """o_proj: head_dim frozen, only num_heads reduced."""
        d, d_prime = 1024, 512
        nh, hd = 16, 64
        target_nh = 8
        W_o = torch.randn(d, nh * hd)
        result = compressor._project_weight(W_o, "model.layers.0.self_attn.o_proj.weight")
        assert result.shape == (d_prime, target_nh * hd)  # 512, 512

    def test_k_proj_input_side_with_slice(self, compressor):
        """k_proj: head_dim frozen, only num_kv_heads reduced."""
        d, d_prime = 1024, 512
        num_kv, hd = compressor.arch.num_kv_heads, compressor.arch.head_dim  # 4, 64
        target_kv = compressor.arch.target_num_kv_heads  # 2
        W_k = torch.randn(num_kv * hd, d)
        result = compressor._project_weight(W_k, "model.layers.0.self_attn.k_proj.weight")
        assert result.shape == (target_kv * hd, d_prime)  # 2*64=128

    def test_v_proj_input_side_with_slice(self, compressor):
        """v_proj: same as k_proj."""
        d, d_prime = 1024, 512
        num_kv, hd = 4, 64
        target_kv = 2
        W_v = torch.randn(num_kv * hd, d)
        result = compressor._project_weight(W_v, "model.layers.0.self_attn.v_proj.weight")
        assert result.shape == (target_kv * hd, d_prime)  # 2*64=128

    def test_gate_proj_input_side_with_slice(self, compressor):
        """gate_proj [intermediate, d] → [target_intermediate, d']."""
        d, d_prime = 1024, 512
        intermed = compressor.arch.intermediate_size  # 4096
        target_int = compressor.arch.target_intermediate_size  # 2048

        W_gate = torch.randn(intermed, d)
        result = compressor._project_weight(W_gate, "model.layers.0.mlp.gate_proj.weight")
        assert result.shape == (target_int, d_prime)

    def test_down_proj_output_side_with_slice(self, compressor):
        """down_proj [d, intermediate] → [d', target_intermediate]."""
        d, d_prime = 1024, 512
        intermed = 4096
        target_int = 2048

        W_down = torch.randn(d, intermed)
        result = compressor._project_weight(W_down, "model.layers.0.mlp.down_proj.weight")
        assert result.shape == (d_prime, target_int)

    def test_embed_tokens(self, compressor):
        """embed [V, d] → [V, d']."""
        d, d_prime = 1024, 512
        V = 32000

        W_emb = torch.randn(V, d)
        result = compressor._project_weight(W_emb, "model.embed_tokens.weight")
        assert result.shape == (V, d_prime)

    def test_lm_head(self, compressor):
        """lm_head (untied) [V, d] → [V, d']."""
        d, d_prime = 1024, 512
        V = 32000

        W_lm = torch.randn(V, d)
        result = compressor._project_weight(W_lm, "lm_head.weight")
        assert result.shape == (V, d_prime)

    def test_norm_weight(self, compressor):
        """norm [d] → Q^T @ weight → [d'] (projected, not sliced)."""
        d, d_prime = 1024, 512
        w_norm = torch.randn(d)
        result = compressor._project_weight(w_norm, "model.norm.weight")
        assert result.shape == (d_prime,)
        # Verify result is NOT just the first d' elements
        assert not torch.allclose(result, w_norm[:d_prime])

    def test_layer_norm_weight(self, compressor):
        """RMSNorm weight is projected, not sliced."""
        w_norm = torch.randn(1024)
        result = compressor._project_weight(w_norm, "model.layers.0.input_layernorm.weight")
        assert result.shape == (512,)
        assert not torch.allclose(result, w_norm[:512])


class TestQOrthogonality:
    """Verify Q_top.T @ Q_top ≈ I."""

    def test_random_orthogonal_q(self):
        d, d_prime = 128, 64
        Q = torch.randn(d, d)
        Q, _ = torch.linalg.qr(Q)
        Q_top = Q[:, :d_prime]

        identity = Q_top.T @ Q_top
        expected = torch.eye(d_prime)
        assert torch.allclose(identity, expected, atol=1e-5)
