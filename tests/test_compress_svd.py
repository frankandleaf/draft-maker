"""Tests for SVD hybrid compression."""

import torch
import pytest
from draft_adapter.compress_svd import (
    DecomposedLinear,
    SVDDecomposer,
    SVDCompressor,
    SVDChannelScorer,
)
from draft_adapter.inspect import ModelArchitecture


def create_dummy_arch(hidden_size=256, num_heads=8, num_kv=2, head_dim=32,
                      intermediate=1024, vocab=1000, num_layers=4,
                      es=0.5, ls=0.75):
    """head_dim is FROZEN."""
    target_embed_dim = max(head_dim, int(hidden_size * es))
    target_embed_dim = (target_embed_dim // head_dim) * head_dim
    target_num_heads = target_embed_dim // head_dim
    kv_groups = num_heads // num_kv
    target_num_kv_heads = max(1, target_num_heads // kv_groups)
    target_num_layers = max(1, int(num_layers * ls))

    return ModelArchitecture(
        model_type="llama",
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        target_embed_dim=target_embed_dim,
        target_head_dim=head_dim,
        target_num_heads=target_num_heads,
        target_num_kv_heads=target_num_kv_heads,
        target_intermediate_size=max(8, int(intermediate * es)),
        target_num_layers=target_num_layers,
    )


class TestDecomposedLinear:
    """Verify DecomposedLinear module behavior."""

    def test_forward_shape(self):
        d_in, d_out, r = 64, 128, 16
        mod = DecomposedLinear(d_in, d_out, r, bias=False)
        # Set known weights
        mod.proj_in.weight.data.normal_()
        mod.proj_out.weight.data.normal_()
        x = torch.randn(2, 4, d_in)
        out = mod(x)
        assert out.shape == (2, 4, d_out)

    def test_forward_preserves_grad(self):
        d_in, d_out, r = 32, 64, 8
        mod = DecomposedLinear(d_in, d_out, r, bias=False)
        mod.proj_in.weight.data.normal_()
        mod.proj_out.weight.data.normal_()
        x = torch.randn(2, 4, d_in, requires_grad=False)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        assert mod.proj_in.weight.grad is not None
        assert mod.proj_out.weight.grad is not None

    def test_matches_original_when_full_rank(self):
        """DecomposedLinear at full rank = original nn.Linear."""
        d_in, d_out = 16, 32
        W = torch.randn(d_out, d_in)

        # Full-rank SVD decomposition
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        r = min(d_out, d_in)  # full rank
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        S_sqrt = S_r.sqrt()
        W_out = U_r * S_sqrt.unsqueeze(0)
        W_in = Vt_r * S_sqrt.unsqueeze(1)

        mod = DecomposedLinear(d_in, d_out, r, bias=False)
        mod.proj_in.weight.data = W_in
        mod.proj_out.weight.data = W_out

        x = torch.randn(1, 3, d_in)
        expected = x @ W.t()
        actual = mod(x)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_parameters_count(self):
        d_in, d_out, r = 100, 200, 10
        mod = DecomposedLinear(d_in, d_out, r, bias=False)
        n_params = sum(p.numel() for p in mod.parameters())
        expected = r * d_in + r * d_out  # proj_in + proj_out
        assert n_params == expected


class TestSVDDecomposer:
    """Verify SVDDecomposer weight decomposition."""

    def test_decompose_rank(self):
        decomposer = SVDDecomposer(rank_factor=0.5)
        d_in, d_out = 100, 200
        W = torch.randn(d_out, d_in)
        mod = decomposer.decompose_weight(W, "test")
        expected_rank = max(1, int(min(d_out, d_in) * 0.5))
        assert mod.rank == expected_rank

    def test_decompose_output_shape(self):
        decomposer = SVDDecomposer(rank_factor=0.3)
        d_in, d_out = 64, 128
        W = torch.randn(d_out, d_in)
        mod = decomposer.decompose_weight(W, "test")
        x = torch.randn(2, 8, d_in)
        out = mod(x)
        assert out.shape == (2, 8, d_out)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_decompose_preserves_cuda_device_and_dtype(self):
        decomposer = SVDDecomposer(rank_factor=0.5)
        weight = torch.randn(32, 16, device="cuda", dtype=torch.float16)

        mod = decomposer.decompose_weight(weight, "cuda")

        assert mod.proj_in.weight.device.type == "cuda"
        assert mod.proj_in.weight.dtype == torch.float16
        output = mod(torch.randn(2, 16, device="cuda", dtype=torch.float16))
        assert output.shape == (2, 32)

    def test_rank_factor_one(self):
        """rank_factor=1 means full rank decomposition."""
        decomposer = SVDDecomposer(rank_factor=1.0)
        d_in, d_out = 32, 64
        W = torch.randn(d_out, d_in)
        mod = decomposer.decompose_weight(W, "test")
        assert mod.rank == min(d_out, d_in)

    def test_rank_factor_invalid(self):
        with pytest.raises(ValueError, match="rank_factor"):
            SVDDecomposer(rank_factor=0.0)
        with pytest.raises(ValueError, match="rank_factor"):
            SVDDecomposer(rank_factor=1.5)

    def test_decompose_truncated_approx(self):
        """Truncated SVD should be close to original if W is low-rank."""
        # Create a genuinely low-rank matrix
        d_in, d_out = 50, 80
        true_rank = 10
        U = torch.randn(d_out, true_rank)
        V = torch.randn(d_in, true_rank)
        W = U @ V.t()  # rank 10

        decomposer = SVDDecomposer(rank_factor=0.3)  # should get rank > true_rank
        mod = decomposer.decompose_weight(W, "test")
        # With rank >= true_rank, reconstruction should be exact
        assert mod.rank >= true_rank

        W_recovered = mod.proj_out.weight @ mod.proj_in.weight
        assert torch.allclose(W_recovered, W, atol=1e-4)

    def test_randomized_svd_transposes_v_for_wide_matrix(self, monkeypatch):
        """torch.svd_lowrank returns V, not V^T, for large wide weights."""
        out_f, in_f = 2049, 2050
        decomposer = SVDDecomposer(rank_factor=0.001)
        target_rank = max(1, int(min(out_f, in_f) * decomposer.rank_factor))

        def fake_svd_lowrank(weight, q, niter):
            assert weight.shape == (out_f, in_f)
            return (
                torch.randn(out_f, q),
                torch.ones(q),
                torch.randn(in_f, q),
            )

        monkeypatch.setattr(torch, "svd_lowrank", fake_svd_lowrank)
        mod = decomposer.decompose_weight(torch.empty(out_f, in_f), "wide")

        assert mod.proj_in.weight.shape == (target_rank, in_f)
        assert mod.proj_out.weight.shape == (out_f, target_rank)


class TestSVDCompressorSliceWeight:
    """Verify SVDCompressor weight slicing logic."""

    @pytest.fixture
    def arch(self):
        return create_dummy_arch()

    @pytest.fixture
    def scorer(self, arch):
        scorer = SVDChannelScorer(arch)
        # Fake scores: linearly decreasing
        scorer.channel_scores = torch.arange(arch.hidden_size, 0, -1).float()
        return scorer

    @pytest.fixture
    def compressor(self, arch, scorer):
        return SVDCompressor(arch, scorer=scorer)

    @pytest.fixture
    def fwd_map(self, arch, compressor):
        return compressor._build_channel_maps()[0]

    def test_q_proj_slicing(self, compressor, fwd_map, arch):
        """q_proj [nh*hd, d] → [target_nh*hd, target_d]."""
        nh, hd = arch.num_attention_heads, arch.head_dim
        d = arch.hidden_size
        W = torch.randn(nh * hd, d)
        result = compressor._slice_weight(W, "model.layers.0.self_attn.q_proj.weight", fwd_map)
        assert result.shape == (arch.target_num_heads * hd, arch.target_embed_dim)

    def test_o_proj_slicing(self, compressor, fwd_map, arch):
        """o_proj [d, nh*hd] → [target_d, target_nh*hd]."""
        nh, hd = arch.num_attention_heads, arch.head_dim
        d = arch.hidden_size
        W = torch.randn(d, nh * hd)
        result = compressor._slice_weight(W, "model.layers.0.self_attn.o_proj.weight", fwd_map)
        assert result.shape == (arch.target_embed_dim, arch.target_num_heads * hd)

    def test_k_proj_slicing(self, compressor, fwd_map, arch):
        """k_proj [nk*hd, d] → [target_nk*hd, target_d]."""
        nk, hd = arch.num_kv_heads, arch.head_dim
        d = arch.hidden_size
        W = torch.randn(nk * hd, d)
        result = compressor._slice_weight(W, "model.layers.0.self_attn.k_proj.weight", fwd_map)
        assert result.shape == (arch.target_num_kv_heads * hd, arch.target_embed_dim)

    def test_gate_proj_slicing(self, compressor, fwd_map, arch):
        """gate_proj [ff, d] → [target_ff, target_d]."""
        ff = arch.intermediate_size
        d = arch.hidden_size
        W = torch.randn(ff, d)
        result = compressor._slice_weight(W, "model.layers.0.mlp.gate_proj.weight", fwd_map)
        assert result.shape == (arch.target_intermediate_size, arch.target_embed_dim)

    def test_down_proj_slicing(self, compressor, fwd_map, arch):
        """down_proj [d, ff] → [target_d, target_ff]."""
        ff = arch.intermediate_size
        d = arch.hidden_size
        W = torch.randn(d, ff)
        result = compressor._slice_weight(W, "model.layers.0.mlp.down_proj.weight", fwd_map)
        assert result.shape == (arch.target_embed_dim, arch.target_intermediate_size)

    def test_embed_slicing(self, compressor, fwd_map, arch):
        """embed [V, d] → [V, target_d]."""
        d = arch.hidden_size
        V = arch.vocab_size
        W = torch.randn(V, d)
        result = compressor._slice_weight(W, "model.embed_tokens.weight", fwd_map)
        assert result.shape == (V, arch.target_embed_dim)

    def test_norm_slicing(self, compressor, fwd_map, arch):
        """norm [d] → [target_d]."""
        d = arch.hidden_size
        w = torch.randn(d)
        result = compressor._slice_weight(w, "model.norm.weight", fwd_map)
        assert result.shape == (arch.target_embed_dim,)

    def test_skip_unknown_key(self, compressor, fwd_map):
        """Unknown keys are returned unchanged."""
        W = torch.randn(10, 20)
        result = compressor._slice_weight(W, "some.random.key", fwd_map)
        assert torch.equal(result, W)

    def test_head_norm_unchanged(self, compressor, fwd_map, arch):
        """q_norm/k_norm are not sliced."""
        hd = arch.head_dim
        w = torch.randn(hd)
        result = compressor._slice_weight(w, "model.layers.0.self_attn.q_norm.weight", fwd_map)
        assert result.shape == (hd,)
        assert torch.equal(result, w)


class TestSVDChannelScorer:
    """Verify channel scoring logic."""

    def test_score_channels_flattens_batch_and_sequence(self, monkeypatch):
        arch = create_dummy_arch(
            hidden_size=8, num_heads=4, num_kv=2, head_dim=2,
            intermediate=16, num_layers=2,
        )
        scorer = SVDChannelScorer(arch)
        batch_output = torch.randn(2, 3, arch.hidden_size)

        def fake_collect_layer_outputs(model, input_ids, layer_indices=None):
            return [[batch_output] for _ in layer_indices]

        monkeypatch.setattr(
            "draft_adapter.compress_svd.collect_layer_outputs",
            fake_collect_layer_outputs,
        )

        scores = scorer.score_channels(object(), torch.ones(2, 3, dtype=torch.long))

        assert scores.shape == (arch.hidden_size,)
        assert torch.isfinite(scores).all()

    def test_get_channel_order_descending(self):
        arch = create_dummy_arch()
        scorer = SVDChannelScorer(arch)
        # Set known scores
        scorer.channel_scores = torch.tensor([0.1, 0.5, 0.3, 0.9])
        order = scorer.get_channel_order()
        assert order == [3, 1, 2, 0]  # descending

    def test_score_channels_requires_model(self):
        arch = create_dummy_arch()
        scorer = SVDChannelScorer(arch)
        with pytest.raises(RuntimeError, match="Call score_channels"):
            scorer.get_channel_order()


class TestConfig:
    """Verify config changes for rank_factor."""

    def test_rank_factor_default(self):
        from draft_adapter.config import WidthConfig
        cfg = WidthConfig()
        assert cfg.rank_factor == 0.5

    def test_rank_factor_valid(self):
        from draft_adapter.config import WidthConfig
        cfg = WidthConfig(rank_factor=0.3)
        assert cfg.rank_factor == 0.3

    def test_rank_factor_zero_raises(self):
        from draft_adapter.config import WidthConfig
        with pytest.raises(ValueError):
            WidthConfig(rank_factor=0.0)

    def test_rank_factor_over_one_raises(self):
        from draft_adapter.config import WidthConfig
        with pytest.raises(ValueError):
            WidthConfig(rank_factor=1.5)
