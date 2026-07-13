"""Tests for importance-aware GQA and FFN pruning."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from draft_adapter.compress_ffn import SwiftSVDCompressor
from draft_adapter.inspect import ModelArchitecture


def _arch():
    return ModelArchitecture(
        model_type="qwen3",
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        target_embed_dim=32,
        target_head_dim=8,
        target_num_heads=2,
        target_num_kv_heads=1,
        target_intermediate_size=32,
        target_num_layers=1,
    )


def test_attention_selection_keeps_complete_gqa_group():
    compressor = SwiftSVDCompressor(_arch())
    compressor.head_importance[0] = torch.tensor([0.1, 0.2, 0.8, 0.9])
    compressor.kv_importance[0] = torch.tensor([0.1, 0.9])

    q_indices, kv_indices = compressor._selected_attention_indices(
        0, torch.device("cpu"),
    )

    assert kv_indices.tolist() == [1]
    assert q_indices.tolist() == [2, 3]


def test_prune_preserves_independent_lm_head():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(1.0)
        model.lm_head.weight.fill_(2.0)

    compressor = SwiftSVDCompressor(_arch())
    compressor.head_importance[0] = torch.tensor([0.1, 0.2, 0.8, 0.9])
    compressor.kv_importance[0] = torch.tensor([0.1, 0.9])
    compressor.ffn_importance[0] = torch.arange(64).float()

    pruned, _ = compressor.prune(model, config)

    assert torch.equal(pruned.lm_head.weight, torch.full_like(pruned.lm_head.weight, 2.0))
    assert not torch.equal(pruned.lm_head.weight, pruned.model.embed_tokens.weight)
    assert pruned(torch.tensor([[1, 2, 3]])).logits.shape == (1, 3, 32)
