"""Round-trip tests for the SVD-hybrid Qwen3 loading class."""

import json

import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from draft_adapter.compress_svd import SVDCompressor, SVDDecomposer
from draft_adapter.export import export_svd_hybrid_to_hf
from draft_adapter.inspect import ModelArchitecture


class _TokenizerStub:
    vocab_size = 64

    def save_pretrained(self, output_dir):
        with open(output_dir + "/tokenizer_config.json", "w") as handle:
            json.dump({}, handle)


def _tiny_qwen3():
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    arch = ModelArchitecture(
        model_type="qwen3",
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        max_position_embeddings=64,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=False,
        target_embed_dim=32,
        target_head_dim=8,
        target_num_heads=4,
        target_num_kv_heads=2,
        target_intermediate_size=64,
        target_num_layers=2,
    )
    compressor = SVDCompressor(
        arch,
        decomposer=SVDDecomposer(rank_factor=0.5),
    )
    return compressor.decompose_model(model), arch


def test_svd_hybrid_round_trip_uses_custom_class(tmp_path):
    model, arch = _tiny_qwen3()
    export_svd_hybrid_to_hf(model, arch, _TokenizerStub(), str(tmp_path))

    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    assert loaded.__class__.__name__ == "DraftQwen3ForCausalLM"
    q_proj = loaded.model.layers[0].self_attn.q_proj
    assert q_proj.__class__.__name__ == "DecomposedLinear"
    assert q_proj.rank == 16
    assert q_proj.proj_in.weight.shape == (16, 32)
    inputs = torch.tensor([[1, 2, 3]])
    output = loaded(inputs, use_cache=True)
    assert output.logits.shape == (1, 3, 64)
    assert output.past_key_values is not None
    generated = loaded.generate(inputs, max_new_tokens=2, do_sample=False)
    assert generated.shape == (1, 5)
