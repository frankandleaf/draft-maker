"""Export compressed model in standard HuggingFace format.

Outputs config.json + model.safetensors + tokenizer files compatible
with vLLM's speculative_model parameter.
"""

import copy
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from .inspect import ModelArchitecture
from .modeling_draft import DecomposedLinear


def export_to_hf(
    model: torch.nn.Module,
    arch: ModelArchitecture,
    tokenizer,
    output_dir: str,
) -> None:
    """Export the compressed model in HF format.

    Creates:
      output_dir/
        config.json          — Model config (uses model's own config)
        model.safetensors     — Compressed weights
        tokenizer.json        — Tokenizer
        tokenizer_config.json — Tokenizer config

    Args:
        model: Compressed HF model (config must be already correct).
        arch: ModelArchitecture with target_* fields set (for logging).
        tokenizer: HF tokenizer.
        output_dir: Output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    t = arch

    # ---- 1. Save config (use model's own config, already correct) ----
    model.config.save_pretrained(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    print(f"  Saved config to {config_path}")

    # ---- 2. Save model weights ----
    state_dict = _prepare_state_dict(model)
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, weights_path)
    print(f"  Saved weights to {weights_path}")

    # ---- 3. Copy tokenizer files ----
    _copy_tokenizer(tokenizer, output_dir)
    print(f"  Copied tokenizer to {output_dir}")

    # ---- 4. Verify round-trip ----
    _verify_export(output_dir, tokenizer)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Draft model exported: {param_count/1e6:.1f}M parameters")
    print(f"  Architecture: {t.model_type}, {t.target_num_layers} layers, "
          f"{t.target_embed_dim}d, {t.target_num_heads}H/{t.target_num_kv_heads}KV, "
           f"{t.target_head_dim}hd ")


def export_svd_hybrid_to_hf(
    model: torch.nn.Module,
    arch: ModelArchitecture,
    tokenizer,
    output_dir: str,
) -> None:
    """Export a factorized Qwen3 model with its custom loading class."""
    if arch.model_type != "qwen3":
        raise ValueError(
            "SVD-hybrid custom export currently supports only Qwen3, got "
            f"{arch.model_type}"
        )

    rank_map = {
        name: module.rank
        for name, module in model.named_modules()
        if isinstance(module, DecomposedLinear)
    }
    if not rank_map:
        raise ValueError("SVD-hybrid export requires decomposed projections")

    os.makedirs(output_dir, exist_ok=True)
    config = copy.deepcopy(model.config)
    config.architectures = ["DraftQwen3ForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_draft.DraftQwen3ForCausalLM",
    }
    config.svd_rank_map = rank_map
    config._draft_adapter = {
        "method": "svd-hybrid",
        "original_model_type": arch.model_type,
    }
    config.save_pretrained(output_dir)

    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(_prepare_state_dict(model), weights_path)

    source_path = os.path.join(os.path.dirname(__file__), "modeling_draft.py")
    shutil.copyfile(source_path, os.path.join(output_dir, "modeling_draft.py"))
    _copy_tokenizer(tokenizer, output_dir)
    _verify_export(output_dir, tokenizer, expected_class="DraftQwen3ForCausalLM")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  SVD-hybrid draft model exported: {param_count/1e6:.1f}M parameters")


def _build_export_config(original_config, arch: ModelArchitecture):
    """Create config dict with compressed dimensions.

    Deep-copies the original config and overrides compressed fields.
    """
    t = arch
    config = copy.deepcopy(original_config)

    config.hidden_size = t.target_embed_dim
    config.num_hidden_layers = t.target_num_layers
    # Sync layer_types (Qwen3 sliding window) with new layer count
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = config.layer_types[:t.target_num_layers]
    config.num_attention_heads = t.target_num_heads
    config.num_key_value_heads = t.target_num_kv_heads
    config.head_dim = t.target_head_dim
    config.intermediate_size = t.target_intermediate_size

    # Ensure embeddings are untied after compression
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False

    return config


def _prepare_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Convert model state dict to CPU float16/bfloat16 safetensors format."""
    state_dict = {}
    for key, tensor in model.state_dict().items():
        # Move to CPU, keep dtype
        state_dict[key] = tensor.cpu().contiguous()
    return state_dict


def _copy_tokenizer(tokenizer, output_dir: str) -> None:
    """Copy tokenizer files to output directory."""
    # Save tokenizer directly
    tokenizer.save_pretrained(output_dir)


def _verify_export(
    output_dir: str,
    tokenizer,
    expected_class: str | None = None,
) -> None:
    """Verify the exported model can be loaded and produces output."""
    try:
        loaded = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        loaded.eval()
        if expected_class is not None:
            assert loaded.__class__.__name__ == expected_class, (
                f"loaded {loaded.__class__.__name__}, expected {expected_class}"
            )

        # Simple forward pass
        dummy = torch.randint(0, min(tokenizer.vocab_size, 1000), (1, 8))
        with torch.no_grad():
            output = loaded(dummy)

        assert output.logits.shape[0] == 1
        assert output.logits.shape[1] == 8
        model_vocab = output.logits.shape[2]
        tok_vocab = tokenizer.vocab_size
        if model_vocab != tok_vocab:
            print(f"  Note: model vocab ({model_vocab}) != tokenizer vocab ({tok_vocab}), "
                  f"this is normal for some models")
        print(f"  Round-trip verification passed (logits: {list(output.logits.shape)})")

    except Exception as e:
        raise RuntimeError(
            f"Exported model failed round-trip verification: {e}"
        ) from e
