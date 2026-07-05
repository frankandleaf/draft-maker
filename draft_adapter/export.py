"""Export compressed model in standard HuggingFace format.

Outputs config.json + model.safetensors + tokenizer files compatible
with vLLM's speculative_model parameter.
"""

import copy
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from .inspect import ModelArchitecture


def export_to_hf(
    model: torch.nn.Module,
    arch: ModelArchitecture,
    tokenizer,
    original_config,
    output_dir: str,
) -> None:
    """Export the compressed model in HF format.

    Creates:
      output_dir/
        config.json          — Updated model config
        model.safetensors     — Compressed weights
        tokenizer.json        — Copied tokenizer
        tokenizer_config.json — Copied tokenizer config

    Args:
        model: Compressed HF model.
        arch: ModelArchitecture with target_* fields set.
        tokenizer: HF tokenizer.
        original_config: Original HF model config.
        output_dir: Output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    t = arch

    # ---- 1. Build config ----
    config = _build_export_config(original_config, arch)
    config_path = os.path.join(output_dir, "config.json")
    config.save_pretrained(output_dir)
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


def _build_export_config(original_config, arch: ModelArchitecture):
    """Create config dict with compressed dimensions.

    Deep-copies the original config and overrides compressed fields.
    """
    t = arch
    config = copy.deepcopy(original_config)

    config.hidden_size = t.target_embed_dim
    config.num_hidden_layers = t.target_num_layers
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


def _verify_export(output_dir: str, tokenizer) -> None:
    """Verify the exported model can be loaded and produces output."""
    try:
        loaded = AutoModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        loaded.eval()

        # Simple forward pass
        dummy = torch.randint(0, min(tokenizer.vocab_size, 1000), (1, 8))
        with torch.no_grad():
            output = loaded(dummy)

        assert output.logits.shape[0] == 1
        assert output.logits.shape[1] == 8
        assert output.logits.shape[2] == tokenizer.vocab_size
        print(f"  Round-trip verification passed (logits: {list(output.logits.shape)})")

    except Exception as e:
        print(f"  Warning: round-trip verification failed: {e}")
        print(f"  The model files are saved but may need manual inspection.")
