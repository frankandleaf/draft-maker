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
        model_vocab = output.logits.shape[2]
        tok_vocab = tokenizer.vocab_size
        if model_vocab != tok_vocab:
            print(f"  Note: model vocab ({model_vocab}) != tokenizer vocab ({tok_vocab}), "
                  f"this is normal for some models")
        print(f"  Round-trip verification passed (logits: {list(output.logits.shape)})")

    except Exception as e:
        import traceback
        print(f"  Warning: round-trip verification failed: {e}")
        traceback.print_exc()
        print(f"  The model files are saved but may need manual inspection.")
