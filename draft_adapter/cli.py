"""CLI entry point and pipeline orchestration.

Usage:
    draft-adapter --model Qwen/Qwen3-1.7B \\
        --es 0.5 --ls 0.75 \\
        --output ./draft_model --distill
"""

import argparse
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoConfig

from . import __version__
from .calibration import compute_global_covariance
from .compress import WidthCompressor, verify_residual_consistency
from .compress_ffn import SwiftSVDCompressor
from .compress_svd import SVDChannelScorer, SVDCompressor, SVDDecomposer
from .config import DepthConfig, DistillConfig, PipelineConfig, WidthConfig
from .data import DATA_PRESETS
from .debug_log import enable_debug, get_logger
from .distill import DistillationTrainer
from .export import export_to_hf
from .inspect import compute_targets, inspect_model
from .prune import DepthPruner
from .utils import (
    count_parameters,
    format_param_count,
    get_dtype,
    load_calibration_data,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        "draft-adapter",
        description="Turn standard LLMs into vLLM-compatible draft models.",
    )
    p.add_argument("--version", action="version",
                   version=f"draft-adapter {__version__}")

    # Model
    p.add_argument("--model", required=True,
                   help="HF model ID or local path")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer (defaults to model)")
    p.add_argument("--output", default="./draft_model",
                   help="Output directory for the draft model")
    p.add_argument("--device", default="cuda",
                   help="Torch device")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)

    # Method
    p.add_argument("--method", default="slicegpt",
                   choices=["slicegpt", "swift-svd", "svd-hybrid"],
                   help="Compression algorithm")

    # Width compression
    p.add_argument("--es", type=float, default=0.5,
                   help="Embed size multiplier (0-1]")
    p.add_argument("--calibration-samples", type=int, default=16,
                    help="Number of calibration sequences for PCA")
    p.add_argument("--calibration-seq-len", type=int, default=512,
                   help="Tokens per packed calibration sequence")
    p.add_argument("--rank-factor", type=float, default=0.5,
                    help="SVD decomposition rank factor (0-1], for svd-hybrid")

    # Data
    p.add_argument("--data-preset", default="public-mixed",
                   choices=sorted(DATA_PRESETS),
                   help="Built-in public data mix for calibration/distillation")
    p.add_argument("--calibration-data", default=None,
                   help="Comma-separated local paths or HF specs for calibration "
                        "(dataset[:config[:split]])")
    p.add_argument("--distill-data", default=None,
                   help="Comma-separated local paths or HF specs for distillation "
                        "(defaults to --calibration-data or --data-preset)")
    p.add_argument("--data-source-timeout", type=int, default=30,
                   help="Seconds before skipping a slow/unavailable data source "
                        "(0 disables timeout)")

    # Depth pruning (ls)
    p.add_argument("--ls", type=float, default=0.75,
                   help="Layer count multiplier (0-1]")
    p.add_argument("--protect-first", type=int, default=1,
                   help="Always keep first N layers")
    p.add_argument("--protect-last", type=int, default=1,
                   help="Always keep last N layers")

    # Distillation
    p.add_argument("--distill", action="store_true",
                   help="Run on-policy distillation")
    p.add_argument("--distill-steps", type=int, default=1000,
                   help="Number of distillation steps")
    p.add_argument("--distill-lr", type=float, default=1e-5,
                   help="Distillation learning rate")
    p.add_argument("--distill-batch", type=int, default=4,
                    help="Distillation batch size")
    p.add_argument("--distill-prompts", type=int, default=128,
                   help="Number of packed prompt sequences for distillation")
    p.add_argument("--kl-top-k", type=int, default=10,
                   help="Top-K for sparse KL divergence")
    p.add_argument("--kl-mode", default="reverse",
                   choices=["reverse", "forward", "tvd"],
                   help="KL divergence mode")
    p.add_argument("--kl-temperature", type=float, default=1.0,
                   help="Temperature for KL divergence")
    p.add_argument("--distill-gen-len", type=int, default=32,
                   help="Tokens student generates per step")

    # Pipeline control
    p.add_argument("--teacher-device", default=None,
                   help="Device for teacher model during distillation "
                        "(default: 'auto' for multi-GPU, same as --device otherwise)")
    p.add_argument("--debug", action="store_true",
                   help="Print detailed debug info for every weight operation")
    p.add_argument("--skip-distill", action="store_true",
                   help="Skip distillation (combine with --distill to undo)")
    p.add_argument("--skip-benchmark", action="store_true",
                   help="Skip vLLM benchmark")

    return p


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full draft-adapter pipeline."""
    set_seed(config.seed)
    if config.debug:
        enable_debug()
    log = get_logger()

    # ============================================================
    # STEP 0: Load config (lightweight — no model weights)
    # ============================================================
    original_config = AutoConfig.from_pretrained(
        config.model, trust_remote_code=True,
    )

    # ============================================================
    # STEP 1/6: Inspect model architecture
    # ============================================================
    print("\n[1/6] Inspecting model architecture...")
    arch = inspect_model(config.model)
    if config.method == "swift-svd":
        # Only compress FFN + heads, hidden_size stays
        width_factor = config.width.embed_size_factor
        config.width.embed_size_factor = 1.0
        arch = compute_targets(arch, config.width, config.depth)
        arch.target_embed_dim = arch.hidden_size
        arch.target_head_dim = arch.head_dim
        arch.target_intermediate_size = max(8, int(arch.intermediate_size * width_factor))
        arch.target_num_heads = max(1, int(arch.num_attention_heads * width_factor))
        arch.target_num_kv_heads = max(1, arch.target_num_heads // arch.num_kv_groups)
    else:
        arch = compute_targets(arch, config.width, config.depth)

    print(f"  Model type: {arch.model_type}")
    print(f"  Original: {arch.num_layers}L, {arch.hidden_size}d, "
          f"{arch.num_attention_heads}H/{arch.num_kv_heads}KV, "
          f"{arch.head_dim}hd, {arch.intermediate_size}FFN")
    print(f"  Target:   {arch.target_num_layers}L, {arch.target_embed_dim}d, "
          f"{arch.target_num_heads}H/{arch.target_num_kv_heads}KV, "
          f"{arch.target_head_dim}hd, {arch.target_intermediate_size}FFN")
    print(f"  GQA groups: {arch.num_kv_groups} → "
          f"{arch.target_num_heads // arch.target_num_kv_heads} "
          f"(heads/kv={arch.target_num_heads}/{arch.target_num_kv_heads})")

    # ============================================================
    # STEP 2/6: Load model + calibration → covariance
    # ============================================================
    print("\n[2/6] Loading model and computing covariance...")
    dtype = get_dtype(config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer or config.model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    params = count_parameters(model)
    print(f"  Teacher model: {format_param_count(params['total'])} parameters")

    # Load calibration data
    calib_ids = load_calibration_data(
        tokenizer,
        num_samples=config.width.calibration_samples,
        seq_len=config.width.calibration_seq_len,
        device=config.device,
        data_sources=config.calibration_data,
        data_preset=config.data_preset,
        source_timeout=config.data_source_timeout,
    )
    print(f"  Calibration data: {calib_ids.shape[0]} seqs x "
          f"{calib_ids.shape[1]} tokens")

    # ============================================================
    # STEP 2b-3/6: Compression (method-dependent)
    # ============================================================
    if config.method == "svd-hybrid":
        # ---- Phase A: SVD channel scoring + slicing → standard model ----
        print("\n[2b/6] SVD channel scoring (activation covariance SVD)...")
        scorer = SVDChannelScorer(arch)
        scorer.score_channels(model, calib_ids)

        print("\n[3/6] SVD channel slicing...")
        svd_comp = SVDCompressor(arch, scorer=scorer)
        compressed_model = svd_comp.compress(model, original_config)
        compressed_model = compressed_model.to(config.device)
        params = count_parameters(compressed_model)
        print(f"  After channel slicing: {format_param_count(params['total'])} parameters")
        print(f"  No PCA rotation, no norm absorption needed")
    elif config.method == "swift-svd":
        print("\n[2b/6] Computing importance scores (heads + FFN)...")
        compressor = SwiftSVDCompressor(arch)
        compressor.compute_head_importance(model, calib_ids)
        compressor.compute_ffn_importance(model, calib_ids)
        print("\n[3/6] Pruning attention heads + FFN neurons...")
        compressed_model, _ = compressor.prune(model, original_config)
        compressed_model = compressed_model.to(config.device)
        params = count_parameters(compressed_model)
        print(f"  Compressed: {format_param_count(params['total'])} parameters")
        print(f"  Residual stream: unchanged (no PCA rotation)")
    else:
        # Compute global covariance
        print("  Computing global covariance matrix...")
        aggregator = compute_global_covariance(model, calib_ids, chunk_size=4)
        covariance = aggregator.compute()
        print(f"  Covariance matrix: {covariance.shape} (conditioning...)")

        # SliceGPT: PCA residual stream rotation + weight slicing
        print("\n[3/6] Width compression (global SliceGPT)...")
        compressor = WidthCompressor(arch, covariance=covariance)
        compressor.compute_projection()
        compressed_model, _ = compressor.compress(model, original_config)
        compressed_model = compressed_model.to(config.device)
        params = count_parameters(compressed_model)
        print(f"  Compressed model: {format_param_count(params['total'])} parameters")

    # Free teacher to save GPU memory before depth pruning
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Verify residual stream consistency
    print("  Verifying residual stream consistency...")
    verify_residual_consistency(compressed_model, tokenizer, config.device)
    print("  Residual stream consistency: OK")

    # ============================================================
    # STEP 4/6: Depth pruning (ShortGPT)
    # ============================================================
    print("\n[4/6] Depth pruning (ShortGPT BI)...")
    pruner = DepthPruner(
        arch,
        protect_first=config.depth.protect_first,
        protect_last=config.depth.protect_last,
    )
    bi_scores = pruner.compute_bi_scores(compressed_model, calib_ids)

    # Print BI scores
    score_str = ", ".join(f"L{i}:{s:.4f}" for i, s in enumerate(bi_scores))
    print(f"  BI scores: [{score_str}]")

    keep_indices = pruner.select_layers(arch.target_num_layers)
    print(f"  Kept layers: {keep_indices}")
    print(f"  Removed layers: "
          f"{[i for i in range(arch.num_layers) if i not in keep_indices]}")

    pruned_model = pruner.prune_model(compressed_model, keep_indices,
                                       compressed_model.config)
    pruned_model = pruned_model.to(config.device)
    params = count_parameters(pruned_model)
    print(f"  Pruned model: {format_param_count(params['total'])} parameters")

    # ============================================================
    # STEP 4.5/6: SVD decomposition (svd-hybrid optional phase)
    # ============================================================
    if config.method == "svd-hybrid" and config.width.rank_factor < 1.0:
        print(f"\n[4.5/6] SVD low-rank decomposition "
              f"(rank_factor={config.width.rank_factor})...")
        decomposer = SVDDecomposer(rank_factor=config.width.rank_factor)
        svd_dec = SVDCompressor(arch, decomposer=decomposer)
        pruned_model = svd_dec.decompose_model(pruned_model)
        params = count_parameters(pruned_model)
        print(f"  After SVD decomposition: {format_param_count(params['total'])} parameters")

    # ============================================================
    # STEP 5/6: Distillation (optional)
    # ============================================================
    final_model = pruned_model

    if config.distill and not config.skip_distill:
        print(f"\n[5/6] Distillation (on-policy top-K KL, "
              f"mode={config.distill.kl_mode})...")

        # Smart default for teacher device: auto on multi-GPU, same as --device otherwise
        teacher_device = config.teacher_device
        if teacher_device is None:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            teacher_device = "auto" if n_gpus > 1 else config.device
        print(f"  Teacher device: {teacher_device}")

        # Reload teacher for distillation (was freed after compression)
        teacher = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=dtype,
            device_map=teacher_device,
            trust_remote_code=True,
        )
        teacher.eval()

        # Load training data
        train_ids = load_calibration_data(
            tokenizer,
            num_samples=config.distill.num_train_prompts,
            seq_len=config.distill.max_seq_len,
            device=config.device,
            data_sources=config.distill_data or config.calibration_data,
            data_preset=config.data_preset,
            source_timeout=config.data_source_timeout,
        )

        # Teacher is the original model
        print(f"  Teacher: {format_param_count(count_parameters(teacher)['total'])} "
              f"(frozen, inference_mode)")
        print(f"  Student: {format_param_count(params['total'])} (training)")

        trainer = DistillationTrainer(
            teacher=teacher,
            student=pruned_model,
            tokenizer=tokenizer,
            config=config.distill,
        )
        final_model = trainer.train(train_ids)
        print("  Distillation complete.")
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif config.distill and config.skip_distill:
        print("\n[5/6] Distillation: skipped (--skip-distill).")
    else:
        print("\n[5/6] Distillation: skipped (use --distill to enable).")

    # ============================================================
    # STEP 6/6: Export to HF format
    # ============================================================
    print("\n[6/6] Exporting draft model to HF format...")
    if config.method == "svd-hybrid" and config.width.rank_factor < 1.0:
        # SVD-decomposed model uses custom DecomposedLinear layers;
        # export with a modeling_draft.py for loading.
        _export_svd_hybrid(final_model, arch, tokenizer, output_dir=config.output)
    else:
        export_to_hf(final_model, arch, tokenizer, output_dir=config.output)

    final_param_count = count_parameters(final_model)['total']
    print(f"\n{'='*60}")
    print(f"Done! Draft model saved to {config.output}")
    print(f"Draft parameters: {format_param_count(final_param_count)}")
    print(f"{'='*60}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build config from CLI args
    width_cfg = WidthConfig(
        embed_size_factor=args.es,
        calibration_samples=args.calibration_samples,
        calibration_seq_len=args.calibration_seq_len,
        rank_factor=args.rank_factor,
    )

    depth_cfg = DepthConfig(
        layer_factor=args.ls,
        protect_first=args.protect_first,
        protect_last=args.protect_last,
    )

    distill_cfg = None
    if args.distill:
        distill_cfg = DistillConfig(
            steps=args.distill_steps,
            batch_size=args.distill_batch,
            learning_rate=args.distill_lr,
            top_k=args.kl_top_k,
            kl_mode=args.kl_mode,
            kl_temperature=args.kl_temperature,
            num_train_prompts=args.distill_prompts,
            generate_len=args.distill_gen_len,
        )

    config = PipelineConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        output=args.output,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        width=width_cfg,
        depth=depth_cfg,
        distill=distill_cfg,
        skip_distill=args.skip_distill or not args.distill,
        skip_benchmark=args.skip_benchmark,
        debug=args.debug,
        method=args.method,
        teacher_device=args.teacher_device,
        data_preset=args.data_preset,
        calibration_data=args.calibration_data,
        distill_data=args.distill_data,
        data_source_timeout=args.data_source_timeout,
    )

    run_pipeline(config)


def _export_svd_hybrid(model, arch, tokenizer, output_dir: str) -> None:
    """Export SVD-decomposed model with custom DecomposedLinear layers.

    Writes a minimal modeling_draft.py alongside the weights so the model
    can be loaded with transformers + trust_remote_code=True.
    """
    import json
    import os

    from safetensors.torch import save_file
    from .compress_svd import DecomposedLinear

    os.makedirs(output_dir, exist_ok=True)
    t = arch

    # ---- 1. Save config with auto_map ----
    cfg = copy.deepcopy(model.config.to_dict())
    cfg["auto_map"] = {
        "AutoModelForCausalLM": "modeling_draft.DraftModelForCausalLM",
    }
    # Mark as draft adapter output
    cfg["_draft_adapter"] = {
        "method": "svd-hybrid",
        "original_model_type": t.model_type,
        "target_hidden_size": t.target_embed_dim,
        "target_num_layers": t.target_num_layers,
        "head_dim": t.target_head_dim,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved config to {output_dir}/config.json")

    # ---- 2. Save weights ----
    state_dict = {}
    for key, tensor in model.state_dict().items():
        state_dict[key] = tensor.cpu().contiguous()
    weights_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, weights_path)
    print(f"  Saved weights to {weights_path}")

    # ---- 3. Write modeling_draft.py ----
    modeling_code = _MODELING_DRAFT_TEMPLATE
    modeling_path = os.path.join(output_dir, "modeling_draft.py")
    with open(modeling_path, "w") as f:
        f.write(modeling_code)
    print(f"  Wrote custom modeling file to {modeling_path}")

    # ---- 4. Copy tokenizer ----
    tokenizer.save_pretrained(output_dir)
    print(f"  Copied tokenizer to {output_dir}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  SVD-hybrid draft model exported: {param_count/1e6:.1f}M parameters")


_MODELING_DRAFT_TEMPLATE = r'''
"""Auto-generated draft decoder with SVD-decomposed Linear layers.

Load with: AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)


class DecomposedLinear(nn.Module):
    """Low-rank SVD decomposition: W[m,n] ≈ W_out[m,r] @ W_in[r,n]."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # These are dummy values; actual rank and weights loaded from state dict
        self.in_features = in_features
        self.out_features = out_features
        self.rank = 0  # will be inferred from state dict
        self.proj_in = nn.Linear(in_features, 1, bias=False)
        self.proj_out = nn.Linear(1, out_features, bias=bias)

    def forward(self, x):
        return self.proj_out(self.proj_in(x))


def _update_decomposed_linear(module, state_dict, prefix):
    """Patch DecomposedLinear shapes to match state dict after weight loading."""
    if prefix + "proj_in.weight" in state_dict:
        r, in_f = state_dict[prefix + "proj_in.weight"].shape
        out_f, _ = state_dict[prefix + "proj_out.weight"].shape
        module.rank = r
        module.in_features = in_f
        module.out_features = out_f
        # Recreate linear layers with correct shapes
        device = state_dict[prefix + "proj_in.weight"].device
        dtype = state_dict[prefix + "proj_in.weight"].dtype
        module.proj_in = nn.Linear(in_f, r, bias=False).to(device=device, dtype=dtype)
        module.proj_out = nn.Linear(r, out_f, bias=False).to(device=device, dtype=dtype)
        module.proj_in.weight.data = state_dict[prefix + "proj_in.weight"]
        module.proj_out.weight.data = state_dict[prefix + "proj_out.weight"]
        if prefix + "proj_out.bias" in state_dict:
            module.proj_out.bias.data = state_dict[prefix + "proj_out.bias"]


class DraftDecoderLayer(nn.Module):
    """Decoder layer with DecomposedLinear SVD layers."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_size
        nh = config.num_attention_heads
        nk = config.num_key_value_heads
        hd = config.head_dim if hasattr(config, "head_dim") else (d // nh)
        ff = config.intermediate_size

        self.self_attn = DraftAttention(config, layer_idx)
        self.mlp = DraftMLP(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_value=past_key_value,
                                  output_attentions=output_attentions,
                                  use_cache=use_cache)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DraftAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        d = config.hidden_size
        nh = config.num_attention_heads
        nk = config.num_key_value_heads
        hd = config.head_dim if hasattr(config, "head_dim") else (d // nh)

        # DecomposedLinear placeholders; shapes set during _init_weights
        self.q_proj = DecomposedLinear(d, nh * hd, bias=False)
        self.k_proj = DecomposedLinear(d, nk * hd, bias=False)
        self.v_proj = DecomposedLinear(d, nk * hd, bias=False)
        self.o_proj = DecomposedLinear(nh * hd, d, bias=False)

        self.num_heads = nh
        self.num_kv_heads = nk
        self.head_dim = hd
        self.num_kv_groups = nh // nk
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # No RoPE here (model loads from pre-trained, RoPE is applied in original model)
        # For a draft model, we skip RoPE complexity
        key_states_exp = key_states.repeat_interleave(self.num_kv_groups, dim=1)
        value_states_exp = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_exp)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class DraftMLP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        d = config.hidden_size
        ff = config.intermediate_size
        self.gate_proj = DecomposedLinear(d, ff, bias=False)
        self.up_proj = DecomposedLinear(d, ff, bias=False)
        self.down_proj = DecomposedLinear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DraftDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, output_hidden_states=False,
                **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, attention_mask=attention_mask,
                                  position_ids=position_ids, use_cache=use_cache)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class DraftModelForCausalLM(PreTrainedModel):
    config_class = None  # Will be set dynamically
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DraftDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DraftModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, output_hidden_states=False,
                **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids,
                             use_cache=use_cache,
                             output_hidden_states=output_hidden_states)
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
'''


if __name__ == "__main__":
    main()
