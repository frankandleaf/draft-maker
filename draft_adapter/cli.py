"""CLI entry point and pipeline orchestration.

Usage:
    draft-adapter --model Qwen/Qwen3-1.7B \\
        --hd 0.75 --hs 0.75 --es 0.5 --ls 0.75 \\
        --output ./draft_model --distill
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoConfig

from . import __version__
from .calibration import compute_global_covariance
from .compress import WidthCompressor, verify_residual_consistency
from .config import DepthConfig, DistillConfig, PipelineConfig, WidthConfig
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

    # Width compression (hd, hs, es)
    p.add_argument("--hd", type=float, default=0.5,
                   help="Head dim multiplier (0-1]")
    p.add_argument("--hs", type=float, default=0.5,
                   help="Head count multiplier (0-1]")
    p.add_argument("--es", type=float, default=0.5,
                   help="Embed size multiplier (0-1]")
    p.add_argument("--calibration-samples", type=int, default=16,
                   help="Number of calibration sequences for PCA")

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
    )
    print(f"  Calibration data: {calib_ids.shape[0]} seqs x "
          f"{calib_ids.shape[1]} tokens")

    # Compute global covariance (Swift-SVD style incremental aggregation)
    print("  Computing global covariance matrix...")
    aggregator = compute_global_covariance(
        model, calib_ids,
        chunk_size=4,
    )
    covariance = aggregator.compute()
    print(f"  Covariance matrix: {covariance.shape} (conditioning...)")

    # ============================================================
    # STEP 3/6: Width compression (SliceGPT)
    # ============================================================
    print("\n[3/6] Width compression (global SliceGPT)...")
    compressor = WidthCompressor(arch, covariance=covariance)
    compressor.compute_projection()

    compressed_model, _ = compressor.compress(model, original_config)
    compressed_model = compressed_model.to(config.device)
    params = count_parameters(compressed_model)
    print(f"  Compressed model: {format_param_count(params['total'])} parameters")

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
    # STEP 5/6: Distillation (optional)
    # ============================================================
    final_model = pruned_model

    if config.distill and not config.skip_distill:
        print(f"\n[5/6] Distillation (on-policy top-K KL, "
              f"mode={config.distill.kl_mode})...")

        # Load training data
        train_ids = load_calibration_data(
            tokenizer,
            num_samples=config.distill.num_train_prompts,
            seq_len=config.distill.max_seq_len,
            device=config.device,
        )

        # Teacher is the original model (already loaded)
        print(f"  Teacher: {format_param_count(count_parameters(model)['total'])} "
              f"(frozen, inference_mode)")
        print(f"  Student: {format_param_count(params['total'])} (training)")

        trainer = DistillationTrainer(
            teacher=model,
            student=pruned_model,
            tokenizer=tokenizer,
            config=config.distill,
        )
        final_model = trainer.train(train_ids)
        print("  Distillation complete.")

    elif config.distill and config.skip_distill:
        print("\n[5/6] Distillation: skipped (--skip-distill).")
    else:
        print("\n[5/6] Distillation: skipped (use --distill to enable).")

    # Clean up teacher to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # STEP 6/6: Export to HF format
    # ============================================================
    print("\n[6/6] Exporting draft model to HF format...")
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
        head_dim_factor=args.hd,
        head_size_factor=args.hs,
        embed_size_factor=args.es,
        calibration_samples=args.calibration_samples,
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
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
