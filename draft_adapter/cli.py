"""Small command-line surface for the currently supported baseline.

Draft generation by structural compression was removed from this package.
The next implementation target is a vocabulary-alignment adapter trained
without calibration data.  Until that adapter lands, this CLI only exposes
the reproducible speculative-decoding benchmark.
"""

from __future__ import annotations

import argparse

from . import __version__
from .benchmark import benchmark_speculative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draft-adapter",
        description=(
            "Benchmark an independent Hugging Face CausalLM draft against a "
            "target model."
        ),
    )
    parser.add_argument("--version", action="version", version=f"draft-adapter {__version__}")
    parser.add_argument("--target", required=True, help="Target model ID or local path")
    parser.add_argument("--draft", required=True, help="Independent draft model path")
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Prompt to benchmark; repeat for multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for both models: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_speculative(
        target_model_id=args.target,
        draft_model_path=args.draft,
        prompts=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_speculative_tokens=args.speculative_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
