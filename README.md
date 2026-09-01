# Draft-Adapter

Draft-Adapter is a small, measurement-first project for speculative
decoding. The current supported path is deliberately narrow:

```text
independent Hugging Face CausalLM draft
        ↓
exact autoregressive speculative decoding
        ↓
acceptance + end-to-end throughput measurement
```

## Install

Three commands usually get you there:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install <matching-torch-wheel>
python -m pip install -e .
```

Use the official PyTorch install command for your backend (CPU, CUDA, or
ROCm) in place of `<matching-torch-wheel>`. Optional extras:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"
python -m pip install -e ".[bench]"
```

The first benchmark run may still spend most of its time downloading model
weights from Hugging Face.

The repository previously contained SliceGPT, Swift-SVD, structural layer
pruning, EAGLE3, Medusa and MTP prototypes. Those branches did not produce a
reliable, portable speedup on the target setup, so they have been removed
from the active codebase. In particular, an acceptance rate without a wall
clock speedup is not considered a successful draft.

## What works today

- Load any compatible target and independent draft through
  `AutoModelForCausalLM`.
- Run on CPU or CUDA/ROCm with `--device auto`; the current benchmark requires
  the target and draft to have the same vocabulary size.
- Run exact greedy or rejection-sampling speculative decoding with persistent
  KV caches.
- Report acceptance rate, verified tokens, rounds and tokens/second.
- Keep the draft as a normal HF CausalLM, so it remains usable by standard
  speculative-decoding runtimes such as vLLM and llama.cpp-compatible
  workflows.
- Retain two standalone research scripts for reproducing the existing
  teacher-self-distillation and greedy prefix-gated distillation baselines.

The current on-policy training path is deliberately aligned with exact greedy
speculative decoding: a student proposes a deterministic block, and loss is
computed only up to the first teacher/student mismatch. Its primary reports
are mean committed prefix length, target-only throughput, speculative
throughput, and their speedup. Per-token agreement is retained only as a
diagnostic.

## Benchmark

```bash
draft-adapter \
  --target Qwen/Qwen3-1.7B \
  --draft ./draft_qwen \
  --prompt "Explain speculative decoding in one paragraph." \
  --prompt "用中文解释 KV cache。" \
  --max-new-tokens 128 \
  --speculative-tokens 5 \
  --temperature 0
```

The benchmark reports acceptance and throughput separately. A high acceptance
rate can still be slower when target and draft compete for the same device;
both numbers must be evaluated.

## Current baseline

The best validated local baseline is an independently trained Qwen3 draft
with approximately 86–87% aggregate acceptance under greedy decoding. On the
shared CPU/accelerator setup it did not yet provide end-to-end acceleration,
which is why the next milestone is runtime-aware rather than another
compression recipe.

## Next milestone: vocabulary alignment

The next implementation will focus on a model-agnostic vocabulary bridge:

1. map tokens between different tokenizer vocabularies;
2. initialize the bridge from existing embeddings/logits;
3. train only a small adapter, ideally with zero target examples;
4. optionally perform a very short online calibration pass;
5. optimize for accepted tokens per target forward, not language-model
   perplexity alone.

The target is roughly 90% greedy acceptance and 1.5–2× end-to-end speedup,
but those are goals to validate, not claims about the current release.

An initial offline converter is now available for experiments:

```bash
python scripts/unified_vocab_distill.py \
  --target <target-model-or-cache-path> \
  --student <small-causal-lm-or-cache-path> \
  --output ./runs/unified-draft \
  --analyze-only
```

It uses the target tokenizer as the canonical vocabulary, initializes exact
and decomposable rows from the student tokenizer, keeps special tokens as
trainable anchors, and exports a standard HF CausalLM checkpoint. The
four-stage training path is intentionally experimental and must be measured
on the target hardware before being treated as a usable draft.

## Deliberately out of scope

This repository does not currently ship structural width/depth compression,
SVD factorization, EAGLE3 sidecars, Medusa heads, native MTP stitching, a
calibration-data pipeline, or a custom inference runtime. Those approaches
require separate, verified runtime support and will only return if they
demonstrate a portable speedup.

The scripts `scripts/teacher_self_distill.py` and
`scripts/on_policy_acceptance_distill.py` are retained as reproducible
baselines, not as the public model-construction API.
