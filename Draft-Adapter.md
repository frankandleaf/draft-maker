# Project direction

## Current release (2026-08-29)

Draft-Adapter currently provides one reliable primitive: benchmark an
independent autoregressive CausalLM draft against a target model using exact
speculative decoding and persistent KV caches. This keeps the draft portable
and compatible with standard serving runtimes.

The old structural-compression pipeline (SliceGPT, Swift-SVD, SVD-hybrid,
ShortGPT-style pruning) has been retired. Its outputs were useful for
experiments but did not establish a repeatable end-to-end speedup. The
target-bound EAGLE3, Medusa and MTP prototypes are retired for the same
reason; they are not part of the supported API.

The scripts `scripts/teacher_self_distill.py` and
`scripts/on_policy_acceptance_distill.py` remain only as historical,
reproducible experiment baselines. They are not a public model-construction
API and are not invoked by the benchmark or CLI.

## Success criteria

Every proposed change must report all of:

- greedy acceptance rate and accepted length;
- target-only throughput;
- speculative throughput;
- speedup on the intended hardware;
- tokenizer/vocabulary compatibility;
- a standalone checkpoint format that the serving runtime can load.

Acceptance rate alone is not a success criterion.

## Next research milestone

Build a unified vocabulary adapter that can sit between a target model and a
small independent CausalLM. The initial version should use existing token
embeddings and a deterministic token mapping, require no teacher-generated
dataset, and train only a small number of parameters. The objective should
directly reward target-token agreement/accepted prefixes while preserving the
ordinary CausalLM interface.

The project goal is approximately 90% greedy acceptance and 1.5–2× measured
speedup, subject to hardware and model pair.

## Cross-vocabulary experiment

`draft_adapter.vocab` and `scripts/unified_vocab_distill.py` provide an
offline physical vocabulary-unification path. The target tokenizer is
canonical, ordinary text tokens are initialized from exact matches or exact
source-token decompositions, and special/control tokens remain explicit
trainable fallback rows. The output is a regular Hugging Face CausalLM.

On the local Qwen3-0.6B → Gemma-4-12B-it tokenizer pair (August 29, 2026),
262,144 target rows contained 96,079 exact matches, 165,372 decomposable rows,
and 693 fallback rows. This is a tokenizer-initialization statistic, not an
acceptance-rate result. The large-teacher four-stage run still requires a
separate hardware-sized training job.
