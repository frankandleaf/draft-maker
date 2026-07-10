# Draft-Adapter

Turn standard Hugging Face LLMs into vLLM-compatible draft models for speculative decoding — with a single command.

Given any GQA decoder model (Llama, Qwen3, Mistral, Gemma2), Draft-Adapter automatically produces a compact draft model via width compression, depth pruning, and on-policy distillation. The output is a standard HF-format model that plugs directly into vLLM's `speculative_model` parameter.

Default factors (es=0.5, ls=0.75) produce a draft ~10% the size of the original.

## Quick Start

```bash
# Install
pip install -e .

# Generate a draft model from Qwen3-1.7B
draft-adapter --model Qwen/Qwen3-1.7B --output ./draft_qwen

# With distillation for better acceptance rate
draft-adapter --model Qwen/Qwen3-1.7B --output ./draft_qwen --distill

# Use the draft model with vLLM
python -c "
from vllm import LLM
llm = LLM(model='Qwen/Qwen3-1.7B', speculative_model='./draft_qwen')
"
```

## Pipeline

```
Original Model (HF)
    │
    ├── [1] inspect     — Architecture detection, target dimension computation
    ├── [2] calibrate   — Swift-SVD incremental covariance aggregation
    ├── [3] compress    — SliceGPT global PCA projection + width slicing
    ├── [4] prune       — ShortGPT Block Influence (BI) layer ranking
    ├── [5] distill     — DistillSpec on-policy top-K KL (optional)
    └── [6] export      — HF-format draft model (config.json + safetensors)
```

## Parameters

| Flag | Range | Description |
|------|-------|-------------|
| `--es` | (0, 1] | Embed dimension multiplier |
| `--ls` | (0, 1] | Layer count multiplier |
| `--distill` | flag | Enable on-policy distillation |
| `--kl-mode` | reverse/forward/tvd | KL divergence type for distillation |

## Calibration Data

Draft-Adapter can stream public Hugging Face datasets and pack them into fixed-length token sequences for calibration/distillation.

```bash
# Online machine: export public HF data to local JSONL first
python -m draft_adapter.data prepare \
  --data-preset public-zh-fast \
  --num-samples 1024 \
  --output ./data/calib_public_zh_fast.jsonl \
  --source-timeout 90

# Offline GPU machine: read only local JSONL, no HF network needed
draft-adapter --model Qwen/Qwen3-1.7B --method svd-hybrid \
  --calibration-data ./data/calib_public_zh_fast.jsonl \
  --calibration-samples 512 --calibration-seq-len 512 \
  --output ./draft_qwen_svd --skip-distill

# Chinese-first public data mix
draft-adapter --model Qwen/Qwen3-1.7B --method svd-hybrid \
  --data-preset public-zh-fast --calibration-samples 512 --calibration-seq-len 512 \
  --data-source-timeout 60 --output ./draft_qwen_svd --skip-distill

# Explicit public HF sources: dataset[:config[:split]]
draft-adapter --model Qwen/Qwen3-1.7B --method svd-hybrid \
  --calibration-data allenai/c4:zh:train,Salesforce/wikitext:wikitext-2-raw-v1:train \
  --calibration-samples 512 --output ./draft_qwen_svd --skip-distill

# Local JSONL/TXT replacement once you have real prompts
draft-adapter --model Qwen/Qwen3-1.7B --method svd-hybrid \
  --calibration-data ./data/calib.jsonl --distill-data ./data/distill.jsonl \
  --distill --distill-prompts 2048 --output ./draft_qwen_svd
```

Supported local rows can contain `text`, `content`, `messages`, `conversations`, or instruction-style fields such as `instruction`, `input`, and `output`.
Each public HF source is fetched in a short-lived subprocess; if a source is slow or unavailable, `--data-source-timeout` controls when it is skipped.

## Technical Details

### Width Compression (SliceGPT)

We compute a global PCA projection matrix Q from calibration data, then apply it to all weight matrices. Unlike naive dimension slicing, the PCA rotation ensures we delete only the *least important* principal components.

For weights interacting with the residual stream:
- **Both-side projection** (`q_proj`, `o_proj`): W' = Q_top.T @ W @ Q_top
- **Input-side** (`k_proj`, `v_proj`, `gate_proj`, `up_proj`): W' = W @ Q_top
- **Output-side** (`down_proj`): W' = Q_top.T @ W
- **Embeddings** (`embed_tokens`, `lm_head`): W' = W @ Q_top

### Depth Pruning (ShortGPT)

Block Influence (BI) measures how much each layer transforms hidden states:
BI_i = 1 - mean(cos_sim(X_i, X_{i+1})). Lower BI = more redundant. First and last layers are always protected.

### Distillation (DistillSpec)

On-policy distillation: the student (draft) generates tokens, the teacher scores them. Top-K sparse KL divergence prevents the student from wasting capacity on near-zero teacher logits.

## Supported Models

| Family | Model Type | Status |
|--------|------------|--------|
| Llama 2/3/4 | `llama` | Supported |
| Qwen2 / Qwen2.5 | `qwen2` | Supported |
| Qwen3 | `qwen3` | Supported |
| Mistral | `mistral` | Supported |
| Gemma 2 | `gemma2` | Supported |
| StableLM | `stablelm` | Supported |
| MoE variants | — | Not supported |
| MLA (DeepSeek) | — | Not supported |
| Mamba (SSM) | — | Not supported |

## References

- [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](https://arxiv.org/abs/2401.15024) — Ashkboos et al., ICLR 2024
- [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) — Men et al., 2024
- [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) — Zhou et al., ICLR 2024
- [AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders](https://arxiv.org/abs/2510.19779) — Hu et al., NeurIPS 2025
- [Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression](https://arxiv.org/abs/2604.01609) — Qi et al., ICML 2026
