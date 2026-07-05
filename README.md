# Draft-Adapter

Turn standard Hugging Face LLMs into vLLM-compatible draft models for speculative decoding — with a single command.

Given any GQA decoder model (Llama, Qwen3, Mistral, Gemma2), Draft-Adapter automatically produces a compact draft model via width compression, depth pruning, and on-policy distillation. The output is a standard HF-format model that plugs directly into vLLM's `speculative_model` parameter.

Default factors (hd=0.5, hs=0.5, es=0.5, ls=0.75) produce a draft ~10% the size of the original.

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
| `--hd` | (0, 1] | Head dimension multiplier |
| `--hs` | (0, 1] | Head count multiplier (affects both Q and KV heads) |
| `--es` | (0, 1] | Embed dimension multiplier |
| `--ls` | (0, 1] | Layer count multiplier |
| `--distill` | flag | Enable on-policy distillation |
| `--kl-mode` | reverse/forward/tvd | KL divergence type for distillation |

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

## License

MIT
