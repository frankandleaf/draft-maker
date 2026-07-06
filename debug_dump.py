"""Dump activation statistics at each stage of the compressed model forward pass.

Pinpoints where the representation collapses.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
TARGET_ID = "Qwen/Qwen3-1.7B"
DRAFT_PATH = "./draft_qwen"
PROMPT = "你好，请介绍一下你自己"


def hook_outputs(model, name):
    """Register hooks that print statistics of each layer's output."""
    handles = []

    # Embedding
    def emb_hook(m, inp, out):
        print(f"  [{name}] embed_tokens: shape={out.shape}, "
              f"mean={out.float().mean():.4f}, std={out.float().std():.4f}, "
              f"norm_rms={out.float().pow(2).mean(-1).sqrt().mean():.4f}")
    handles.append(model.model.embed_tokens.register_forward_hook(emb_hook))

    # Each layer
    for i, layer in enumerate(model.model.layers):
        # Input layernorm
        def make_ln_hook(idx):
            def hook(m, inp, out):
                print(f"  [{name}] L{idx} input_ln: shape={out.shape}, "
                      f"mean={out.float().mean():.4f}, std={out.float().std():.4f}")
            return hook
        handles.append(layer.input_layernorm.register_forward_hook(make_ln_hook(i)))

        # Attention output
        def make_attn_hook(idx):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                print(f"  [{name}] L{idx} attn_out: shape={o.shape}, "
                      f"mean={o.float().mean():.4f}, std={o.float().std():.4f}")
            return hook
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))

        # Post-attention residual
        def make_post_attn_hook(idx):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                print(f"  [{name}] L{idx} post_attn: shape={o.shape}, "
                      f"mean={o.float().mean():.4f}, std={o.float().std():.4f}")
            return hook
        # Hook on the layer itself (post-attention residual)
        handles.append(layer.register_forward_hook(make_post_attn_hook(i)))

        if i >= 2:  # Only first 3 layers
            break

    # Final norm
    def final_norm_hook(m, inp, out):
        print(f"  [{name}] final_norm: shape={out.shape}, "
              f"mean={out.float().mean():.4f}, std={out.float().std():.4f}")
    handles.append(model.model.norm.register_forward_hook(final_norm_hook))

    # LM head
    def lm_head_hook(m, inp, out):
        print(f"  [{name}] logits: shape={out.shape}, "
              f"mean={out.float().mean():.4f}, std={out.float().std():.4f}, "
              f"min={out.float().min():.4f}, max={out.float().max():.4f}")
    handles.append(model.lm_head.register_forward_hook(lm_head_hook))

    return handles


print("Loading models...")
target = AutoModelForCausalLM.from_pretrained(
    TARGET_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
tok = AutoTokenizer.from_pretrained(DRAFT_PATH)
target.eval(); draft.eval()

inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
print(f"\nPrompt: {PROMPT}")
print(f"Input shape: {inputs.input_ids.shape}")
print(f"Target dims: d={target.config.hidden_size}, heads={target.config.num_attention_heads}, "
      f"kv={target.config.num_key_value_heads}, hd={target.config.head_dim}")
print(f"Draft dims:  d={draft.config.hidden_size}, heads={draft.config.num_attention_heads}, "
      f"kv={draft.config.num_key_value_heads}, hd={draft.config.head_dim}")

print("\n" + "=" * 60)
print("TARGET MODEL ACTIVATIONS")
print("=" * 60)
t_handles = hook_outputs(target, "TGT")
with torch.no_grad():
    target(**inputs)
for h in t_handles:
    h.remove()

print("\n" + "=" * 60)
print("DRAFT MODEL ACTIVATIONS")
print("=" * 60)
d_handles = hook_outputs(draft, "DRF")
with torch.no_grad():
    draft(**inputs)
for h in d_handles:
    h.remove()

# Also compare embedding lookup directly
print("\n" + "=" * 60)
print("EMBEDDING COMPARISON")
print("=" * 60)
t_emb = target.model.embed_tokens(inputs.input_ids)
d_emb = draft.model.embed_tokens(inputs.input_ids)
# Project target embedding into draft space for comparison
# Can't directly compare since dimensions differ
print(f"Target emb: shape={t_emb.shape}, norm_mean={t_emb.float().norm(dim=-1).mean():.4f}")
print(f"Draft  emb: shape={d_emb.shape}, norm_mean={d_emb.float().norm(dim=-1).mean():.4f}")

# Check pairwise cosine similarity of embeddings
print("\nToken embedding pairwise similarity (first 100 tokens):")
t_emb_sub = t_emb[0, :10].float()  # [10, 2048]
d_emb_sub = d_emb[0, :10].float()  # [10, 1152]

t_sim = F.cosine_similarity(t_emb_sub.unsqueeze(1), t_emb_sub.unsqueeze(0), dim=-1)
d_sim = F.cosine_similarity(d_emb_sub.unsqueeze(1), d_emb_sub.unsqueeze(0), dim=-1)
print(f"Target embedding inter-token cos_sim: mean={t_sim.mean():.4f}, "
      f"min={t_sim.min():.4f}, max={t_sim.max():.4f}")
print(f"Draft  embedding inter-token cos_sim: mean={d_sim.mean():.4f}, "
      f"min={d_sim.min():.4f}, max={d_sim.max():.4f}")
if d_sim.mean() > 0.99:
    print("  ⚠️  DRAFT EMBEDDING COLLAPSED — all tokens look identical!")
