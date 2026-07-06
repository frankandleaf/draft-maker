"""Slice-only compression test: no PCA rotation, just structured weight slicing.

This isolates whether the model collapse is caused by PCA projection
or by something else (config mismatch, tied embeddings, head slicing, etc.)
"""
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen3-1.7B"

print("Loading original model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model.eval()

# Target dimensions (same as hd=0.75 hs=0.75 es=0.75)
orig = model.config
target_hd = int(orig.head_dim * 0.75)       # 128 → 96
target_nh = int(orig.num_attention_heads * 0.75)  # 16 → 12
target_nk = int(orig.num_key_value_heads * 0.75)  # 8 → 6
target_d = target_nh * target_hd                   # 1152
target_ff = int(orig.intermediate_size * 0.75)     # 6144 → 4608
# Keep all layers for this test
target_nl = orig.num_hidden_layers                # 28

print(f"Original: {orig.hidden_size}d, {orig.num_attention_heads}H/{orig.num_key_value_heads}KV, "
      f"{orig.head_dim}hd, {orig.intermediate_size}FFN, {orig.num_hidden_layers}L")
print(f"Target:   {target_d}d, {target_nh}H/{target_nk}KV, "
      f"{target_hd}hd, {target_ff}FFN, {target_nl}L")

# Build sliced state dict
src = model.state_dict()
dst = {}

for key, w in src.items():
    if "layernorm" in key and "weight" in key:
        # RMSNorm: slice to target_d
        dst[key] = w[:target_d].clone()
    elif "embed_tokens.weight" in key:
        # Embedding: slice columns to target_d
        dst[key] = w[:, :target_d].clone()
    elif "lm_head.weight" in key:
        dst[key] = w[:, :target_d].clone()
    elif "q_proj.weight" in key:
        nh, hd = orig.num_attention_heads, orig.head_dim
        # [nh*hd, hidden] → [nh, hd, hidden] → slice → flatten
        w3 = w.reshape(nh, hd, -1)
        w3 = w3[:target_nh, :target_hd, :target_d]
        dst[key] = w3.reshape(target_nh * target_hd, target_d).clone()
    elif "k_proj.weight" in key or "v_proj.weight" in key:
        nk, hd = orig.num_key_value_heads, orig.head_dim
        w3 = w.reshape(nk, hd, -1)
        w3 = w3[:target_nk, :target_hd, :target_d]
        dst[key] = w3.reshape(target_nk * target_hd, target_d).clone()
    elif "o_proj.weight" in key:
        nh, hd = orig.num_attention_heads, orig.head_dim
        # [hidden, nh*hd] → [hidden, nh, hd] → slice
        w3 = w.reshape(-1, nh, hd)
        w3 = w3[:target_d, :target_nh, :target_hd]
        dst[key] = w3.reshape(target_d, target_nh * target_hd).clone()
    elif "gate_proj.weight" in key or "up_proj.weight" in key:
        # [intermediate, hidden]
        dst[key] = w[:target_ff, :target_d].clone()
    elif "down_proj.weight" in key:
        # [hidden, intermediate]
        dst[key] = w[:target_d, :target_ff].clone()
    elif "q_norm.weight" in key or "k_norm.weight" in key:
        dst[key] = w[:target_hd].clone()
    elif "bias" in key:
        # Skip bias — handle later if needed
        pass
    else:
        # Copy unchanged (rotary_emb etc.)
        dst[key] = w.clone()

# Create compressed model
cfg = copy.deepcopy(model.config)
cfg.hidden_size = target_d
cfg.num_attention_heads = target_nh
cfg.num_key_value_heads = target_nk
cfg.head_dim = target_hd
cfg.intermediate_size = target_ff
cfg.tie_word_embeddings = False

print("\nCreating slice-only model...")
compressed = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
compressed = compressed.to(DEVICE)

missing, unexpected = compressed.load_state_dict(dst, strict=False)
if missing:
    print(f"Missing: {missing[:5]}...")
if unexpected:
    print(f"Unexpected: {unexpected[:5]}...")

compressed.eval()

# --- Test generation ---
prompts = ["你好，请介绍一下你自己", "The capital of France is", "1 + 1 =",
           "def fibonacci(n):", "量子计算机的基本工作原理是"]

print("\n" + "=" * 60)
print("GENERATION TEST (slice-only, no PCA)")
print("=" * 60)
for prompt in prompts:
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = compressed.generate(**inputs, max_new_tokens=30, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {text[:200]}")
