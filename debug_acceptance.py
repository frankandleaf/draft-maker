"""Diagnose why draft model has 0% acceptance rate in speculative decoding.

Checks:
  1. Draft model standalone generation quality
  2. Token-level argmax match between draft and target
  3. Logit distribution similarity (KL divergence)
  4. Layer-by-layer hidden state drift
  5. Whether the issue is pre-existing (pre-distill) or caused by distillation
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
TARGET_ID = "Qwen/Qwen3-1.7B"
DRAFT_PATH = "./draft_qwen"
PROMPTS = [
    "你好，请介绍一下你自己",
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "量子计算机的基本工作原理是",
]


def load_models():
    print("=" * 60)
    print("Loading models...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
        trust_remote_code=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    tok = AutoTokenizer.from_pretrained(DRAFT_PATH)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    target.eval()
    draft.eval()

    t_params = sum(p.numel() for p in target.parameters()) / 1e9
    d_params = sum(p.numel() for p in draft.parameters()) / 1e6
    print(f"Target: {t_params:.2f}B, Draft: {d_params:.1f}M")
    return target, draft, tok


# ---- Test 1: Standalone generation quality ----
def test_generation(target, draft, tok):
    print("\n" + "=" * 60)
    print("TEST 1: Standalone generation quality")
    print("=" * 60)

    for prompt in PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            t_gen = target.generate(**inputs, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            d_gen = draft.generate(**inputs, max_new_tokens=30, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
        t_text = tok.decode(t_gen[0], skip_special_tokens=True)
        d_text = tok.decode(d_gen[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Target: {t_text[:200]}")
        print(f"Draft:  {d_text[:200]}")
        print(f"Match:  {t_text == d_text}")


# ---- Test 2: Token-level argmax comparison ----
def test_argmax_match(target, draft, tok):
    print("\n" + "=" * 60)
    print("TEST 2: Token-level argmax comparison")
    print("=" * 60)

    for prompt in PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            t_out = target(**inputs)
            d_out = draft(**inputs)

        t_logits = t_out.logits[0]  # [seq, vocab]
        d_logits = d_out.logits[0]

        total = t_logits.shape[0]
        match = 0
        top5_overlap = 0
        top10_overlap = 0

        for pos in range(total):
            t_top1 = t_logits[pos].argmax().item()
            d_top1 = d_logits[pos].argmax().item()
            if t_top1 == d_top1:
                match += 1

            t_top5 = set(t_logits[pos].topk(5).indices.tolist())
            d_top5 = set(d_logits[pos].topk(5).indices.tolist())
            top5_overlap += len(t_top5 & d_top5)

            t_top10 = set(t_logits[pos].topk(10).indices.tolist())
            d_top10 = set(d_logits[pos].topk(10).indices.tolist())
            top10_overlap += len(t_top10 & d_top10)

        print(f"\nPrompt: {prompt}")
        print(f"  Argmax match: {match}/{total} = {match/total*100:.1f}%")
        print(f"  Avg top-5 overlap: {top5_overlap/total:.1f}/5")
        print(f"  Avg top-10 overlap: {top10_overlap/total:.1f}/10")

        # Show a concrete comparison at the last position
        pos = total - 1
        print(f"  Last position ({pos}):")
        t_top5 = t_logits[pos].topk(5)
        d_top5 = d_logits[pos].topk(5)
        print(f"    Target top-3: {[(tok.decode([t]), f'{v:.1f}') for t,v in zip(t_top5.indices[:3], t_top5.values[:3])]}")
        print(f"    Draft  top-3: {[(tok.decode([t]), f'{v:.1f}') for t,v in zip(d_top5.indices[:3], d_top5.values[:3])]}")


# ---- Test 3: KL divergence between distributions ----
def test_kl_divergence(target, draft, tok):
    print("\n" + "=" * 60)
    print("TEST 3: Distribution similarity (KL divergence)")
    print("=" * 60)

    for prompt in PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            t_out = target(**inputs)
            d_out = draft(**inputs)

        t_logits = t_out.logits[0]
        d_logits = d_out.logits[0]

        t_probs = F.softmax(t_logits, dim=-1)
        d_probs = F.softmax(d_logits, dim=-1)
        # Add epsilon to avoid log(0)
        eps = 1e-10
        t_probs = t_probs.clamp(min=eps)
        d_probs = d_probs.clamp(min=eps)

        # KL(t || d) and KL(d || t)
        kl_td = (t_probs * (t_probs.log() - d_probs.log())).sum(dim=-1).mean().item()
        kl_dt = (d_probs * (d_probs.log() - t_probs.log())).sum(dim=-1).mean().item()

        # Top-10 KL (only on top 10 teacher tokens)
        _, top10_idx = t_logits.topk(10, dim=-1)
        t_top10 = t_probs.gather(-1, top10_idx)
        d_top10 = d_probs.gather(-1, top10_idx)
        t_top10 = t_top10 / t_top10.sum(dim=-1, keepdim=True)
        d_top10 = d_top10 / d_top10.sum(dim=-1, keepdim=True)
        kl_top10 = (t_top10 * (t_top10.log() - d_top10.log())).sum(dim=-1).mean().item()

        print(f"\nPrompt: {prompt}")
        print(f"  KL(target || draft): {kl_td:.4f}")
        print(f"  KL(draft || target): {kl_dt:.4f}")
        print(f"  Top-10 KL(target || draft): {kl_top10:.4f}")


# ---- Test 4: Hidden state drift per layer ----
def test_hidden_state_drift(target, draft, tok):
    print("\n" + "=" * 60)
    print("TEST 4: Per-layer hidden state cosine similarity")
    print("=" * 60)

    prompt = PROMPTS[0]
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        t_out = target(**inputs, output_hidden_states=True)
        d_out = draft(**inputs, output_hidden_states=True)

    t_hs = t_out.hidden_states  # list of [1, seq, d_target]
    d_hs = d_out.hidden_states  # list of [1, seq, d_draft]

    print(f"Target layers: {len(t_hs)}, Draft layers: {len(d_hs)}")
    print(f"Target hidden dim: {t_hs[0].shape[-1]}, Draft hidden dim: {d_hs[0].shape[-1]}")

    # Compare embedding layer
    t_emb = t_hs[0][0, -1].float()
    d_emb = d_hs[0][0, -1].float()
    # Can't directly compare diff-dim vectors — compare via PCA projection space
    # Instead, check norm ratio
    print(f"\nEmbedding layer:")
    print(f"  Target norm: {t_emb.norm().item():.4f}")
    print(f"  Draft  norm: {d_emb.norm().item():.4f}")

    # Compare logits at final layer
    print(f"\nLogit statistics:")
    t_logits = t_out.logits[0, -1].float()
    d_logits = d_out.logits[0, -1].float()
    print(f"  Target logits: mean={t_logits.mean():.4f}, std={t_logits.std():.4f}, "
          f"min={t_logits.min():.4f}, max={t_logits.max():.4f}")
    print(f"  Draft  logits: mean={d_logits.mean():.4f}, std={d_logits.std():.4f}, "
          f"min={d_logits.min():.4f}, max={d_logits.max():.4f}")
    print(f"  NaN in target: {torch.isnan(t_logits).any().item()}")
    print(f"  NaN in draft:  {torch.isnan(d_logits).any().item()}")
    print(f"  Inf in target: {torch.isinf(t_logits).any().item()}")
    print(f"  Inf in draft:  {torch.isinf(d_logits).any().item()}")


# ---- Test 5: Interactive speculative decoding trace ----
def test_speculative_trace(target, draft, tok):
    print("\n" + "=" * 60)
    print("TEST 5: Speculative decoding step-by-step trace")
    print("=" * 60)

    prompt = PROMPTS[0]
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    K = 5  # num_speculative_tokens

    for round_num in range(3):
        print(f"\n--- Round {round_num+1} ---")
        current_len = input_ids.shape[1]

        # Draft generates K tokens
        with torch.no_grad():
            d_out = draft.generate(
                input_ids, max_new_tokens=K, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
        d_new = d_out.sequences[0, current_len:]  # [K]
        print(f"  Draft proposes {len(d_new)} tokens: {tok.decode(d_new.tolist())}")

        # Target verifies
        with torch.no_grad():
            t_out = target(d_out.sequences)
        t_logits = t_out.logits[0]  # [len, vocab]

        for i in range(len(d_new)):
            pos = current_len + i
            t_token = t_logits[pos - 1].argmax().item()
            d_token = d_new[i].item()
            match = t_token == d_token
            print(f"    pos {pos}: draft={d_token}({tok.decode([d_token])}) "
                  f"target={t_token}({tok.decode([t_token])}) "
                  f"{'ACCEPT' if match else 'REJECT'}")

            if match:
                input_ids = torch.cat([input_ids, d_new[i:i+1].unsqueeze(0)], dim=1)
            else:
                # Use target token instead
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[t_token]], device=DEVICE)
                ], dim=1)
                print(f"    -> using target token, breaking chain")
                break

        print(f"  Current text: {tok.decode(input_ids[0], skip_special_tokens=True)[:200]}")
        if input_ids.shape[1] >= inputs.input_ids.shape[1] + 30:
            break


if __name__ == "__main__":
    target, draft, tok = load_models()
    test_generation(target, draft, tok)
    test_argmax_match(target, draft, tok)
    test_kl_divergence(target, draft, tok)
    test_hidden_state_drift(target, draft, tok)
    test_speculative_trace(target, draft, tok)
