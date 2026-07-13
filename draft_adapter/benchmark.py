"""Native speculative decoding benchmark — no vLLM dependency.

Verifies the draft model by running actual speculative decoding:
  1. Draft model proposes K tokens autoregressively (fast, small model)
  2. Target model verifies all K tokens in one forward pass (slow, big model)
  3. Count accepted tokens → acceptance rate

Reference:
  Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
"""

import time

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _validate_draft_model(draft_model, require_svd_hybrid: bool = False) -> None:
    metadata = getattr(draft_model.config, "_draft_adapter", {})
    is_svd_hybrid = metadata.get("method") == "svd-hybrid"
    if require_svd_hybrid and not is_svd_hybrid:
        raise RuntimeError(
            "Draft model is not an SVD-hybrid export; regenerate it with "
            "--method svd-hybrid"
        )
    if is_svd_hybrid and draft_model.__class__.__name__ != "DraftQwen3ForCausalLM":
        raise RuntimeError(
            "SVD-hybrid draft loaded with "
            f"{draft_model.__class__.__name__}; expected DraftQwen3ForCausalLM"
        )


@torch.no_grad()
def speculative_generate(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    num_speculative_tokens: int = 5,
    temperature: float = 1.0,
    device: str = "cuda",
) -> dict:
    """Run speculative decoding with a target and draft model.

    Algorithm:
      1. Run draft model K steps (autoregressive) to propose K tokens
      2. Run target model once to verify all K tokens in parallel
      3. Accept tokens greedily or via rejection sampling
      4. Repeat until max_new_tokens reached

    Args:
        target_model: Large target model (teacher).
        draft_model: Small draft model (student).
        tokenizer: Shared tokenizer.
        prompt: Input text.
        max_new_tokens: Maximum new tokens to generate.
        num_speculative_tokens: K — tokens proposed per draft round.
        temperature: Sampling temperature (>0 for sampling, <=0 for greedy).
        device: Torch device.

    Returns:
        Dict with: generated_text, total_time, tokens_generated,
                   tokens_accepted, acceptance_rate, num_rounds
    """
    target_model.eval()
    draft_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids  # [1, seq_len]
    prefix_len = input_ids.shape[1]

    tokens_generated = 0
    tokens_accepted = 0
    num_rounds = 0
    start_time = time.time()

    pbar = tqdm(total=max_new_tokens, desc="Speculative decoding", unit="tok")
    K = num_speculative_tokens

    while tokens_generated < max_new_tokens:
        num_rounds += 1
        current_len = input_ids.shape[1]

        # ---- Phase 1: Draft model generates K tokens ----
        draft_output = draft_model.generate(
            input_ids,
            max_new_tokens=K,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        draft_ids = draft_output.sequences  # [1, current_len + K]
        draft_new_ids = draft_ids[0, current_len:]  # [K]

        # ---- Phase 2: Target model verifies all K tokens in one pass ----
        target_output = target_model(draft_ids)
        target_logits = target_output.logits  # [1, current_len+K, vocab]

        # Verification: for position p in [current_len, current_len+K-1],
        # target_logits[0, p-1] gives logits for predicting position p
        # Verify token at position p using target logits at p-1
        K_actual = draft_new_ids.shape[0]
        accepted = 0

        for i in range(K_actual):
            pos = current_len + i  # current position in full sequence
            logits_at_pos = target_logits[0, pos - 1]  # logits predicting position pos

            if temperature <= 0:
                # Greedy: accept if draft matches target argmax
                target_token = logits_at_pos.argmax(dim=-1).item()
                draft_token = draft_new_ids[i].item()
                if draft_token == target_token:
                    accepted += 1
                    input_ids = torch.cat([input_ids, draft_new_ids[i:i+1].unsqueeze(0)], dim=1)
                else:
                    # Reject: use target token instead
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[target_token]], device=device)
                    ], dim=1)
                    accepted += 0  # draft token was rejected
                    break  # Stop verification chain
            else:
                # Rejection sampling (Leviathan et al. 2023, Algorithm 2)
                probs_t = torch.softmax(logits_at_pos / temperature, dim=-1)
                probs_d = torch.softmax(
                    draft_output.scores[i][0] / temperature, dim=-1
                )
                draft_token = draft_new_ids[i].item()

                # Accept with probability min(1, p_t(x) / p_d(x))
                p_t = probs_t[draft_token].item()
                p_d = probs_d[draft_token].item()
                accept_prob = min(1.0, p_t / p_d) if p_d > 0 else 0.0

                if torch.rand(1).item() < accept_prob:
                    accepted += 1
                    input_ids = torch.cat([input_ids, draft_new_ids[i:i+1].unsqueeze(0)], dim=1)
                else:
                    # Reject: sample from residual distribution
                    residual = torch.clamp(probs_t - probs_d, min=0)
                    residual_sum = residual.sum()
                    if residual_sum > 0:
                        residual = residual / residual_sum
                        target_token = torch.multinomial(residual, 1).item()
                    else:
                        target_token = probs_t.argmax().item()
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[target_token]], device=device)
                    ], dim=1)
                    break

        round_generated = input_ids.shape[1] - current_len
        tokens_generated += round_generated
        tokens_accepted += accepted
        pbar.update(round_generated)

        # Check EOS
        if input_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    pbar.close()
    total_time = time.time() - start_time
    generated_text = tokenizer.decode(
        input_ids[0, prefix_len:], skip_special_tokens=True
    )

    return {
        "generated_text": generated_text,
        "total_time": total_time,
        "tokens_generated": tokens_generated,
        "tokens_accepted": tokens_accepted,
        "acceptance_rate": tokens_accepted / max(num_rounds * num_speculative_tokens, 1),
        "num_rounds": num_rounds,
        "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0,
    }


def benchmark_speculative(
    target_model_id: str,
    draft_model_path: str,
    prompts: list[str],
    max_new_tokens: int = 128,
    num_speculative_tokens: int = 5,
    temperature: float = 0.0,
    device: str = "cuda",
    require_svd_hybrid: bool = False,
) -> dict:
    """Run speculative decoding benchmark comparing target-only vs speculative.

    Returns:
        Dict with aggregated results and per-prompt details.
    """
    target = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    _validate_draft_model(draft, require_svd_hybrid=require_svd_hybrid)
    tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Target model: {sum(p.numel() for p in target.parameters())/1e9:.2f}B params")
    print(f"Draft model:  {sum(p.numel() for p in draft.parameters())/1e6:.1f}M params")
    print(f"Prompts: {len(prompts)}, Spec tokens: {num_speculative_tokens}, "
          f"Temp: {temperature}, Max new: {max_new_tokens}\n")

    all_results = []
    total_accepted = 0
    total_rounds = 0
    total_time = 0
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
        result = speculative_generate(
            target, draft, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            num_speculative_tokens=num_speculative_tokens,
            temperature=temperature,
            device=device,
        )
        all_results.append(result)
        total_accepted += result["tokens_accepted"]
        total_rounds += result["num_rounds"]
        total_time += result["total_time"]
        total_tokens += result["tokens_generated"]

        print(f"  Speed: {result['tokens_per_second']:.1f} tok/s, "
              f"Accept: {result['acceptance_rate']:.1%}, "
              f"Generated: {result['tokens_generated']} tokens\n")

    avg_acceptance = total_accepted / max(total_rounds * num_speculative_tokens, 1)
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    print(f"{'='*60}")
    print(f"Summary over {len(prompts)} prompts:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Acceptance rate: {avg_acceptance:.1%}")
    print(f"  Average throughput: {avg_tps:.1f} tok/s")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    return {
        "acceptance_rate": avg_acceptance,
        "tokens_per_second": avg_tps,
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
    }
