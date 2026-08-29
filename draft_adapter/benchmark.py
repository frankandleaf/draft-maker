"""Native autoregressive speculative-decoding benchmark.

The draft model proposes tokens one at a time using its own KV cache.  The
target model verifies a whole proposed block in one batched forward pass.
That target pass is not an implementation accident: exact speculative
decoding needs the target distribution at every proposed position.  The
optimization is that K positions are verified together, replacing K separate
target decoding steps with one forward pass.

This module intentionally keeps the only supported path as a standard
independent Hugging Face ``AutoModelForCausalLM`` draft.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` and fail clearly for an unavailable CUDA device."""
    if device in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device!r}, but torch.cuda.is_available() is False"
        )
    return str(resolved)


def _load_model(model_id: str, device: str):
    """Load a regular HF CausalLM on one device.

    The benchmark intentionally does not depend on a custom model class or
    sharding strategy.  ROCm exposes its accelerator through ``cuda`` in
    PyTorch, while CPU runs use float32 for broad operator compatibility.
    """
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except (ValueError, KeyError, ImportError):
        # Gemma 4 IT is packaged as a unified text/vision conditional model,
        # but its text path exposes the same logits/cache interface needed by
        # this benchmark.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    return model.to(device).eval()


def _vocab_size(model) -> int | None:
    value = getattr(getattr(model, "config", None), "vocab_size", None)
    return int(value) if value is not None else None


def _empty_result() -> dict:
    return {
        "generated_text": "",
        "total_time": 0.0,
        "tokens_generated": 0,
        "tokens_accepted": 0,
        "acceptance_rate": 0.0,
        "tokens_verified": 0,
        "num_rounds": 0,
        "tokens_per_second": 0.0,
    }


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
    """Run exact speculative decoding with persistent KV caches.

    In greedy mode a proposed token is accepted iff it equals the target
    argmax.  For ``temperature > 0`` this uses the standard rejection-sampling
    correction, so the returned sequence has the same distribution as the
    target model.  Both caches are cropped after a rejection; the prompt is
    never recomputed.
    """
    target_model.eval()
    draft_model.eval()
    device = _resolve_device(device)

    if max_new_tokens <= 0:
        return _empty_result()
    K = int(num_speculative_tokens)
    if K <= 0:
        raise ValueError("num_speculative_tokens must be positive")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prefix_len = input_ids.shape[1]
    prompt_len = prefix_len

    # Prefill each model exactly once.  Qwen3 uses cache_position together
    # with DynamicCache; this also works with regular HF cache implementations.
    prompt_mask = torch.ones_like(input_ids)
    prompt_pos = torch.arange(prompt_len, device=device)
    target_state = target_model(
        input_ids,
        attention_mask=prompt_mask,
        use_cache=True,
        cache_position=prompt_pos,
    )
    draft_state = draft_model(
        input_ids,
        attention_mask=prompt_mask,
        use_cache=True,
        cache_position=prompt_pos,
    )
    target_cache = target_state.past_key_values
    draft_cache = draft_state.past_key_values
    target_next = target_state.logits[:, -1]
    draft_next = draft_state.logits[:, -1]

    def append_one(model, cache, token, position):
        """Advance one model by one committed token without recomputing prefix."""
        next_len = int(position) + 1
        mask = torch.ones((1, next_len), dtype=torch.long, device=device)
        output = model(
            token.view(1, 1),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position], device=device),
        )
        return output.past_key_values, output.logits[:, -1]

    tokens_generated = 0
    tokens_accepted = 0
    tokens_verified = 0
    num_rounds = 0
    start_time = time.perf_counter()

    while tokens_generated < max_new_tokens:
        num_rounds += 1
        round_start = input_ids.shape[1]
        block_len = min(K, max_new_tokens - tokens_generated)

        proposed = []
        proposed_logits = []
        # The draft remains genuinely autoregressive: each proposal consumes
        # the previous proposal through its own cache.
        for offset in range(block_len):
            proposed_logits.append(draft_next)
            if temperature <= 0:
                token = draft_next.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(draft_next.float() / temperature, dim=-1)
                token = torch.multinomial(probs, 1)
            proposed.append(token)
            draft_cache, draft_next = append_one(
                draft_model,
                draft_cache,
                token,
                round_start + offset,
            )
        draft_block = torch.cat(proposed, dim=1)

        # One target forward verifies all K positions in parallel.  This is
        # the essential speculative-decoding speedup; skipping this full target
        # computation would no longer guarantee target-equivalent output.
        verify_mask = torch.ones(
            (1, round_start + block_len),
            dtype=torch.long,
            device=device,
        )
        verify_pos = torch.arange(
            round_start,
            round_start + block_len,
            device=device,
        )
        target_verify = target_model(
            draft_block,
            attention_mask=verify_mask,
            past_key_values=target_cache,
            use_cache=True,
            cache_position=verify_pos,
        )
        # target_next predicts the first proposed token; each returned logit
        # then predicts the next token in the proposed block.
        target_logits = torch.cat(
            [target_next.unsqueeze(1), target_verify.logits[:, :-1]],
            dim=1,
        )

        accepted = 0
        verified = 0
        replacement = None
        for index in range(block_len):
            verified += 1
            t_logits = target_logits[:, index]
            draft_token = draft_block[:, index]

            if temperature <= 0:
                target_token = t_logits.argmax(dim=-1)
                if draft_token.item() == target_token.item():
                    accepted += 1
                    continue
                replacement = target_token
                break

            t_probs = torch.softmax(t_logits.float() / temperature, dim=-1)
            d_probs = torch.softmax(
                proposed_logits[index].float() / temperature,
                dim=-1,
            )
            p_t = t_probs.gather(-1, draft_token.view(1, 1)).item()
            p_d = d_probs.gather(-1, draft_token.view(1, 1)).item()
            accept_prob = min(1.0, p_t / p_d) if p_d > 0 else 0.0
            if torch.rand((), device=device).item() < accept_prob:
                accepted += 1
                continue

            residual = torch.clamp(t_probs - d_probs, min=0)
            residual_sum = residual.sum()
            replacement = (
                torch.multinomial(residual / residual_sum, 1).view(-1)
                if residual_sum.item() > 0
                else t_probs.argmax(dim=-1)
            )
            break

        if replacement is None:
            # Every draft token survived.  target_verify already contains the
            # correct cache for the committed block.
            target_cache = target_verify.past_key_values
            target_next = target_verify.logits[:, -1]
            input_ids = torch.cat([input_ids, draft_block], dim=1)
            tokens_generated += block_len
            tokens_accepted += accepted
            tokens_verified += verified
        else:
            # Drop target/draft states for the uncommitted suffix and append
            # the target replacement token to both caches.
            committed_len = round_start + accepted
            target_cache.crop(committed_len)
            draft_cache.crop(committed_len)
            target_cache, target_next = append_one(
                target_model,
                target_cache,
                replacement,
                committed_len,
            )
            draft_cache, draft_next = append_one(
                draft_model,
                draft_cache,
                replacement,
                committed_len,
            )
            committed = torch.cat(
                [draft_block[:, :accepted], replacement.view(1, 1)],
                dim=1,
            )
            input_ids = torch.cat([input_ids, committed], dim=1)
            tokens_generated += committed.shape[1]
            tokens_accepted += accepted
            tokens_verified += verified

        if input_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    total_time = time.perf_counter() - start_time
    return {
        "generated_text": tokenizer.decode(
            input_ids[0, prefix_len:],
            skip_special_tokens=True,
        ),
        "total_time": total_time,
        "tokens_generated": tokens_generated,
        "tokens_accepted": tokens_accepted,
        "acceptance_rate": tokens_accepted / max(tokens_verified, 1),
        "tokens_verified": tokens_verified,
        "num_rounds": num_rounds,
        "tokens_per_second": (
            tokens_generated / total_time if total_time > 0 else 0.0
        ),
    }


# Backward-compatible name used by earlier scripts.
speculative_generate_kvcache = speculative_generate


def benchmark_speculative(
    target_model_id: str,
    draft_model_path: str,
    prompts: list[str],
    max_new_tokens: int = 128,
    num_speculative_tokens: int = 5,
    temperature: float = 0.0,
    device: str = "auto",
) -> dict:
    """Compare an independent draft model against a target model."""
    device = _resolve_device(device)
    target = _load_model(target_model_id, device)
    draft = _load_model(draft_model_path, device)
    target_vocab = _vocab_size(target)
    draft_vocab = _vocab_size(draft)
    if (
        target_vocab is not None
        and draft_vocab is not None
        and target_vocab != draft_vocab
    ):
        raise ValueError(
            "Target and draft must currently use the same vocabulary size "
            f"(target={target_vocab}, draft={draft_vocab}). "
            "Vocabulary alignment is a future research milestone."
        )
    tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Target model: "
        f"{sum(p.numel() for p in target.parameters()) / 1e9:.2f}B params"
    )
    print(
        f"Draft model: "
        f"{sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M params"
    )
    print(
        f"Prompts: {len(prompts)}, Spec tokens: {num_speculative_tokens}, "
        f"Temp: {temperature}, Max new: {max_new_tokens}\n"
    )

    results = []
    total_accepted = 0
    total_verified = 0
    total_rounds = 0
    total_time = 0.0
    total_tokens = 0

    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt[:60]}...")
        result = speculative_generate(
            target,
            draft,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            num_speculative_tokens=num_speculative_tokens,
            temperature=temperature,
            device=device,
        )
        results.append(result)
        total_accepted += result["tokens_accepted"]
        total_verified += result["tokens_verified"]
        total_rounds += result["num_rounds"]
        total_time += result["total_time"]
        total_tokens += result["tokens_generated"]
        print(
            f"  Speed: {result['tokens_per_second']:.1f} tok/s, "
            f"Accept: {result['acceptance_rate']:.1%}, "
            f"Generated: {result['tokens_generated']} tokens\n"
        )

    avg_acceptance = total_accepted / max(total_verified, 1)
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0
    print("=" * 60)
    print(f"Summary over {len(prompts)} prompts:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Total rounds: {total_rounds}")
    print(f"  Acceptance rate: {avg_acceptance:.1%}")
    print(f"  Average throughput: {avg_tps:.1f} tok/s")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)

    return {
        "acceptance_rate": avg_acceptance,
        "tokens_per_second": avg_tps,
        "total_tokens": total_tokens,
        "total_rounds": total_rounds,
        "results": results,
    }
