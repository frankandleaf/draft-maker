"""Teacher-self-data distillation for an independent CausalLM draft model.

The teacher creates both the prompts and the responses.  The student is a
normal Hugging Face CausalLM and is trained only on the resulting conversations.
Teacher top-k probabilities and tail mass are cached before student training,
so long runs do not repeatedly execute the teacher.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEACHER = "/home/frank/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_STUDENT = "/home/frank/draft-adapter/models/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--output", default="./runs/teacher-self-distill")
    parser.add_argument("--records-cache", default=None)
    parser.add_argument("--reuse-records", action="store_true")
    parser.add_argument("--questions", type=int, default=512)
    parser.add_argument("--question-batch-size", type=int, default=8)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda", help="Torch device; ROCm uses 'cuda'")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def bulk_move_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Move a CPU model to ROCm with one large transfer instead of per-tensor copies.

    ROCm can spend seconds in each allocator call when a model has hundreds of
    parameters.  Keeping every parameter as a view into one transferred
    buffer preserves the normal CausalLM interface while avoiding that cost.
    """
    if device == "cpu":
        return model
    params = list(model.named_parameters())
    if not params:
        return model.to(device)
    dtype = params[0][1].dtype
    if any(param.dtype != dtype for _, param in params):
        return model.to(device)
    total = sum(param.numel() for _, param in params)
    flat_cpu = torch.empty(total, dtype=dtype)
    offset = 0
    locations = []
    for _, param in params:
        size = param.numel()
        flat_cpu[offset:offset + size].copy_(param.detach().reshape(-1))
        locations.append((param, offset, size))
        offset += size
    flat_device = flat_cpu.to(device)
    for param, start, size in locations:
        param.data = flat_device[start:start + size].view_as(param)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    return model


def chat_ids(tokenizer, user_text: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids[0]
    return torch.tensor(encoded["input_ids"], dtype=torch.long)[0]


def chat_batch(tokenizer, user_texts: list[str]) -> dict[str, torch.Tensor]:
    conversations = [[{"role": "user", "content": text}] for text in user_texts]
    encoded = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
    }


@torch.inference_mode()
def generate_text(model, tokenizer, ids: torch.Tensor, max_new_tokens: int,
                  do_sample: bool) -> tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    ids = ids.to(device)
    output = model.generate(
        ids.unsqueeze(0),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.8 if do_sample else 1.0,
        top_p=0.95 if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )[0]
    generated = output[ids.shape[0]:].cpu()
    return generated, tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int,
    do_sample: bool,
) -> list[tuple[torch.Tensor, str]]:
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.8 if do_sample else 1.0,
        top_p=0.95 if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generated = output[:, input_ids.shape[1]:].cpu()
    result = []
    for row in generated:
        if tokenizer.eos_token_id is not None:
            eos = (row == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos.numel():
                row = row[: eos[0, 0] + 1]
        result.append((row, tokenizer.decode(row, skip_special_tokens=True).strip()))
    return result


def make_questions(teacher, tokenizer, count: int, batch_size: int) -> list[str]:
    questions: list[str] = []
    categories = (
        "factual science and history",
        "mathematics and logical reasoning",
        "Python and software engineering",
        "everyday planning and practical advice",
        "explaining concepts and comparing options",
        "creative but answerable questions",
    )
    while len(questions) < count:
        prompts = []
        for offset in range(batch_size):
            category = categories[(len(questions) // 8 + offset) % len(categories)]
            prompts.append(
                "Generate 8 diverse, answerable user questions about "
                f"{category}. Return exactly one question per line, with no "
                "preamble or answers."
            )
        batch = chat_batch(tokenizer, prompts)
        generated = generate_batch(teacher, tokenizer, batch, 192, True)
        before = len(questions)
        for _, text in generated:
            for line in text.splitlines():
                line = line.strip().lstrip("- ").lstrip("0123456789.) ")
                if line.endswith("?") and len(line) >= 12 and line not in questions:
                    questions.append(line)
                    if len(questions) >= count:
                        break
            if len(questions) >= count:
                break
        if len(questions) == before:
            raise RuntimeError(f"Teacher generated no usable questions at {len(questions)}")
        print(f"Generated questions {len(questions)}/{count}")
    return questions[:count]


def make_records(teacher, tokenizer, questions: list[str], answer_tokens: int) -> list[dict]:
    records = []
    batch_size = 8
    for start in range(0, len(questions), batch_size):
        batch_questions = questions[start:start + batch_size]
        batch = chat_batch(tokenizer, batch_questions)
        generated = generate_batch(
            teacher, tokenizer, batch, answer_tokens, do_sample=True,
        )
        for question, (answer_ids, answer), input_row in zip(
            batch_questions, generated, batch["input_ids"],
        ):
            if answer_ids.numel() == 0:
                continue
            prompt_len = int((input_row != tokenizer.pad_token_id).sum().item())
            prompt_ids = input_row[-prompt_len:].clone()
            records.append({
                "question": question,
                "answer": answer,
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
            })
        print(f"Generated answers {len(records)}/{len(questions)}")
    return records


def _response_logits(logits: torch.Tensor, prompt_len: int,
                     answer_len: int) -> torch.Tensor:
    # Logits at position p-1 predict token p.
    return logits[:, prompt_len - 1: prompt_len - 1 + answer_len]


@torch.no_grad()
def cache_teacher_targets(
    teacher, records: list[dict], top_k: int,
) -> None:
    """Cache sparse teacher distributions on CPU for repeated student updates."""
    teacher.eval()
    device = next(teacher.parameters()).device
    for index, record in enumerate(records, 1):
        prompt_ids = record["prompt_ids"]
        answer_ids = record["answer_ids"]
        full_ids = torch.cat((prompt_ids, answer_ids)).to(device).unsqueeze(0)
        logits = _response_logits(
            teacher(full_ids).logits, prompt_ids.numel(), answer_ids.numel(),
        ).float()
        logp = F.log_softmax(logits, dim=-1)
        k = min(top_k, logp.shape[-1])
        top_logp, top_indices = logp.topk(k, dim=-1)
        top_probs = top_logp.exp()
        tail = (1.0 - top_probs.sum(-1)).clamp_min(1e-8)
        record["teacher_top_logp"] = top_logp.cpu()
        record["teacher_top_indices"] = top_indices.to(torch.int32).cpu()
        record["teacher_tail_logp"] = tail.log().cpu()
        if index == 1 or index % 64 == 0 or index == len(records):
            print(f"Cached teacher targets {index}/{len(records)}")


@torch.no_grad()
def evaluate(student, records: list[dict]) -> dict[str, float]:
    student.eval()
    total = correct = 0
    nll_sum = 0.0
    kl_sum = 0.0
    device = next(student.parameters()).device
    for record in records:
        prompt_ids = record["prompt_ids"]
        answer_ids = record["answer_ids"]
        full_ids = torch.cat((prompt_ids, answer_ids)).to(device).unsqueeze(0)
        s_logits = _response_logits(
            student(full_ids).logits, prompt_ids.numel(), answer_ids.numel(),
        ).float()
        target = answer_ids.to(device).unsqueeze(0)
        total += target.numel()
        top_indices = record["teacher_top_indices"].to(device).long()
        top_logp = record["teacher_top_logp"].to(device)
        correct += (s_logits.argmax(-1) == top_indices[..., 0]).sum().item()
        nll_sum += F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), target.reshape(-1)).item() * target.numel()

        s_logp = F.log_softmax(s_logits.float(), dim=-1)
        t_probs = top_logp.exp()
        s_probs = s_logp.gather(-1, top_indices).exp()
        t_tail = record["teacher_tail_logp"].to(device).exp()
        s_tail = (1.0 - s_probs.sum(-1)).clamp_min(1e-8)
        bucket_kl = (t_probs * (top_logp - s_logp.gather(-1, top_indices))).sum(-1)
        bucket_kl += t_tail * (record["teacher_tail_logp"].to(device) - s_tail.log())
        kl_sum += bucket_kl.sum().item()
    return {
        "tokens": float(total),
        "top1_agreement": correct / max(total, 1),
        "student_nll": nll_sum / max(total, 1),
        "topk_tail_kl": kl_sum / max(total, 1),
    }


def train(
    student,
    records: list[dict],
    steps: int,
    lr: float,
    warmup_steps: int,
    hard_weight: float,
    kl_weight: float,
    eval_records: list[dict],
    eval_every: int,
) -> dict[str, float]:
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.1 + 0.9 * max(0.0, 1.0 - progress)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    device = next(student.parameters()).device
    best = {"student_nll": float("inf")}
    best_state = None
    for step in range(steps):
        record = records[step % len(records)]
        prompt_ids = record["prompt_ids"]
        answer_ids = record["answer_ids"]
        full_ids = torch.cat((prompt_ids, answer_ids)).to(device).unsqueeze(0)
        s_logits = _response_logits(student(full_ids).logits, prompt_ids.numel(), answer_ids.numel()).float()
        labels = answer_ids.to(device).unsqueeze(0)
        top_indices = record["teacher_top_indices"].to(device).long()
        top_logp = record["teacher_top_logp"].to(device)
        teacher_tail_logp = record["teacher_tail_logp"].to(device)
        s_logp = F.log_softmax(s_logits, dim=-1)
        hard = F.cross_entropy(
            s_logits.reshape(-1, s_logits.shape[-1]), labels.reshape(-1),
        )
        t_probs = top_logp.exp()
        s_probs = s_logp.gather(-1, top_indices).exp()
        t_tail = teacher_tail_logp.exp()
        s_tail = (1.0 - s_probs.sum(-1)).clamp_min(1e-8)
        sparse_kl = (t_probs * (top_logp - s_logp.gather(-1, top_indices))).sum(-1)
        sparse_kl = sparse_kl + t_tail * (teacher_tail_logp - s_tail.log())
        loss = hard_weight * hard + kl_weight * sparse_kl.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(
                f"step={step + 1}/{steps} lr={scheduler.get_last_lr()[0]:.2e} "
                f"loss={loss.item():.4f} hard={hard.item():.4f} "
                f"sparse_kl={sparse_kl.mean().item():.4f}"
            )
        if eval_records and ((step + 1) % eval_every == 0 or step + 1 == steps):
            metrics = evaluate(student, eval_records)
            print(f"eval step={step + 1}: {json.dumps(metrics)}")
            if metrics["student_nll"] < best["student_nll"]:
                best = metrics
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in student.state_dict().items()
                }
    if best_state is not None:
        student.load_state_dict(best_state)
    return best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"teacher={args.teacher}")
    print(f"student={args.student}")
    print(f"device={args.device} dtype={args.dtype}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.records_cache) if args.records_cache else output / "records_cache.pt"

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student, padding_side="left")
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise RuntimeError("Teacher and student tokenizers differ; this experiment requires aligned vocabularies.")

    # ROCm is unusually slow when transformers copies hundreds of individual
    # tensors.  Load on CPU, then bind parameters to one bulk device transfer.
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=dtype, low_cpu_mem_usage=True,
    )
    teacher = bulk_move_model(teacher, args.device).eval()
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, low_cpu_mem_usage=True,
    )
    student = bulk_move_model(student, args.device).eval()
    if args.reuse_records:
        if not cache_path.exists():
            raise FileNotFoundError(f"Requested --reuse-records but cache is missing: {cache_path}")
        records = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded {len(records)} cached conversations from {cache_path}")
    else:
        questions = make_questions(
            teacher, teacher_tokenizer, args.questions, args.question_batch_size,
        )
        print("\nTeacher-generated questions:")
        for i, question in enumerate(questions, 1):
            print(f"  {i}. {question}")
        records = make_records(teacher, teacher_tokenizer, questions, args.answer_tokens)
        if not records:
            raise RuntimeError("Teacher generated no non-empty answers")
        print(f"Collected {len(records)} teacher conversations / {sum(r['answer_ids'].numel() for r in records)} response tokens")

    split = max(1, int(len(records) * (1.0 - args.val_fraction)))
    random.shuffle(records)
    train_records = records[:split]
    val_records = records[split:] or records[-1:]
    print(f"Split {len(train_records)} train / {len(val_records)} validation conversations")
    if not all("teacher_top_logp" in record for record in records):
        cache_teacher_targets(teacher, records, args.top_k)
        torch.save(records, cache_path)
        print(f"Saved reusable records and teacher targets to {cache_path}")
    before = evaluate(student, val_records)
    print("\nBefore:", json.dumps(before, ensure_ascii=False))
    best = train(
        student,
        train_records,
        args.steps,
        args.lr,
        args.warmup_steps,
        args.hard_weight,
        args.kl_weight,
        val_records,
        args.eval_every,
    )
    after = evaluate(student, val_records)
    print("After:", json.dumps(after, ensure_ascii=False))

    student.save_pretrained(output)
    student_tokenizer.save_pretrained(output)
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"question": record["question"], "answer": record["answer"]}, ensure_ascii=False) + "\n")
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"before": before, "after": after, "best": best, "args": vars(args)},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved independent student checkpoint to {output}")


if __name__ == "__main__":
    main()
