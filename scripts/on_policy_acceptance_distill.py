"""Greedy on-policy prefix distillation for an independent CausalLM draft.

The student proposes a deterministic K-token block. Only positions before the
first teacher/student greedy mismatch contribute to the loss, matching the
prefix that exact speculative decoding can actually commit.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make both the source checkout and sibling training script importable when
# this file is run directly (``python scripts/on_policy_acceptance_distill.py``).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from teacher_self_distill import bulk_move_model, chat_batch, chat_ids, generate_batch
from draft_adapter.benchmark import speculative_generate


DEFAULT_TEACHER = "/home/frank/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_STUDENT = "/home/frank/draft-adapter/runs/teacher-self-distill-formal-512-stable"
DEFAULT_RECORDS = "/home/frank/draft-adapter/runs/teacher-self-distill-formal-512-stable/records_cache.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--output", default="./runs/on-policy-acceptance-smoke")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--dynamic-prompts", action="store_true")
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--feeder-log", default=None)
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Number of independent rollouts accumulated per optimizer step.",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Deprecated; training and evaluation are always greedy.",
    )
    parser.add_argument(
        "--greedy-weight",
        type=float,
        default=0.1,
        help="Deprecated; use --prefix-hard-weight.",
    )
    parser.add_argument(
        "--first-token-weight",
        type=float,
        default=4.0,
        help="Extra multiplier for the first speculative token greedy CE.",
    )
    parser.add_argument(
        "--survival-weight",
        type=float,
        default=0.5,
        help="Deprecated; exact prefix gating replaces this surrogate.",
    )
    parser.add_argument(
        "--accept-log-weight",
        type=float,
        default=0.5,
        help="Deprecated; exact prefix gating replaces this surrogate.",
    )
    parser.add_argument(
        "--first-token-ratio-weight",
        type=float,
        default=0.0,
        help="Deprecated; greedy top-1 CE is used instead.",
    )
    parser.add_argument(
        "--kl-anchor-weight",
        type=float,
        default=0.5,
        help="Deprecated; use --prefix-kl-weight.",
    )
    parser.add_argument(
        "--sample-ce-weight",
        type=float,
        default=0.1,
        help="Deprecated; student-sample CE is intentionally disabled.",
    )
    parser.add_argument(
        "--prefix-hard-weight",
        type=float,
        default=1.0,
        help="Weight for teacher top-1 CE on the surviving prefix.",
    )
    parser.add_argument(
        "--prefix-kl-weight",
        type=float,
        default=1.0,
        help="Weight for teacher/student KL on the surviving prefix.",
    )
    parser.add_argument(
        "--throughput-new-tokens",
        type=int,
        default=64,
        help="Tokens used for the end-to-end throughput evaluation.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TeacherPromptFeeder:
    """Continuously generate diverse user questions in a background thread."""

    _categories = (
        "science and history",
        "mathematics and logical reasoning",
        "Python debugging and software engineering",
        "algorithms, databases, and systems",
        "machine learning and data analysis",
        "economics, society, and public policy",
        "health, habits, and everyday decisions",
        "travel, planning, and practical constraints",
        "creative writing, design, and music",
        "language learning and communication",
        "ethics, tradeoffs, and decision making",
    )

    def __init__(
        self,
        teacher,
        tokenizer,
        batch_size: int,
        queue_size: int,
        teacher_lock: threading.Lock,
        log_path: str | None = None,
    ) -> None:
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.questions: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self.teacher_lock = teacher_lock
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.round = 0
        self.seen: set[str] = set()
        self.log_path = Path(log_path) if log_path else None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)

    def get(self, timeout: float = 300.0) -> str:
        try:
            return self.questions.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("Dynamic teacher prompt feeder timed out") from exc

    @staticmethod
    def _parse(text: str) -> list[str]:
        result = []
        for line in text.splitlines():
            line = line.strip().lstrip("- ").lstrip("0123456789.) ")
            if line.endswith("?") and len(line) >= 12 and line not in result:
                result.append(line)
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            prompts = []
            for offset in range(self.batch_size):
                category = self._categories[(self.round + offset) % len(self._categories)]
                prompts.append(
                    "Generate 8 answerable user questions about "
                    f"{category}. Vary the wording and difficulty; include concrete "
                    "constraints, scenarios, comparisons, or examples. Avoid "
                    "reusing common textbook questions. Return one question per line."
                )
            batch = chat_batch(self.tokenizer, prompts)
            with self.teacher_lock:
                generated = generate_batch(
                    self.teacher, self.tokenizer, batch, 192, do_sample=True,
                )
            added = 0
            for _, text in generated:
                for question in self._parse(text):
                    key = self._normalize(question)
                    if key in self.seen:
                        continue
                    try:
                        self.questions.put(question, timeout=1.0)
                        self.seen.add(key)
                        if self.log_path:
                            with self.log_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps({
                                    "time": time.time(),
                                    "question": question,
                                    "round": self.round,
                                }, ensure_ascii=False) + "\n")
                        added += 1
                    except queue.Full:
                        break
            self.round += 1
            if added == 0:
                continue
            print(
                f"dynamic feeder: added {added} questions, "
                f"queue={self.questions.qsize()}"
            )


@torch.no_grad()
def sample_student_block(
    student,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Generate K greedy tokens from the current student policy.

    The training objective is intentionally aligned with the benchmark:
    no sampling temperature is used, and every proposal is an argmax token.
    """
    device = next(student.parameters()).device
    sequence = prompt_ids.to(device).unsqueeze(0)
    attention_mask = torch.ones_like(sequence)
    cache_position = torch.arange(sequence.shape[1], device=device)
    output = student(
        sequence,
        attention_mask=attention_mask,
        use_cache=True,
        cache_position=cache_position,
    )
    for _ in range(num_tokens):
        logits = output.logits[:, -1].float()
        token = logits.argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
        attention_mask = torch.ones_like(sequence)
        cache_position = torch.tensor([sequence.shape[1] - 1], device=device)
        output = student(
            token,
            attention_mask=attention_mask,
            past_key_values=output.past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
    return sequence[0]


def response_logits(logits: torch.Tensor, prompt_len: int, block_len: int) -> torch.Tensor:
    return logits[:, prompt_len - 1:prompt_len - 1 + block_len]


def greedy_prefix_mask(
    teacher_top1: torch.Tensor,
    student_top1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token matches and the exact surviving-prefix mask."""
    matches = (teacher_top1 == student_top1).float()
    if matches.ndim == 2:
        matches = matches[0]
    prefix_mask = torch.cat([
        torch.ones(1, device=matches.device),
        torch.cumprod(matches[:-1], dim=0),
    ])
    return matches, prefix_mask


@torch.no_grad()
def rollout_and_teacher(
    teacher,
    student,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    teacher_lock: threading.Lock,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a greedy rollout, teacher logits, and student top-1 logits."""
    student.eval()
    full_ids = sample_student_block(student, prompt_ids, num_tokens, 0.0)
    prompt_len = prompt_ids.numel()
    block_len = full_ids.numel() - prompt_len
    with teacher_lock:
        teacher_logits = response_logits(
            teacher(full_ids.unsqueeze(0)).logits,
            prompt_len,
            block_len,
        ).float()
    student_logits = response_logits(
        student(full_ids.unsqueeze(0)).logits,
        prompt_len,
        block_len,
    ).float()
    return full_ids, teacher_logits, student_logits


@torch.no_grad()
def score_rollout(
    teacher,
    student,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    temperature: float = 0.0,
    teacher_lock: threading.Lock | None = None,
) -> dict[str, float]:
    full_ids = sample_student_block(student, prompt_ids, num_tokens, temperature)
    prompt_len = prompt_ids.numel()
    block = full_ids[prompt_len:]
    if teacher_lock is None:
        teacher_output = teacher(full_ids.unsqueeze(0))
    else:
        with teacher_lock:
            teacher_output = teacher(full_ids.unsqueeze(0))
    t_logits = response_logits(
        teacher_output.logits, prompt_len, block.numel(),
    ).float()
    s_logits = response_logits(
        student(full_ids.unsqueeze(0)).logits, prompt_len, block.numel(),
    ).float()
    teacher_top1 = t_logits.argmax(-1)
    student_top1 = s_logits.argmax(-1)
    matches, prefix_mask = greedy_prefix_mask(teacher_top1, student_top1)
    mismatch = (matches == 0).nonzero(as_tuple=False)
    committed = int(mismatch[0].item()) if mismatch.numel() else int(block.numel())
    # ``prefix_mask`` includes positions whose proposal is accepted.  The
    # first mismatch itself is not committed.
    prefix_mask = torch.arange(block.numel(), device=block.device) < committed
    return {
        "committed_prefix_length": float(committed),
        "first_token_match": float(matches[0].item()),
        "complete_block": float(committed == block.numel()),
        "top1_agreement": float(matches.float().mean().item()),
        "active_prefix_fraction": float(prefix_mask.float().mean().item()),
        "tokens": float(block.numel()),
    }


def evaluate(
    teacher,
    student,
    prompt_ids_list: list[torch.Tensor],
    args: argparse.Namespace,
    tokenizer,
    teacher_lock: threading.Lock | None = None,
) -> dict[str, float]:
    metrics = [score_rollout(
        teacher, student, prompt_ids, args.speculative_tokens, 0.0,
        teacher_lock,
    ) for prompt_ids in prompt_ids_list]
    keys = metrics[0].keys()
    result = {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}
    if args.throughput_new_tokens > 0 and prompt_ids_list:
        # The throughput measurement uses the same exact greedy speculative
        # loop as production, rather than a token-level proxy.
        selected = prompt_ids_list[: max(1, min(len(prompt_ids_list), 8))]
        spec_tokens = 0
        spec_time = 0.0
        target_tokens = 0
        target_time = 0.0
        for prompt_ids in selected:
            spec = speculative_generate(
                teacher,
                student,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.throughput_new_tokens,
                num_speculative_tokens=args.speculative_tokens,
                temperature=0.0,
                device=str(next(student.parameters()).device),
            )
            spec_tokens += spec["tokens_generated"]
            spec_time += spec["total_time"]
            ids = prompt_ids.to(next(teacher.parameters()).device).unsqueeze(0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = teacher.generate(
                    ids,
                    max_new_tokens=args.throughput_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            target_time += time.perf_counter() - start
            target_tokens += int(output.shape[1] - ids.shape[1])
        result["speculative_tokens_per_second"] = spec_tokens / max(spec_time, 1e-9)
        result["target_only_tokens_per_second"] = target_tokens / max(target_time, 1e-9)
        result["throughput_speedup"] = (
            result["speculative_tokens_per_second"]
            / max(result["target_only_tokens_per_second"], 1e-9)
        )
    return result


def train(
    teacher,
    student,
    prompt_source,
    tokenizer,
    args: argparse.Namespace,
    teacher_lock: threading.Lock,
) -> list[dict[str, float]]:
    if args.temperature != 0.0:
        print("Ignoring --temperature: prefix-gated training is greedy by design.")
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        for batch_offset in range(max(int(args.train_batch_size), 1)):
            if isinstance(prompt_source, TeacherPromptFeeder):
                prompt_ids = chat_ids(tokenizer, prompt_source.get())
            else:
                prompt_ids = prompt_source[
                    (step * max(int(args.train_batch_size), 1) + batch_offset)
                    % len(prompt_source)
                ]
            with torch.no_grad():
                full_ids = sample_student_block(
                    student, prompt_ids, args.speculative_tokens, args.temperature,
                )
                prompt_len = prompt_ids.numel()
                block = full_ids[prompt_len:]
                with teacher_lock:
                    teacher_logits = response_logits(
                        teacher(full_ids.unsqueeze(0)).logits,
                        prompt_len,
                        block.numel(),
                    ).float()
            student.train()
            student_logits = response_logits(
                student(full_ids.unsqueeze(0)).logits, prompt_len, block.numel(),
            ).float()
            teacher_top1 = teacher_logits.argmax(-1)
            student_top1 = student_logits.detach().argmax(-1)
            matches, prefix_mask = greedy_prefix_mask(teacher_top1, student_top1)
            # Position j is trainable only when every earlier proposal was
            # accepted.  This is the exact greedy prefix condition.
            position_weights = torch.ones_like(prefix_mask)
            position_weights[0] = max(float(args.first_token_weight), 1.0)
            active_weights = prefix_mask * position_weights
            normalizer = active_weights.sum().clamp_min(1.0)
            greedy_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                teacher_top1.reshape(-1),
                reduction="none",
            ).view(1, -1)
            loss_hard = (greedy_ce[0] * active_weights).sum() / normalizer
            teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
            student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
            token_kl = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(-1)[0]
            loss_prefix_kl = (token_kl * active_weights).sum() / normalizer
            loss = (
                args.prefix_hard_weight * loss_hard
                + args.prefix_kl_weight * loss_prefix_kl
            )
            loss.backward()
            batch_rows.append({
                "loss": loss,
                "prefix_hard": loss_hard,
                "prefix_kl": loss_prefix_kl,
                "committed_prefix_length": prefix_mask.sum().detach(),
                "first_token_match": matches[0].detach(),
                "top1_agreement": matches.mean().detach(),
            })
        batch_size = len(batch_rows)
        # The individual losses were backpropagated above; average the
        # gradients so changing batch size does not change their scale.
        for parameter in student.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(batch_size)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        def batch_mean(key: str) -> float:
            return float(sum(row[key].detach().item() for row in batch_rows) / batch_size)
        row = {
            "step": float(step + 1),
            "loss": batch_mean("loss"),
            "prefix_hard": batch_mean("prefix_hard"),
            "prefix_kl": batch_mean("prefix_kl"),
            "committed_prefix_length": batch_mean("committed_prefix_length"),
            "first_token_match": batch_mean("first_token_match"),
            "top1_agreement": batch_mean("top1_agreement"),
        }
        history.append(row)
        print(
            f"step={step + 1}/{args.steps} loss={row['loss']:.4f} "
            f"prefix={row['committed_prefix_length']:.3f} "
            f"first={row['first_token_match']:.3f}"
        )
    student.eval()
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"teacher={args.teacher}")
    print(f"student={args.student}")
    print(f"device={args.device} dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.student, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = None
    if not args.dynamic_prompts:
        records = torch.load(args.records, map_location="cpu", weights_only=False)
        records = records[:args.prompts]
        if not records:
            raise RuntimeError("No records available for on-policy rollout")

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=dtype, low_cpu_mem_usage=True,
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, low_cpu_mem_usage=True,
    )
    teacher = bulk_move_model(teacher, args.device).eval()
    student = bulk_move_model(student, args.device).eval()

    teacher_lock = threading.Lock()
    feeder = None
    if args.dynamic_prompts:
        feeder_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher, padding_side="left",
        )
        if feeder_tokenizer.pad_token_id is None:
            feeder_tokenizer.pad_token = feeder_tokenizer.eos_token
        feeder = TeacherPromptFeeder(
            teacher,
            feeder_tokenizer,
            args.prompt_batch_size,
            args.queue_size,
            teacher_lock,
            args.feeder_log or str(output / "teacher_questions.jsonl"),
        )
        feeder.start()
        eval_prompt_ids = [
            chat_ids(tokenizer, feeder.get()) for _ in range(args.prompts)
        ]
        prompt_source = feeder
    else:
        eval_prompt_ids = [record["prompt_ids"] for record in records]
        prompt_source = eval_prompt_ids

    try:
        set_seed(args.seed + 1000)
        before = evaluate(
            teacher, student, eval_prompt_ids, args, tokenizer, teacher_lock,
        )
        print("Before:", json.dumps(before))
        history = train(
            teacher,
            student,
            prompt_source,
            tokenizer,
            args,
            teacher_lock,
        )
        set_seed(args.seed + 1000)
        after = evaluate(
            teacher, student, eval_prompt_ids, args, tokenizer, teacher_lock,
        )
        print("After:", json.dumps(after))
    finally:
        if feeder is not None:
            feeder.stop()

    student.save_pretrained(output)
    tokenizer.save_pretrained(output)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"before": before, "after": after, "history": history, "args": vars(args)},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved on-policy acceptance draft to {output}")


if __name__ == "__main__":
    main()
