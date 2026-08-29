"""On-policy acceptance-aware distillation for an independent CausalLM draft.

The student samples its own K-token prefixes.  The teacher then scores those
same prefixes, and the student is updated to maximize the probability that the
whole draft block survives speculative verification.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import re
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from teacher_self_distill import bulk_move_model, chat_batch, chat_ids, generate_batch


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
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy-weight", type=float, default=0.1)
    parser.add_argument(
        "--first-token-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for the first speculative token greedy CE.",
    )
    parser.add_argument("--survival-weight", type=float, default=0.5)
    parser.add_argument("--accept-log-weight", type=float, default=0.5)
    parser.add_argument(
        "--first-token-ratio-weight",
        type=float,
        default=0.0,
        help="Weight for the exact first-token acceptance-ratio loss.",
    )
    parser.add_argument("--kl-anchor-weight", type=float, default=0.5)
    parser.add_argument("--sample-ce-weight", type=float, default=0.1)
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
    temperature: float,
) -> torch.Tensor:
    """Sample K tokens from the current student policy.

    Uses Qwen3's DynamicCache so only one token is processed after the prompt.
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
        probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
        token = torch.multinomial(probs, 1)
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


@torch.no_grad()
def score_rollout(
    teacher,
    student,
    prompt_ids: torch.Tensor,
    num_tokens: int,
    temperature: float,
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
    t_probs = F.softmax(t_logits / max(temperature, 1e-5), dim=-1)
    s_probs = F.softmax(s_logits / max(temperature, 1e-5), dim=-1)
    tv = 0.5 * (t_probs - s_probs).abs().sum(-1)
    alpha = (1.0 - tv).clamp(0.0, 1.0)
    survival = torch.cumprod(alpha.clamp_min(1e-5), dim=-1)
    sampled_t = t_probs[0, torch.arange(block.numel(), device=t_probs.device), block]
    sampled_s = s_probs[0, torch.arange(block.numel(), device=s_probs.device), block]
    ratio_accept = torch.minimum(
        torch.ones_like(sampled_t),
        sampled_t / sampled_s.clamp_min(1e-8),
    )
    greedy_match = (
        t_logits.argmax(-1) == s_logits.argmax(-1)
    ).float().mean()
    return {
        "greedy_top1": float(greedy_match.item()),
        "mean_alpha": float(alpha.mean().item()),
        "expected_accept_length": float(survival.sum().item()),
        "sampled_accept_length": float(torch.cumprod(ratio_accept, dim=-1).sum().item()),
        "tokens": float(block.numel()),
    }


def evaluate(
    teacher,
    student,
    prompt_ids_list: list[torch.Tensor],
    args: argparse.Namespace,
    teacher_lock: threading.Lock | None = None,
) -> dict[str, float]:
    metrics = [score_rollout(
        teacher, student, prompt_ids, args.speculative_tokens, args.temperature,
        teacher_lock,
    ) for prompt_ids in prompt_ids_list]
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def train(
    teacher,
    student,
    prompt_source,
    tokenizer,
    args: argparse.Namespace,
    teacher_lock: threading.Lock,
) -> list[dict[str, float]]:
    student.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    device = next(student.parameters()).device
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
            student_logits = response_logits(
                student(full_ids.unsqueeze(0)).logits, prompt_len, block.numel(),
            ).float()
            temp = max(args.temperature, 1e-5)
            teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
            student_probs = F.softmax(student_logits / temp, dim=-1)
            tv = 0.5 * (teacher_probs - student_probs).abs().sum(-1)
            alpha = (1.0 - tv).clamp(1e-4, 1.0)
            survival = torch.cumprod(alpha, dim=-1)
            loss_survival = (1.0 - survival).sum() / block.numel()
            loss_accept_log = -alpha.log().mean()
            loss_kl = (
                teacher_probs
                * (
                    teacher_probs.clamp_min(1e-8).log()
                    - student_probs.clamp_min(1e-8).log()
                )
            ).sum(-1).mean()
            teacher_top1 = teacher_logits.argmax(-1)
            greedy_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                teacher_top1.reshape(-1),
                reduction="none",
            ).view(1, -1)
            first_weight = max(float(args.first_token_weight), 1.0)
            token_weights = torch.ones_like(greedy_ce)
            token_weights[:, 0] = first_weight
            loss_greedy = (greedy_ce * token_weights).sum() / token_weights.sum()
            loss_sample_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                block.unsqueeze(0).reshape(-1),
            )
            token_indices = torch.arange(
                block.numel(), device=block.device,
            )
            sampled_teacher = teacher_probs[
                0, token_indices, block
            ]
            sampled_student = student_probs[
                0, token_indices, block
            ]
            ratio_accept = torch.minimum(
                torch.ones_like(sampled_teacher),
                sampled_teacher / sampled_student.clamp_min(1e-8),
            )
            loss_first_token_ratio = -ratio_accept[0].clamp_min(1e-8).log()
            loss = (
                args.survival_weight * loss_survival
                + args.accept_log_weight * loss_accept_log
                + args.kl_anchor_weight * loss_kl
                + args.greedy_weight * loss_greedy
                + args.sample_ce_weight * loss_sample_ce
                + args.first_token_ratio_weight * loss_first_token_ratio
            )
            loss.backward()
            batch_rows.append({
                "loss": loss,
                "survival": loss_survival,
                "accept_log": loss_accept_log,
                "kl_anchor": loss_kl,
                "greedy_ce": loss_greedy,
                "sample_ce": loss_sample_ce,
                "first_token_ratio": loss_first_token_ratio,
                "mean_alpha": alpha.detach().mean(),
                "expected_accept_length": survival.detach().sum(),
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
            "survival": batch_mean("survival"),
            "accept_log": batch_mean("accept_log"),
            "kl_anchor": batch_mean("kl_anchor"),
            "greedy_ce": batch_mean("greedy_ce"),
            "sample_ce": batch_mean("sample_ce"),
            "first_token_ratio": batch_mean("first_token_ratio"),
            "mean_alpha": batch_mean("mean_alpha"),
            "expected_accept_length": batch_mean("expected_accept_length"),
        }
        history.append(row)
        print(
            f"step={step + 1}/{args.steps} loss={row['loss']:.4f} "
            f"accept_log={row['accept_log']:.4f} "
            f"mean_alpha={row['mean_alpha']:.3f} "
            f"expected_accept_length={row['expected_accept_length']:.3f}"
        )
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
        before = evaluate(teacher, student, eval_prompt_ids, args, teacher_lock)
        print("Before:", json.dumps(before))
        history = train(
            teacher, student, prompt_source, tokenizer, args, teacher_lock,
        )
        set_seed(args.seed + 1000)
        after = evaluate(teacher, student, eval_prompt_ids, args, teacher_lock)
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
