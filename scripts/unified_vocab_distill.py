"""Four-stage physical vocabulary unification for an independent draft.

The target tokenizer becomes canonical and the output remains a standard
Hugging Face CausalLM:

1. initialize target-vocabulary rows from exact/decomposed source tokens;
2. train embedding and lm_head rows;
3. train the final transformer layers with teacher CE/KL;
4. run on-policy accepted-prefix distillation.

Teacher questions and answers are generated locally, so no external dataset is
required.  ``--analyze-only`` performs the tokenizer/mapping smoke test without
loading the large teacher model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

# Allow ``python scripts/unified_vocab_distill.py`` from a source checkout
# without requiring the caller to set PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draft_adapter.vocab import build_vocabulary_mapping, initialize_model_for_target_vocab


DEFAULT_TARGET = "/home/frank/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it/snapshots/55cdba0740a9765956f49501f689a66b098feda3"
DEFAULT_STUDENT = "/home/frank/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--output", default="./runs/qwen3-gemma4-unified")
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions", type=int, default=128)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--stage1-steps", type=int, default=1000)
    parser.add_argument("--stage2-steps", type=int, default=4000)
    parser.add_argument("--stage3-steps", type=int, default=4000)
    parser.add_argument("--stage4-steps", type=int, default=2000)
    parser.add_argument("--lr-emb", type=float, default=2e-4)
    parser.add_argument("--lr-body", type=float, default=5e-5)
    parser.add_argument("--lr-on-policy", type=float, default=2e-5)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument("--final-layers", type=int, default=4)
    return parser.parse_args()


def load_model(path: str, dtype: torch.dtype, device: str, teacher: bool = False):
    kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except (ValueError, KeyError, ImportError):
        if not teacher:
            raise
        model = AutoModelForImageTextToText.from_pretrained(path, **kwargs)
    return model.to(device).eval()


def chat_prompt(tokenizer, question: str) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        return encoded["input_ids"]
    return tokenizer(question, return_tensors="pt").input_ids


@torch.inference_mode()
def teacher_generate(teacher, tokenizer, questions: list[str], answer_tokens: int) -> list[dict]:
    records = []
    for question in questions:
        prompt = chat_prompt(tokenizer, question)
        output = teacher.generate(
            prompt.to(next(teacher.parameters()).device),
            max_new_tokens=answer_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0].cpu()
        records.append({"prompt_ids": prompt[0], "answer_ids": output[prompt.shape[1]:]})
    return records


def freeze_for_stage(model, stage: int, final_layers: int) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == 1:
        modules = [model.get_input_embeddings()]
    elif stage == 2:
        modules = [model.get_input_embeddings(), model.get_output_embeddings()]
    else:
        modules = [model.get_input_embeddings(), model.get_output_embeddings()]
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            modules.extend(list(layers[-final_layers:]))
        norm = getattr(getattr(model, "model", None), "norm", None)
        if norm is not None:
            modules.append(norm)
    parameters = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def records_to_batch(records: list[dict], device: str):
    rows = []
    for record in records:
        rows.append(torch.cat((record["prompt_ids"], record["answer_ids"])))
    max_len = max(row.numel() for row in rows)
    input_ids = torch.full((len(rows), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    for index, row in enumerate(rows):
        input_ids[index, : row.numel()] = row
        attention_mask[index, : row.numel()] = 1
        prompt_len = records[index]["prompt_ids"].numel()
        labels[index, prompt_len: row.numel()] = row[prompt_len:]
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def train_supervised(model, records, steps, lr, device, stage, final_layers):
    parameters = freeze_for_stage(model, stage, final_layers)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    model.train()
    for step in range(steps):
        sample = [records[(step + offset) % len(records)] for offset in range(min(2, len(records)))]
        input_ids, attention_mask, labels = records_to_batch(sample, device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(steps // 10, 1) == 0:
            print(f"stage{stage} step={step + 1}/{steps} loss={loss.item():.4f}")
    model.eval()


@torch.no_grad()
def sample_student_rollout(student, prompt_ids: torch.Tensor, num_tokens: int, device: str):
    """Generate a greedy target-token block from the current student policy."""
    sequence = prompt_ids.to(device).unsqueeze(0)
    for _ in range(num_tokens):
        logits = student(sequence).logits[:, -1]
        token = logits.argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, token), dim=1)
    return sequence


def train_on_policy(student, teacher, records, steps, lr, device, num_tokens, final_layers):
    """Stage 4: teacher-weighted CE on blocks sampled by the student.

    Discrete sampling itself is not differentiated.  The sampled sequence is
    scored by both models and the student is trained toward the teacher's
    greedy token at each position, with prefix-survival weights.
    """
    parameters = freeze_for_stage(student, 3, final_layers=final_layers)
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    student.train()
    teacher.eval()
    for step in range(steps):
        record = records[step % len(records)]
        prompt = record["prompt_ids"]
        with torch.no_grad():
            rollout = sample_student_rollout(student, prompt, num_tokens, device)
            teacher_logits = teacher(rollout).logits.float()
        student_logits = student(rollout).logits.float()
        start = prompt.numel() - 1
        t_logits = teacher_logits[:, start : start + num_tokens]
        s_logits = student_logits[:, start : start + num_tokens]
        labels = t_logits.argmax(dim=-1)
        matches = (s_logits.detach().argmax(dim=-1) == labels).float()
        weights = torch.cumprod((0.5 + 0.5 * matches).clamp_min(1e-3), dim=-1)
        loss = (
            F.cross_entropy(
                s_logits.reshape(-1, s_logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(weights) * weights
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(steps // 10, 1) == 0:
            print(
                f"stage4 step={step + 1}/{steps} loss={loss.item():.4f} "
                f"prefix_weight={weights.mean().item():.3f}"
            )
    student.eval()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    mapping_path = Path(args.mapping) if args.mapping else output / "vocab_mapping.json"
    mapping = build_vocabulary_mapping(source_tokenizer, target_tokenizer)
    mapping.save(mapping_path)
    print(json.dumps(mapping.stats(), ensure_ascii=False, indent=2))
    print(f"saved mapping: {mapping_path}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    student = load_model(args.student, dtype, args.device)
    init_stats = initialize_model_for_target_vocab(
        student, mapping, source_tokenizer, target_tokenizer, seed=args.seed,
    )
    print(json.dumps(init_stats, ensure_ascii=False, indent=2))
    student.save_pretrained(output)
    target_tokenizer.save_pretrained(output)
    with (output / "alignment_stats.json").open("w", encoding="utf-8") as handle:
        json.dump({"mapping": mapping.stats(), "initialization": init_stats}, handle, indent=2)
    if args.analyze_only:
        print(f"saved vocabulary-unified student smoke checkpoint: {output}")
        return

    teacher = load_model(args.target, dtype, args.device, teacher=True)
    questions = [
        "Explain one practical consequence of quantum entanglement.",
        "Write a Python function that merges overlapping intervals.",
        "Compare a relational database with a document database.",
        "用中文解释什么是 speculative decoding。",
    ]
    questions = (questions * ((args.questions + len(questions) - 1) // len(questions)))[: args.questions]
    records = teacher_generate(teacher, target_tokenizer, questions, args.answer_tokens)
    train_supervised(student, records, args.stage1_steps, args.lr_emb, args.device, 1, args.final_layers)
    train_supervised(student, records, args.stage2_steps, args.lr_body, args.device, 2, args.final_layers)
    train_supervised(student, records, args.stage3_steps, args.lr_body, args.device, 3, args.final_layers)
    train_on_policy(
        student, teacher, records, args.stage4_steps, args.lr_on_policy,
        args.device, args.speculative_tokens, args.final_layers,
    )
    student.save_pretrained(output)
    target_tokenizer.save_pretrained(output)
    print(f"saved four-stage checkpoint: {output}")


if __name__ == "__main__":
    main()
