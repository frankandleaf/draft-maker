"""DistillSpec-style knowledge distillation with two-phase training.

Phase 1 (Recovery): Teacher generates → student imitates (off-policy)
  — stable, prevents on-policy collapse from broken initial model
Phase 2 (Speculative): Student generates → teacher scores (on-policy)
  — maximizes token acceptance rate for speculative decoding

DistillSpec (ICLR 2024): on-policy is key for SD, but requires a
non-degenerate student to bootstrap from.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import DistillConfig
from .utils import load_calibration_data


class DistillationTrainer:
    """Two-phase distillation: off-policy recovery → on-policy SD tuning."""

    def __init__(self, teacher, student, tokenizer, config: DistillConfig):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=config.learning_rate)
        self.scheduler = None

    # ---- Loss functions ----

    @staticmethod
    def full_kl(student_logits: Tensor, teacher_logits: Tensor,
                T: float = 1.0) -> Tensor:
        """Full-vocab forward KL(teacher || student) — stable, mode-covering."""
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        # KL(t || s) = sum(p_t * log(p_t / p_s))
        t_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        return kl.mean()

    @staticmethod
    def top_k_kl(student_logits: Tensor, teacher_logits: Tensor,
                 k: int = 10, T: float = 1.0, mode: str = "forward") -> Tensor:
        """Sparse KL on teacher's top-k tokens only."""
        k = min(k, teacher_logits.shape[-1])
        _, topk = teacher_logits.topk(k, dim=-1)
        t_topk = teacher_logits.gather(-1, topk)
        s_topk = student_logits.gather(-1, topk)

        t_probs = F.softmax(t_topk / T, dim=-1)
        s_probs = F.softmax(s_topk / T, dim=-1)
        s_log_probs = F.log_softmax(s_topk / T, dim=-1)
        t_log_probs = F.log_softmax(t_topk / T, dim=-1)

        if mode == "forward":
            kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        elif mode == "reverse":
            kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
        else:  # tvd
            kl = 0.5 * (t_probs - s_probs).abs().sum(dim=-1)
        return kl.mean()

    # ---- Teacher forward (no grad, compiled) ----
    @torch.no_grad()
    def _teacher_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        out = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.logits.detach()

    # ---- Phase 1: Off-policy recovery (teacher generates, student imitates) ----

    def train_step_recovery(
        self,
        prompt_ids: Tensor,
        prompt_attention_mask: Tensor,
    ) -> dict:
        """Teacher generates; student learns to match on teacher's tokens."""
        gen_len = self.config.generate_len
        device = prompt_ids.device

        # Teacher generates high-quality tokens
        teacher_gen = self.teacher.generate(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=gen_len, do_sample=True,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id
                         or self.tokenizer.eos_token_id,
        )
        full_ids = teacher_gen[:, :prompt_ids.shape[1] + gen_len]
        generated_len = full_ids.shape[1] - prompt_ids.shape[1]
        full_attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    prompt_ids.shape[0], generated_len,
                    device=prompt_ids.device,
                    dtype=prompt_attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        # Student forward, CE on generated tokens only
        student_out = self.student(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
        )
        student_logits = student_out.logits  # [batch, seq, vocab]

        # CE loss: logits at position i predict token at i+1
        p_len = prompt_ids.shape[1]
        # Only compute on generated portion: positions [p_len-1, seq-2]
        s_gen = student_logits[:, p_len-1:-1, :].contiguous()
        labels_gen = full_ids[:, p_len:].contiguous()
        loss = F.cross_entropy(
            s_gen.reshape(-1, s_gen.shape[-1]),
            labels_gen.reshape(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "phase": "recovery"}

    # ---- Phase 2: On-policy speculative (student generates, teacher scores) ----

    def train_step_speculative(
        self,
        prompt_ids: Tensor,
        prompt_attention_mask: Tensor,
    ) -> dict:
        """Student generates; teacher scores → top-K KL on student's tokens."""
        gen_len = min(self.config.generate_len, 32)
        device = prompt_ids.device

        # Student generates (on-policy)
        self.student.eval()
        with torch.no_grad():
            student_gen = self.student.generate(
                prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=gen_len, do_sample=True,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
                             or self.tokenizer.eos_token_id,
            )
        self.student.train()

        full_ids = student_gen[:, :prompt_ids.shape[1] + gen_len]
        generated_len = full_ids.shape[1] - prompt_ids.shape[1]
        full_attention_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    prompt_ids.shape[0], generated_len,
                    device=prompt_ids.device,
                    dtype=prompt_attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        # Teacher scores + student forward on student-generated tokens
        teacher_logits = self._teacher_forward(
            full_ids,
            attention_mask=full_attention_mask,
        )
        student_out = self.student(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
        )
        student_logits = student_out.logits

        n_gen = full_ids.shape[1] - prompt_ids.shape[1]
        t_logits = teacher_logits[:, prompt_ids.shape[1]-1:-1, :]
        s_logits = student_logits[:, prompt_ids.shape[1]-1:-1, :]

        loss = self.top_k_kl(
            s_logits, t_logits,
            k=self.config.top_k, T=self.config.kl_temperature,
            mode=self.config.kl_mode,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "phase": "speculative"}

    # ---- Full training loop ----

    @staticmethod
    def _left_pad_prompts(
        prompt_ids: Tensor,
        pad_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Move right padding to the left for decoder-only generation."""
        attention_mask = prompt_ids.ne(pad_token_id)
        left_padded = torch.full_like(prompt_ids, pad_token_id)
        left_mask = torch.zeros_like(prompt_ids)
        for row in range(prompt_ids.shape[0]):
            tokens = prompt_ids[row][attention_mask[row]]
            if tokens.numel():
                left_padded[row, -tokens.numel():] = tokens
                left_mask[row, -tokens.numel():] = 1
        return left_padded, left_mask

    def train(self, train_input_ids: Tensor) -> torch.nn.Module:
        total = self.config.steps
        bs = self.config.batch_size
        n_prompts = train_input_ids.shape[0]
        rec_steps = total // 2  # 50% off-policy recovery, 50% on-policy SD

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total, eta_min=self.config.learning_rate / 50)
        losses = []
        pbar = tqdm(range(total), desc="Distilling")
        for step in pbar:
            idx = torch.randint(0, n_prompts, (bs,))
            batch = train_input_ids[idx].to(self.student.device)
            # Prompt = first half, student/teacher generates second half
            prompt_len = batch.shape[1] // 2
            prompts = batch[:, :prompt_len]
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            prompts, prompt_attention_mask = self._left_pad_prompts(
                prompts, pad_token_id,
            )

            if step < rec_steps:
                metrics = self.train_step_recovery(
                    prompts, prompt_attention_mask,
                )
            else:
                metrics = self.train_step_speculative(
                    prompts, prompt_attention_mask,
                )

            losses.append(metrics["loss"])
            self.scheduler.step()
            phase = metrics["phase"]

            if step == rec_steps - 1:
                print(f"\n  Switching to on-policy at step {step+1}")
            if step % 50 == 0 or step == total - 1:
                avg = sum(losses[-50:]) / min(50, len(losses))
                pbar.set_postfix({"loss": f"{avg:.4f}", "phase": phase})

        self.student.eval()
        return self.student
