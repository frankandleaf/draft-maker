"""DistillSpec-style on-policy knowledge distillation for draft models.

Core insight from DistillSpec (ICLR 2024):
  On-policy generation is the key ingredient. The student (draft model)
  generates tokens autoregressively, and the teacher scores them.

Three KL divergence modes:
  - "reverse": KL(p_s || p_t) — recommended for greedy decoding
  - "forward": KL(p_t || p_s) — recommended for temperature sampling
  - "tvd":     0.5 * sum|p_s - p_t| — directly optimizes acceptance rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from .config import DistillConfig
from .utils import load_calibration_data, set_seed


class DistillationTrainer:
    """On-policy distillation with top-K sparse KL divergence.

    Key design choices:
      1. ON-POLICY: student generates, teacher scores.
         NOT teacher-generates-student-imitates.
      2. TOP-K SPARSE KL: only compute KL on teacher's top-k tokens.
         Prevents student from wasting capacity on near-zero logits.
      3. Teacher in inference_mode, student in train mode.
      4. Teacher loaded with device_map="auto" for memory efficiency.
    """

    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 tokenizer,
                 config: DistillConfig):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config

        # Teacher never trains
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student trains
        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
        )

    # ---- Core loss functions ----

    def top_k_kl_loss(self,
                       student_logits: Tensor,
                       teacher_logits: Tensor) -> Tensor:
        """Top-K sparse KL divergence.

        Only computes KL on teacher's top-k token positions.
        All other positions have gradient zero.

        Args:
            student_logits: [batch, seq, vocab], student output logits.
            teacher_logits: [batch, seq, vocab], teacher output logits.

        Returns:
            Scalar loss (mean over batch and seq).
        """
        k = min(self.config.top_k, teacher_logits.shape[-1])
        T = self.config.kl_temperature

        # Get teacher's top-k token indices
        _, topk_indices = teacher_logits.topk(k, dim=-1)  # [batch, seq, k]

        # Gather logits at top-k positions
        teacher_topk = teacher_logits.gather(-1, topk_indices)  # [batch, seq, k]
        student_topk = student_logits.gather(-1, topk_indices)  # [batch, seq, k]

        # Softmax within the top-k subset
        teacher_probs = F.softmax(teacher_topk / T, dim=-1)
        student_probs = F.softmax(student_topk / T, dim=-1)
        student_log_probs = F.log_softmax(student_topk / T, dim=-1)

        mode = self.config.kl_mode

        if mode == "reverse":
            # KL(student || teacher) = sum(p_s * (log p_s - log p_t))
            teacher_log_probs = F.log_softmax(teacher_topk / T, dim=-1)
            kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

        elif mode == "forward":
            # KL(teacher || student) = sum(p_t * (log p_t - log p_s))
            teacher_log_probs = F.log_softmax(teacher_topk / T, dim=-1)
            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

        elif mode == "tvd":
            # 0.5 * sum |p_s - p_t|
            kl = 0.5 * (student_probs - teacher_probs).abs().sum(dim=-1)

        else:
            raise ValueError(f"Unknown kl_mode: {mode}")

        return kl.mean()

    # ---- Training step ----

    @torch.no_grad()
    def _teacher_forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Teacher forward pass (no gradient, inference mode)."""
        self.teacher.eval()
        output = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return output.logits.detach()

    def train_step(self, input_ids: Tensor) -> dict[str, float]:
        """Single training step.

        1. Student generates tokens (on-policy)
        2. Teacher scores the student-generated sequence
        3. Compute top-K KL loss
        4. Backpropagate through student only
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ---- On-policy generation ----
        # Student generates a few tokens given the prefix
        gen_len = min(self.config.generate_len, 32)
        prompt_len = max(input_ids.shape[1] - gen_len, 1)

        prompt_ids = input_ids[:, :prompt_len]  # Use prefix as prompt

        self.student.eval()
        with torch.no_grad():
            generated = self.student.generate(
                prompt_ids,
                max_new_tokens=gen_len,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
                             or self.tokenizer.eos_token_id,
            )
        self.student.train()

        # Full sequence: prompt + generated
        full_ids = generated  # Already includes prompt

        # Pad or truncate to consistent length
        max_len = min(full_ids.shape[1], self.config.max_seq_len)
        full_ids = full_ids[:, :max_len]
        attn_mask = torch.ones_like(full_ids)

        # ---- Teacher forward (scores student's generated tokens) ----
        teacher_logits = self._teacher_forward(full_ids, attn_mask)

        # ---- Student forward ----
        student_output = self.student(
            input_ids=full_ids,
            attention_mask=attn_mask,
        )
        student_logits = student_output.logits

        # Align lengths (teacher and student may produce different seq lens)
        min_len = min(teacher_logits.shape[1], student_logits.shape[1])
        teacher_logits = teacher_logits[:, :min_len, :].contiguous()
        student_logits = student_logits[:, :min_len, :].contiguous()

        # Only compute loss on generated tokens (shift by 1 for next-token pred)
        # Use the generated portion only
        shift = prompt_len - 1
        if shift > 0 and min_len > shift:
            teacher_logits = teacher_logits[:, shift:-1, :].contiguous()
            student_logits = student_logits[:, shift:-1, :].contiguous()
        else:
            teacher_logits = teacher_logits[:, :-1, :].contiguous()
            student_logits = student_logits[:, :-1, :].contiguous()

        # ---- Loss computation ----
        loss = self.top_k_kl_loss(student_logits, teacher_logits)

        # ---- Backward ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "batch_size": batch_size}

    # ---- Full training loop ----

    def train(self, train_input_ids: Tensor) -> nn.Module:
        """Run full distillation training.

        Args:
            train_input_ids: [num_prompts, seq_len] tensor.

        Returns:
            Trained student model.
        """
        total_steps = self.config.steps
        batch_size = self.config.batch_size
        num_prompts = train_input_ids.shape[0]

        losses = []
        pbar = tqdm(range(total_steps), desc="Distilling")
        for step in pbar:
            # Sample random batch
            indices = torch.randint(0, num_prompts, (batch_size,))
            batch = train_input_ids[indices].to(self.student.device)

            metrics = self.train_step(batch)
            losses.append(metrics["loss"])

            if step % 50 == 0 or step == total_steps - 1:
                avg_loss = sum(losses[-50:]) / min(50, len(losses))
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        self.student.eval()
        return self.student
