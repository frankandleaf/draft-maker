"""Utility functions: seeding, parameter counting, data loading."""

import random

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_param_count(count: int) -> str:
    """Human-readable parameter count."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    if count >= 1e6:
        return f"{count / 1e6:.2f}M"
    if count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)


def load_calibration_data(tokenizer, num_samples: int = 16,
                          seq_len: int = 512, device: str = "cuda") -> Tensor:
    """Load calibration text from C4 and tokenize.

    Falls back to random tokens if dataset download fails.

    Args:
        tokenizer: HF tokenizer.
        num_samples: Number of sequences to return.
        seq_len: Max tokens per sequence.
        device: Target device.

    Returns:
        input_ids tensor of shape [num_samples, seq_len].
    """
    input_ids_list = []
    try:
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            tokens = tokenizer(
                example["text"],
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            ids = tokens.input_ids[0]
            if ids.shape[0] < seq_len:
                ids = torch.nn.functional.pad(ids, (0, seq_len - ids.shape[0]), value=tokenizer.pad_token_id or 0)
            input_ids_list.append(ids[:seq_len])
    except Exception:
        # Fallback: random token IDs
        for _ in range(num_samples):
            ids = torch.randint(0, tokenizer.vocab_size, (seq_len,))
            input_ids_list.append(ids)

    if not input_ids_list:
        for _ in range(num_samples):
            input_ids_list.append(torch.randint(0, tokenizer.vocab_size, (seq_len,)))

    return torch.stack(input_ids_list).to(device)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16, "float64": torch.float64}[dtype_str]
