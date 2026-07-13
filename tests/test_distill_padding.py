"""Tests for decoder-only prompt preparation during distillation."""

import torch

from draft_adapter.distill import DistillationTrainer


def test_left_pad_prompts_moves_right_padding_and_builds_mask():
    prompts = torch.tensor([
        [11, 12, 0, 0],
        [21, 22, 23, 0],
    ])

    padded, mask = DistillationTrainer._left_pad_prompts(prompts, 0)

    assert padded.tolist() == [[0, 0, 11, 12], [0, 21, 22, 23]]
    assert mask.tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
