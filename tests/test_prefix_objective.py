import torch

from scripts.on_policy_acceptance_distill import greedy_prefix_mask


def test_greedy_prefix_mask_stops_after_first_mismatch():
    teacher = torch.tensor([[4, 8, 2, 9, 1]])
    student = torch.tensor([[4, 8, 7, 9, 1]])
    matches, active = greedy_prefix_mask(teacher, student)
    assert torch.equal(matches, torch.tensor([1., 1., 0., 1., 1.]))
    assert torch.equal(active, torch.tensor([1., 1., 1., 0., 0.]))


def test_greedy_prefix_mask_supports_single_token_blocks():
    matches, active = greedy_prefix_mask(
        torch.tensor([[4]]), torch.tensor([[7]]),
    )
    assert torch.equal(matches, torch.tensor([0.]))
    assert torch.equal(active, torch.tensor([1.]))
