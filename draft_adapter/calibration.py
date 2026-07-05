"""Calibration data loading and Swift-SVD-style incremental covariance aggregation."""

from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import tqdm

from .utils import load_calibration_data


@dataclass
class CovarianceAggregator:
    """Incrementally aggregate covariance matrix from hidden states.

    Swift-SVD style: streams hidden states in batches, accumulates
    C = sum(X_i^T @ X_i) / total_tokens without storing the full
    activation matrix.  Memory: O(d^2) instead of O(N_tokens * d).

    Attributes:
        dim: Dimension of hidden states (d).
        count: Total number of token vectors accumulated.
        C: Accumulated sum X^T @ X, shape [dim, dim].
    """

    dim: int
    count: int = 0
    C: Tensor | None = None

    def update(self, hidden_states: Tensor) -> None:
        """Accumulate X^T @ X from a batch of hidden states.

        Args:
            hidden_states: shape [batch_tokens, dim].  First dim is
                           flattened (batch * seq_len).
        """
        if hidden_states.dim() != 2:
            hidden_states = hidden_states.reshape(-1, self.dim)

        n = hidden_states.shape[0]
        # Accumulate in float32 for numerical stability
        x = hidden_states.float()
        gram = x.T @ x  # [dim, dim]

        if self.C is None:
            self.C = gram
        else:
            self.C += gram
        self.count += n

    def compute(self) -> Tensor:
        """Return normalized covariance matrix C / count.

        Returns:
            C_normalized: shape [dim, dim], symmetric positive semi-definite.
        """
        if self.C is None or self.count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        return self.C / self.count


def collect_layer_outputs(model, input_ids: Tensor,
                          layer_indices: list[int] | None = None) -> list[list[Tensor]]:
    """Collect hidden states from specified layers.

    Hooks into each transformer layer's output (post residual add) and
    collects the hidden states during a forward pass.

    Args:
        model: HF transformer model.
        input_ids: shape [batch, seq_len].
        layer_indices: Which layer outputs to collect (None = all).

    Returns:
        List of lists: outputs[layer_idx][batch_idx] = Tensor [seq_len, hidden_size]
        or None if layer_indices is None and all are collected.
    """
    if layer_indices is None:
        layer_indices = list(range(model.config.num_hidden_layers))

    outputs: dict[int, list[Tensor]] = {i: [] for i in layer_indices}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(idx: int):
        def hook(module, args, output):
            # output[0] is the hidden states tensor
            hs = output[0] if isinstance(output, tuple) else output
            outputs[idx].append(hs.detach().cpu())
        return hook

    try:
        # Register hooks on target layers
        for idx in layer_indices:
            layer = model.model.layers[idx]
            handle = layer.register_forward_hook(make_hook(idx))
            hooks.append(handle)

        with torch.no_grad():
            model(input_ids)

    finally:
        for h in hooks:
            h.remove()

    return [outputs[i] for i in layer_indices]


def compute_global_covariance(model, input_ids: Tensor,
                               aggregator: CovarianceAggregator | None = None,
                               chunk_size: int = 4) -> CovarianceAggregator:
    """Compute global covariance matrix from all layer outputs.

    Runs calibration data through the model and aggregates hidden states
    from all transformer layers into a single covariance matrix.

    Args:
        model: HF transformer model in eval mode.
        input_ids: calibration input_ids [num_samples, seq_len].
        aggregator: Existing aggregator to continue accumulating, or None.
        chunk_size: Process this many samples per forward pass.

    Returns:
        CovarianceAggregator with accumulated data from all layers.
    """
    if aggregator is None:
        aggregator = CovarianceAggregator(dim=model.config.hidden_size)

    total = input_ids.shape[0]

    for start in tqdm(range(0, total, chunk_size), desc="Computing covariance"):
        end = min(start + chunk_size, total)
        batch = input_ids[start:end].to(model.device)

        # Collect from all layers
        layer_outputs = collect_layer_outputs(
            model, batch,
            layer_indices=list(range(model.config.num_hidden_layers))
        )

        # Aggregate ALL hidden states from ALL layers
        for layer_idx in range(len(layer_outputs)):
            for batch_output in layer_outputs[layer_idx]:
                aggregator.update(batch_output)

    return aggregator
