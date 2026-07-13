"""Qwen3 draft model with SVD-decomposed projection layers."""

import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM


class DecomposedLinear(nn.Module):
    """Two linear projections representing a truncated SVD weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.proj_in = nn.Linear(in_features, rank, bias=False)
        self.proj_out = nn.Linear(rank, out_features, bias=bias)

    def forward(self, hidden_states):
        return self.proj_out(self.proj_in(hidden_states))


class DraftQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 decoder semantics with project-owned low-rank projections."""

    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        rank_map = getattr(config, "svd_rank_map", None)
        if not rank_map:
            raise ValueError("SVD-hybrid config is missing svd_rank_map")

        for module_name, rank in rank_map.items():
            parent_name, child_name = module_name.rsplit(".", 1)
            parent = self.get_submodule(parent_name)
            dense = getattr(parent, child_name)
            if not isinstance(dense, nn.Linear):
                raise TypeError(
                    f"Expected {module_name} to be nn.Linear, got "
                    f"{type(dense).__name__}"
                )
            setattr(
                parent,
                child_name,
                DecomposedLinear(
                    dense.in_features,
                    dense.out_features,
                    int(rank),
                    bias=dense.bias is not None,
                ),
            )
