"""ShortGPT-style layer pruning using Block Influence (BI) scores.

ShortGPT (arXiv 2024): BI_i = 1 - mean(cos_sim(X_i[t], X_{i+1}[t]))
Lower BI → layer transforms hidden states less → more redundant.
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM

from .calibration import collect_layer_outputs
from .debug_log import get_logger
from .inspect import ModelArchitecture


class DepthPruner:
    """ShortGPT-style depth pruning with Block Influence scoring.

    Algorithm:
      1. Forward calibration data through model, collect per-layer hidden states
      2. Compute BI_i = 1 - mean(cos_sim(X_i, X_{i+1})) for each adjacent layer pair
      3. Extend BI to all layers (interpolate for endpoints)
      4. Always keep first protect_first and last protect_last layers
      5. Sort remaining by BI descending, keep top-k
      6. Rebuild model with only kept layers
    """

    def __init__(self, arch: ModelArchitecture,
                 protect_first: int = 1, protect_last: int = 1):
        self.arch = arch
        self.protect_first = protect_first
        self.protect_last = protect_last
        self.bi_scores: list[float] | None = None

    # ---- Phase 1: Compute BI scores ----

    def compute_bi_scores(self, model: nn.Module,
                           input_ids: Tensor) -> list[float]:
        """Compute Block Influence scores for all layers.

        BI_i = 1 - mean_{t}(cos_sim(h_i[t], h_{i+1}[t]))

        where h_i[t] is the t-th token's hidden state at layer i's output.

        Args:
            model: HF model in eval mode.
            input_ids: Calibration data [num_samples, seq_len].

        Returns:
            BI scores list, length = num_layers.
        """
        num_layers = self.arch.num_layers

        # Collect hidden states from all layers
        # Returns list of lists: outputs[layer_idx][batch_idx] = [seq_len, hidden_size]
        layer_outputs = collect_layer_outputs(
            model, input_ids,
            layer_indices=list(range(num_layers))
        )

        # Compute BI between each adjacent pair
        bi_pairs: list[float] = []
        for i in range(num_layers - 1):
            bi = self._compute_pair_bi(
                layer_outputs[i],    # outputs from layer i
                layer_outputs[i + 1]  # outputs from layer i+1
            )
            bi_pairs.append(bi)

        # BI_pairs[i] is the BI between layer i and layer i+1
        # Assign each layer a BI score
        bi_scores = []
        for i in range(num_layers):
            if i == 0:
                # First layer: use BI with second layer
                score = bi_pairs[0] if len(bi_pairs) > 0 else 0.0
            elif i == num_layers - 1:
                # Last layer: use BI with previous layer
                score = bi_pairs[-1] if len(bi_pairs) > 0 else 0.0
            else:
                # Middle layers: average of left and right BI values
                score = (bi_pairs[i - 1] + bi_pairs[i]) / 2.0
            bi_scores.append(score)

        self.bi_scores = bi_scores

        log = get_logger()
        log.section("ShortGPT BI Scores")
        for i, s in enumerate(bi_scores):
            bar = "█" * max(1, int(s * 50))
            log.info(f"L{i:2d}: {s:.4f} {bar}")
        log.info(f"Most important: L{max(range(len(bi_scores)), key=lambda i: bi_scores[i])} "
                 f"(BI={max(bi_scores):.4f})")
        log.info(f"Most redundant:  L{min(range(len(bi_scores)), key=lambda i: bi_scores[i])} "
                 f"(BI={min(bi_scores):.4f})")

        return bi_scores

    def _compute_pair_bi(self, outputs_a: list[Tensor],
                          outputs_b: list[Tensor]) -> float:
        """Compute BI between two layers across all batches.

        BI = 1 - mean(cosine_similarity(a, b)) over all tokens.
        Higher BI = layer pair is more different = both layers matter.
        """
        all_cos_sims = []
        for hs_a, hs_b in zip(outputs_a, outputs_b):
            # hs: [seq_len, hidden_size]
            a = hs_a.float().reshape(-1, hs_a.shape[-1])  # [tokens, d]
            b = hs_b.float().reshape(-1, hs_b.shape[-1])  # [tokens, d]

            # Cosine similarity per token
            a_norm = torch.nn.functional.normalize(a, dim=-1)
            b_norm = torch.nn.functional.normalize(b, dim=-1)
            cos_sim = (a_norm * b_norm).sum(dim=-1)  # [tokens]
            all_cos_sims.append(cos_sim)

        if not all_cos_sims:
            return 0.0

        mean_cos_sim = torch.cat(all_cos_sims).mean().item()
        return 1.0 - mean_cos_sim

    # ---- Phase 2: Select layers ----

    def select_layers(self, target_count: int) -> list[int]:
        """Select which layers to keep based on BI scores.

        Algorithm:
          1. Always keep protect_first initial layers
          2. Always keep protect_last final layers
          3. From remaining layers, keep those with highest BI scores

        Args:
            target_count: Target number of layers to keep.

        Returns:
            Sorted list of kept layer indices.
        """
        if self.bi_scores is None:
            raise RuntimeError("Call compute_bi_scores() first.")

        num_layers = self.arch.num_layers
        protect_count = self.protect_first + self.protect_last

        if target_count <= protect_count:
            # Keep only protected layers
            kept = list(range(self.protect_first))
            kept.extend(range(num_layers - self.protect_last, num_layers))
            return sorted(set(kept))

        # Protected indices
        protected = set(range(self.protect_first))
        protected.update(range(num_layers - self.protect_last, num_layers))

        # Remaining layers (middle)
        middle = [(i, self.bi_scores[i])
                  for i in range(num_layers)
                  if i not in protected]

        # Sort by BI descending (higher BI = more important)
        middle.sort(key=lambda x: x[1], reverse=True)

        # Select top-k middle layers
        mid_count = target_count - protect_count
        kept_middle = [i for i, _ in middle[:mid_count]]

        # Combine and sort
        kept = sorted(list(protected) + kept_middle)
        removed = [i for i in range(num_layers) if i not in kept]

        log = get_logger()
        log.section("Layer Selection")
        log.info(f"Target: {target_count} layers (from {num_layers})")
        log.info(f"Protected first: {sorted(protected)[:self.protect_first]}")
        log.info(f"Protected last:  {sorted(protected)[-self.protect_last:]}")
        log.info(f"Kept ({len(kept)}):   {kept}")
        log.info(f"Removed ({len(removed)}): {removed}")
        if kept_middle:
            log.info(f"Top BI selected: {kept_middle[:5]}")

        return kept

    # ---- Phase 3: Rebuild model ----

    def prune_model(self, model: nn.Module,
                     keep_indices: list[int],
                     original_config) -> nn.Module:
        """Create a new model with only the kept layers.

        Args:
            model: Original model with existing weights.
            keep_indices: Sorted list of layer indices to keep.
            original_config: Original HF config.

        Returns:
            New model with reduced layer count.
        """
        # Create config with reduced layer count
        pruned_config = copy.deepcopy(original_config)
        pruned_config.num_hidden_layers = len(keep_indices)

        # Sync layer_types (Qwen3 sliding window config) with kept layer indices
        if hasattr(pruned_config, "layer_types") and pruned_config.layer_types is not None:
            pruned_config.layer_types = [
                pruned_config.layer_types[i] for i in keep_indices
            ]

        # Instantiate new model
        pruned_model = AutoModelForCausalLM.from_config(
            pruned_config,
            torch_dtype=model.dtype,
        )

        # Build state dict mapping
        src_state = model.state_dict()
        dst_state = {}

        # Map: new_layer_idx → original_layer_idx
        layer_map = {new_idx: old_idx
                     for new_idx, old_idx in enumerate(keep_indices)}

        # Copy global weights (unchanged)
        global_keys = ["model.embed_tokens.weight", "model.norm.weight",
                       "lm_head.weight"]
        for key in global_keys:
            if key in src_state:
                dst_state[key] = src_state[key].clone()

        # Copy per-layer weights with re-indexing
        layer_patterns = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "input_layernorm",
            "post_attention_layernorm",
        ]
        weight_or_bias = ["weight", "bias"]

        for new_idx, old_idx in layer_map.items():
            for pattern in layer_patterns:
                for wb in weight_or_bias:
                    src_key = f"model.layers.{old_idx}.{pattern}.{wb}"
                    dst_key = f"model.layers.{new_idx}.{pattern}.{wb}"
                    if src_key in src_state:
                        dst_state[dst_key] = src_state[src_key].clone()

        # Load pruned state dict
        missing, unexpected = pruned_model.load_state_dict(dst_state, strict=False)
        if missing:
            print(f"  Prune missing keys: {missing}")
        if unexpected:
            print(f"  Prune unexpected keys: {unexpected}")

        return pruned_model
