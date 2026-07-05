"""SliceGPT-style width compression with global orthogonal projection Q.

Key insight from the SliceGPT paper (ICLR 2024):
  PCA rotation before slicing deletes the least important principal components,
  not random dimensions. Global unified Q avoids per-layer residual transform
  complexity (Q^T @ Q = I).

Weight projection rules (critical for mathematical correctness):

  Rule "both":  Square matrices [d,d] where both sides touch residual stream.
                e.g. q_proj, o_proj.
                W_new = Q_top.T @ W @ Q_top  ->  [d', d']

  Rule "input": Weights where the INPUT dimension is residual stream (d)
                but OUTPUT is NOT (e.g., k_proj, v_proj, gate_proj, up_proj).
                W_new = W @ Q_top  ->  [out, d'] then slice output dim

  Rule "output": Weights where the OUTPUT dimension is residual stream (d)
                 but INPUT is NOT (e.g., down_proj).
                 W_new = Q_top.T @ W  ->  [d', in] then slice input dim

  Rule "embed":  Embedding/lm_head weights [V, d].
                 W_new = W @ Q_top  ->  [V, d']

  Rule "norm":   RMSNorm weights [d]. Slice to [d'].
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from .calibration import CovarianceAggregator
from .inspect import ModelArchitecture
from .utils import get_dtype


# --- State dict key patterns and their projection rules ---

# Each entry: (key_suffix_pattern, projection_rule)
# Rules: "both" | "input" | "output" | "embed" | "norm" | "skip"
LAYER_WEIGHT_RULES = [
    ("self_attn.q_proj.weight", "both"),
    ("self_attn.q_proj.bias", "output"),
    ("self_attn.k_proj.weight", "input"),
    ("self_attn.k_proj.bias", "skip"),
    ("self_attn.v_proj.weight", "input"),
    ("self_attn.v_proj.bias", "skip"),
    ("self_attn.o_proj.weight", "both"),
    ("self_attn.o_proj.bias", "output"),
    ("mlp.gate_proj.weight", "input"),
    ("mlp.up_proj.weight", "input"),
    ("mlp.down_proj.weight", "output"),
    ("input_layernorm.weight", "norm"),
    ("post_attention_layernorm.weight", "norm"),
]

GLOBAL_WEIGHT_RULES = [
    ("model.embed_tokens.weight", "embed"),
    ("lm_head.weight", "embed"),
    ("model.norm.weight", "norm"),
]


class WidthCompressor:
    """Apply SliceGPT-style width compression using global orthogonal projection.

    Pipeline:
      1. Compute (or receive) global covariance matrix C [d, d]
      2. Eigendecompose C → eigenvalues, Q
      3. Take Q_top = Q[:, -d':]  (top d' principal components)
      4. Apply Q_top to all weights according to projection rules
      5. Slice attention heads and FFN intermediate dimensions
      6. Build compressed model
      7. Verify residual stream consistency
    """

    def __init__(self, arch: ModelArchitecture,
                 covariance: Tensor | None = None):
        """
        Args:
            arch: ModelArchitecture with target_* fields computed.
            covariance: [d, d] covariance matrix. If None, compute later.
        """
        self.arch = arch
        self.covariance = covariance
        self.Q_top: Tensor | None = None  # [d, d']

    # ---- Phase 1: Compute projection matrix ----

    def compute_projection(self) -> Tensor:
        """Eigendecompose covariance → Q_top [d, d'].

        Q_top contains the top target_embed_dim eigenvectors (largest eigenvalues),
        i.e. the most important PCA directions.

        Returns:
            Q_top: shape [hidden_size, target_embed_dim], orthogonal.
        """
        if self.covariance is None:
            raise RuntimeError("No covariance matrix. Set self.covariance first.")

        d = self.arch.hidden_size
        d_prime = self.arch.target_embed_dim

        eigenvalues, Q = torch.linalg.eigh(self.covariance)  # ascending eigenvalues
        # Take last d' eigenvectors (largest eigenvalues = most important)
        Q_top = Q[:, -d_prime:]  # [d, d']

        # Verify orthogonality: Q_top.T @ Q_top ≈ I
        identity_check = Q_top.T @ Q_top
        off_diag = identity_check - torch.eye(d_prime, device=identity_check.device,
                                               dtype=identity_check.dtype)
        ortho_error = off_diag.abs().max().item()
        if ortho_error > 1e-4:
            print(f"  Warning: Q_top not perfectly orthogonal (max off-diag error={ortho_error:.2e})")

        self.Q_top = Q_top.to(dtype=torch.float32)
        return self.Q_top

    # ---- Phase 2: Project weights ----

    def _project_weight(self, weight: Tensor, key: str) -> Tensor:
        """Apply projection + slicing to a single weight tensor.

        Args:
            weight: Original weight tensor.
            key: State dict key (e.g. "model.layers.0.self_attn.q_proj.weight").

        Returns:
            Projected and sliced weight tensor with target dimensions.
        """
        Q = self.Q_top.to(device=weight.device, dtype=weight.dtype)

        # Determine the rule for this key
        rule = _get_projection_rule(key)

        if rule == "both":
            # Q_top.T @ W @ Q_top  [d', d']
            result = Q.T @ weight @ Q  # [d', d, d] @ [d, d'] → [d', d']
            # Both dimensions now equal d' = target_num_heads * target_head_dim

        elif rule == "input":
            # W @ Q_top  [out, d'] then slice out-dim if needed
            result = weight @ Q  # [out, d] @ [d, d'] → [out, d']
            result = self._slice_attention_output(result, key)

        elif rule == "output":
            # Q_top.T @ W  [d', in] then slice in-dim if needed
            result = Q.T @ weight  # [d', d] @ [d, in] → [d', in]
            result = self._slice_attention_output(result, key)

        elif rule == "embed":
            # W @ Q_top  [V, d']
            result = weight @ Q  # [V, d] @ [d, d'] → [V, d']

        elif rule == "norm":
            # Slice first d' elements
            result = weight[:self.arch.target_embed_dim]

        elif rule == "skip":
            # No projection
            result = weight.clone()

        return result.contiguous()

    def _slice_attention_output(self, weight: Tensor, key: str) -> Tensor:
        """Slice attention head dimensions or FFN intermediate dimensions.

        This is applied AFTER Q projection when the output dimension
        is not in residual stream space (e.g., k_proj, v_proj, gate_proj, up_proj,
        down_proj).

        Args:
            weight: Already Q-projected weight.
            key: State dict key for context.

        Returns:
            Sliced weight.
        """
        orig_out_dim, new_in_dim = weight.shape
        target = self.arch

        # k_proj or v_proj: [num_kv_heads * head_dim, d'] → [target_num_kv_heads * target_head_dim, d']
        if "k_proj" in key or "v_proj" in key:
            # reshape → slice → flatten
            num_kv = self.arch.num_kv_heads
            hd = self.arch.head_dim
            # [num_kv*hd, d'] → [num_kv, hd, d']
            w_reshaped = weight.reshape(num_kv, hd, new_in_dim)
            # Slice to target
            w_sliced = w_reshaped[:target.target_num_kv_heads, :target.target_head_dim, :]
            result = w_sliced.reshape(target.target_num_kv_heads * target.target_head_dim, new_in_dim)

        # q_proj: already handled by "both" rule, but verify
        elif "q_proj" in key:
            # Both-side projection already reduces to [d', d'] = [target_heads*head_dim, d']
            result = weight

        # o_proj: already handled by "both" rule
        elif "o_proj" in key:
            result = weight

        # gate_proj or up_proj: [intermediate, d'] → [target_intermediate, d']
        elif "gate_proj" in key or "up_proj" in key:
            result = weight[:target.target_intermediate_size, :]

        # down_proj: [d', intermediate] → [d', target_intermediate]
        elif "down_proj" in key:
            result = weight[:, :target.target_intermediate_size]

        else:
            result = weight

        return result

    # ---- Phase 3: Build compressed model ----

    def compress(self, original_model: nn.Module,
                 original_config) -> tuple[nn.Module, dict]:
        """Create compressed model with projected weights.

        Args:
            original_model: Original HF model (eval mode).
            original_config: Original HF config.

        Returns:
            (compressed_model, compressed_state_dict)
        """
        if self.Q_top is None:
            self.compute_projection()

        target = self.arch
        state_dict = original_model.state_dict()

        # Build target config
        target_config = self._build_target_config(original_config)

        # Project all weights
        new_state_dict: dict[str, Tensor] = {}
        skipped_keys: list[str] = []

        for key, weight in tqdm(state_dict.items(), desc="Projecting weights"):
            try:
                new_state_dict[key] = self._project_weight(weight, key)
            except Exception as e:
                print(f"  Warning: skipping {key} ({e})")
                skipped_keys.append(key)

        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} keys: {skipped_keys}")

        # Create new model with target config
        with torch.no_grad():
            compressed_model = AutoModelForCausalLM.from_config(
                target_config,
                torch_dtype=original_model.dtype,
            )

        # Load projected weights (strict=False handles missing/mismatched keys)
        missing, unexpected = compressed_model.load_state_dict(
            new_state_dict, strict=False
        )
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        return compressed_model, new_state_dict

    def _build_target_config(self, original_config):
        """Create a config for the compressed model (deep copy from original)."""
        t = self.arch
        target_config = copy.deepcopy(original_config)

        # Override with compressed dimensions
        target_config.hidden_size = t.target_embed_dim
        target_config.num_attention_heads = t.target_num_heads
        target_config.num_key_value_heads = t.target_num_kv_heads
        target_config.head_dim = t.target_head_dim
        target_config.intermediate_size = t.target_intermediate_size
        # num_hidden_layers is handled by pruning, keep original here

        # Ensure untied embeddings (compressed embed ≠ compressed lm_head)
        if hasattr(target_config, "tie_word_embeddings"):
            target_config.tie_word_embeddings = False

        return target_config


def _get_projection_rule(key: str) -> str:
    """Determine projection rule from state dict key."""
    for pattern, rule in LAYER_WEIGHT_RULES:
        if pattern in key:
            return rule
    for pattern, rule in GLOBAL_WEIGHT_RULES:
        if pattern in key:
            return rule
    # Unknown key: skip projection (e.g. non-weight params)
    return "skip"


# ---- Verification ----

def verify_residual_consistency(model: nn.Module,
                                 tokenizer,
                                 device: str = "cuda") -> bool:
    """Verify that the compressed model maintains residual stream consistency.

    Runs a forward pass with a dummy input and checks:
      1. hidden_states shape is consistent throughout all layers
      2. No dimension mismatches in residual connections
      3. Output logits have correct vocab_size dimension

    Args:
        model: Compressed HF model.
        tokenizer: HF tokenizer (for creating dummy input).
        device: Device string.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If any consistency check fails.
    """
    model.eval()
    dummy_input = torch.randint(0, min(tokenizer.vocab_size, 1000),
                                 (1, 16), device=device)

    with torch.no_grad():
        output = model(dummy_input, output_hidden_states=True)

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states.")

    hidden_dim = model.config.hidden_size

    # Check every layer's hidden state has consistent last dimension
    for i, hs in enumerate(hidden_states):
        assert hs.shape[-1] == hidden_dim, \
            f"Layer {i} hidden state dim {hs.shape[-1]} != {hidden_dim}"

    # Check input and output dimensions through a full layer
    # The residual stream should maintain consistent dimension
    for i in range(1, len(hidden_states)):
        prev_dim = hidden_states[i - 1].shape[-1]
        curr_dim = hidden_states[i].shape[-1]
        assert prev_dim == curr_dim, \
            f"Residual dim mismatch: layer {i-1} dim={prev_dim}, layer {i} dim={curr_dim}"

    # Check logits dimension
    assert output.logits.shape[-1] == model.config.vocab_size, \
        f"Logits dim {output.logits.shape[-1]} != vocab_size {model.config.vocab_size}"

    return True
