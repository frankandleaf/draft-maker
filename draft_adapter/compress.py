"""SliceGPT-style width compression with global orthogonal projection Q.

Key insight from the SliceGPT paper (ICLR 2024):
  PCA rotation before slicing deletes the least important principal components,
  not random dimensions. Global unified Q avoids per-layer residual transform
  complexity (Q^T @ Q = I).

Weight projection rules (critical for mathematical correctness):

  Rule "input":  Weights where INPUT is residual stream (d)
                but OUTPUT is NOT (q_proj, k_proj, v_proj, gate_proj, up_proj).
                W_new = W @ Q_top  →  [out, d']  then slice out-dim

  Rule "output": Weights where OUTPUT is residual stream (d)
                 but INPUT is NOT (o_proj, down_proj).
                 W_new = Q_top.T @ W  →  [d', in]  then slice in-dim

  Rule "embed":  Embedding/lm_head weights [V, d].
                 W_new = W @ Q_top  ->  [V, d']

  Rule "norm":   RMSNorm weights [d]. Slice to [d'].
  Rule "head_norm": Q/K norm weights [head_dim]. Slice to [target_head_dim].
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from .calibration import CovarianceAggregator
from .debug_log import get_logger
from .inspect import ModelArchitecture
from .utils import get_dtype


# --- Norm absorption (MUST happen before PCA projection) ---


def _absorb_norms(state_dict: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Fuse RMSNorm gamma weights into adjacent Linear layer weights.

    Why: matrix multiplication Q^T @ (X * gamma) cannot be expressed as
    (Q^T @ X) * gamma' for any element-wise gamma'.  The only correct
    approach is to absorb gamma into the adjacent weight matrices BEFORE
    projection, reducing the RMSNorm to a pure scaling-free normalizer
    (gamma = 1), which IS orthogonally invariant.

    Absorption rules:
      input_layernorm[i]  →  q_proj[i], k_proj[i], v_proj[i]
      post_attn_norm[i]   →  gate_proj[i], up_proj[i]
      model.norm          →  lm_head

    After absorption, all norm weights become ones.
    """
    sd = {k: v.clone() for k, v in state_dict.items()}
    log = get_logger()

    log.section("Norm Absorption: fusing RMSNorm γ → adjacent Linear weights")
    log.info(f"Processing {num_layers} layers + global norm")

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # input_layernorm → q_proj, k_proj, v_proj
        in_ln_key = f"{prefix}.input_layernorm.weight"
        if in_ln_key in sd:
            gamma = sd[in_ln_key]
            log.before(f"L{i} input_layernorm → q/k/v",
                       gamma_norm=gamma.float().norm(), gamma_len=gamma.shape[0])
            for proj in ["q_proj", "k_proj", "v_proj"]:
                w_key = f"{prefix}.self_attn.{proj}.weight"
                if w_key in sd:
                    w_before = sd[w_key].clone()
                    sd[w_key] = sd[w_key] * gamma.unsqueeze(0)
                    log.weight_diff(f"L{i}.{proj}", w_before, sd[w_key])
            sd[in_ln_key] = torch.ones_like(gamma)
            log.info("  → γ set to ones")

        # post_attention_layernorm → gate_proj, up_proj
        post_ln_key = f"{prefix}.post_attention_layernorm.weight"
        if post_ln_key in sd:
            gamma = sd[post_ln_key]
            for proj in ["gate_proj", "up_proj"]:
                w_key = f"{prefix}.mlp.{proj}.weight"
                if w_key in sd:
                    sd[w_key] = sd[w_key] * gamma.unsqueeze(0)
            sd[post_ln_key] = torch.ones_like(gamma)

    # model.norm → lm_head
    if "model.norm.weight" in sd and "lm_head.weight" in sd:
        gamma = sd["model.norm.weight"]
        sd["lm_head.weight"] = sd["lm_head.weight"] * gamma.unsqueeze(0)
        sd["model.norm.weight"] = torch.ones_like(gamma)

    return sd


# --- State dict key patterns and their projection rules ---

# Each entry: (key_suffix_pattern, projection_rule)
# Rules: "both" | "input" | "output" | "embed" | "norm" | "skip"
LAYER_WEIGHT_RULES = [
    ("self_attn.q_proj.weight", "input"),   # project residual side only, slice heads
    ("self_attn.q_proj.bias", "head_slice"),
    ("self_attn.k_proj.weight", "input"),
    ("self_attn.k_proj.bias", "head_slice"),
    ("self_attn.v_proj.weight", "input"),
    ("self_attn.v_proj.bias", "head_slice"),
    ("self_attn.o_proj.weight", "output"),  # project residual side only, slice heads
    ("self_attn.o_proj.bias", "output"),
    ("mlp.gate_proj.weight", "input"),
    ("mlp.up_proj.weight", "input"),
    ("mlp.down_proj.weight", "output"),
    ("self_attn.q_norm.weight", "head_norm"),
    ("self_attn.k_norm.weight", "head_norm"),
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
        self.norm_scale: float = 1.0      # RMSNorm compensation factor

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

        log = get_logger()
        log.section("PCA Projection Matrix")
        log.info(f"Covariance: {self.covariance.shape}, eigenvalues range: "
                 f"[{eigenvalues.min().item():.6f}, {eigenvalues.max().item():.6f}]")
        retained_var = eigenvalues[-d_prime:].sum() / eigenvalues.sum()
        log.info(f"Retained variance: {retained_var*100:.1f}% ({d_prime}/{d} dims)")

        # Verify orthogonality: Q_top.T @ Q_top ≈ I
        identity_check = Q_top.T @ Q_top
        off_diag = identity_check - torch.eye(d_prime, device=identity_check.device,
                                               dtype=identity_check.dtype)
        ortho_error = off_diag.abs().max().item()
        if ortho_error > 1e-4:
            print(f"  Warning: Q_top not perfectly orthogonal (max off-diag error={ortho_error:.2e})")

        self.Q_top = Q_top.to(dtype=torch.float32)

        # RMSNorm scale compensation: PCA keeps high-variance dims,
        # so compressed-space RMS > original-space RMS. RMSNorm would
        # over-normalize without this factor.
        lambda_top = eigenvalues[-d_prime:]
        mean_top = lambda_top.mean().clamp(min=1e-8)
        mean_all = eigenvalues.mean().clamp(min=1e-8)
        self.norm_scale = torch.sqrt(mean_top / mean_all).item()
        print(f"  Q_top shape: {list(Q_top.shape)}, "
              f"norm_scale: {self.norm_scale:.4f}")

        return self.Q_top

    # ---- Phase 2: Project weights ----

    def _project_weight(self, weight: Tensor, key: str) -> Tensor:
        Q = self.Q_top.to(device=weight.device, dtype=weight.dtype)
        rule = _get_projection_rule(key)

        # --- 1D tensors (biases, norms) ---
        if weight.dim() == 1:
            if rule == "norm":
                # After norm absorption, gamma=1 in original model.
                # In compressed model, use ones(target_embed_dim) — pure
                # normalization without scaling, which is orthogonal-invariant.
                return torch.ones(self.arch.target_embed_dim,
                                  device=weight.device, dtype=weight.dtype)
            elif rule == "head_norm":
                # head_dim is frozen, no slicing needed
                return weight.clone()
            elif rule == "head_slice":
                return self._slice_head_bias(weight, key)
            elif rule == "input":
                return weight[:self.arch.target_intermediate_size]
            elif rule == "output":
                return (Q.T @ weight).contiguous()
            else:
                return weight.clone()

        # --- 2D weight matrices ---
        if rule == "input":
            result = weight @ Q  # [out, d'] then slice out-dim
            result = self._slice_output_dim(result, key)
        elif rule == "output":
            result = Q.T @ weight  # [d', in] then slice in-dim
            result = self._slice_input_dim(result, key)
        elif rule == "embed":
            result = weight @ Q  # [V, d']
        elif rule == "skip":
            result = weight.clone()
        else:
            result = weight.clone()

        return result.contiguous()

    # ---- Attention head / FFN slicing helpers ----
    # head_dim is FROZEN: only num_heads / num_kv_heads are reduced

    def _slice_output_dim(self, weight: Tensor, key: str) -> Tensor:
        """Slice OUTPUT dimension. head_dim is FROZEN, only reduce head count."""
        target = self.arch
        hd = self.arch.head_dim  # original head_dim, unchanged

        if "q_proj" in key:
            nh = self.arch.num_attention_heads
            w = weight.reshape(nh, hd, -1)
            w = w[:target.target_num_heads, :, :]
            return w.reshape(target.target_num_heads * hd, -1)

        if "k_proj" in key or "v_proj" in key:
            nk = self.arch.num_kv_heads
            w = weight.reshape(nk, hd, -1)
            w = w[:target.target_num_kv_heads, :, :]
            return w.reshape(target.target_num_kv_heads * hd, -1)

        if "gate_proj" in key or "up_proj" in key:
            return weight[:target.target_intermediate_size, :]

        return weight

    def _slice_input_dim(self, weight: Tensor, key: str) -> Tensor:
        """Slice INPUT dimension. head_dim is FROZEN, only reduce head count."""
        target = self.arch
        hd = self.arch.head_dim

        if "o_proj" in key:
            nh = self.arch.num_attention_heads
            w = weight.reshape(-1, nh, hd)
            w = w[:, :target.target_num_heads, :]
            return w.reshape(-1, target.target_num_heads * hd)

        if "down_proj" in key:
            return weight[:, :target.target_intermediate_size]

        return weight

    def _slice_head_bias(self, weight: Tensor, key: str) -> Tensor:
        """Slice bias terms in attention head space. head_dim is frozen."""
        target = self.arch
        hd = self.arch.head_dim
        if "q_proj" in key:
            nh = self.arch.num_attention_heads
            w = weight.reshape(nh, hd)
            w = w[:target.target_num_heads, :]
            return w.reshape(-1)
        if "k_proj" in key or "v_proj" in key:
            nk = self.arch.num_kv_heads
            w = weight.reshape(nk, hd)
            w = w[:target.target_num_kv_heads, :]
            return w.reshape(-1)
        return weight

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

        # ---- Norm absorption: fuse RMSNorm γ into adjacent Linear weights ----
        # This MUST happen before PCA projection. See docstring for proof.
        state_dict = _absorb_norms(state_dict, target.num_layers)

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

        # Count projected weights by rule
        log = get_logger()
        rule_counts: dict[str, int] = {}
        for key in new_state_dict:
            rule_counts[_get_projection_rule(key)] = \
                rule_counts.get(_get_projection_rule(key), 0) + 1
        log.section("Weight Projection Summary")
        for rule, count in sorted(rule_counts.items()):
            log.info(f"  {rule}: {count} weights")

        # Handle tied embeddings: if original had tie_word_embeddings=True,
        # the state dict has no lm_head.weight. The target config has
        # tie_word_embeddings=False, so we must provide lm_head.weight
        # explicitly (copy from projected embed_tokens).
        tied = getattr(original_config, "tie_word_embeddings", False)
        if tied and "lm_head.weight" not in new_state_dict:
            if "model.embed_tokens.weight" in new_state_dict:
                new_state_dict["lm_head.weight"] = new_state_dict[
                    "model.embed_tokens.weight"].clone()
                print("  Untied embeddings: copied embed_tokens.weight → lm_head.weight")

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
    # Use model's own device to avoid device mismatch
    model_device = next(model.parameters()).device
    dummy_input = torch.randint(0, min(tokenizer.vocab_size, 1000),
                                 (1, 16), device=model_device)

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
