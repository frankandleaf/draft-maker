"""SVD Hybrid Compression: channel scoring + slicing + low-rank decomposition.

Combines two SVD-based strategies:
  Method 3 — SVD-Guided Channel Slicing:
    Score hidden_size dimensions using SVD of activation covariance,
    permute dimensions by importance, and slice to target size.
    No norm absorption needed (permutation is element-wise, RMSNorm-compatible).

  Method 1 — SVD Low-Rank Decomposition:
    After slicing, further decompose remaining Linear layers via truncated
    SVD, replacing each W[m,n] with two smaller matrices at rank r.

Pipeline:
  es² × rank_factor × ls → target compression ratio.
  es=0.5, rank_factor=0.5, ls=0.75 → ~9.4%.
"""

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .calibration import collect_layer_outputs
from .debug_log import get_logger
from .inspect import ModelArchitecture


# ============================================================================
# DecomposedLinear: replaces nn.Linear with two low-rank linear layers
# ============================================================================


class DecomposedLinear(nn.Module):
    """Low-rank decomposition of nn.Linear W[m,n] ≈ W_out[m,r] @ W_in[r,n].

    W = U @ S @ V^T  →  W_out = U @ diag(sqrt(S)),  W_in = diag(sqrt(S)) @ V^T
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.proj_in = nn.Linear(in_features, rank, bias=False)
        self.proj_out = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_out(self.proj_in(x))


# ============================================================================
# Phase A: SVD Channel Scoring (Method 3)
# ============================================================================


class SVDChannelScorer:
    """Score hidden_size channels using SVD-derived importance from activations.

    Approach:
      1. Collect hidden states from calibration forward passes.
      2. Build global covariance C = E[x·x^T]  [d, d].
      3. SVD of C: C = V @ Λ @ V^T.
      4. Per-channel score = sum of |V[i,k]|² weighted by Λ_k for top-d' PCs.
         This captures how much each raw dimension contributes to the
         principal directions of highest variance.
      5. NOT the same as per-dimension variance (which is just the diagonal of C).
    """

    def __init__(self, arch: ModelArchitecture):
        self.arch = arch
        self.d = arch.hidden_size
        self.channel_scores: Tensor | None = None  # [d]
        self.channel_order: list[int] | None = None  # sorted by importance desc

    @torch.no_grad()
    def score_channels(self, model: nn.Module,
                        input_ids: Tensor,
                        chunk_size: int = 4) -> Tensor:
        """Compute per-channel importance scores from calibration data.

        Returns:
            channel_scores: [d] tensor, higher = more important.
        """
        d = self.arch.hidden_size
        num_layers = self.arch.num_layers
        total = input_ids.shape[0]

        # Step 1: Collect hidden states from all layers
        cov_sum = torch.zeros(d, device='cpu')
        total_tokens = 0

        for start in tqdm(range(0, total, chunk_size),
                           desc="SVD channel scoring"):
            end = min(start + chunk_size, total)
            batch = input_ids[start:end]

            layer_outputs = collect_layer_outputs(
                model, batch,
                layer_indices=list(range(num_layers)),
            )

            for layer_idx in range(num_layers):
                for batch_output in layer_outputs[layer_idx]:
                    # batch_output: [seq_len, d] on CPU
                    x = batch_output.float()
                    cov_sum += x.pow(2).sum(dim=0)  # per-dim sum of squares
                    total_tokens += x.shape[0]

        # Per-dimension variance (diagonal of covariance)
        if total_tokens > 0:
            per_dim_var = cov_sum / total_tokens  # [d]
        else:
            per_dim_var = torch.ones(d)

        # Step 2: Full covariance for SVD-derived scores
        # To keep memory manageable, we compute SVD of a downsampled covariance
        # First, compute full covariance in chunks
        full_cov = torch.zeros(d, d, device='cpu')
        count = 0
        for start in tqdm(range(0, total, chunk_size),
                           desc="Computing covariance"):
            end = min(start + chunk_size, total)
            batch = input_ids[start:end]
            layer_outputs = collect_layer_outputs(
                model, batch,
                layer_indices=list(range(num_layers)),
            )
            for layer_idx in range(num_layers):
                for batch_output in layer_outputs[layer_idx]:
                    x = batch_output.float()
                    full_cov += (x.T @ x)
                    count += x.shape[0]

        if count > 0:
            full_cov /= count

        # Step 3: Eigendecompose covariance → V, Λ
        # Use float64 for numerical stability
        full_cov_64 = full_cov.to(torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh(full_cov_64)
        # eigenvalues are ascending, eigenvectors: V[:, i] is i-th eigenvector

        # Step 4: Score each raw dimension by its alignment with top PCs
        # Top K = target_embed_dim (number of PCs = number of dims to keep)
        K = self.arch.target_embed_dim
        # Take last K eigenvectors (largest eigenvalues = most important PCs)
        top_K_eigvecs = eigenvectors[:, -K:]  # [d, K]
        top_K_eigvals = eigenvalues[-K:]        # [K]

        # Score(i) = Σ_{k} λ_k · |V[i, k]|²
        # Weight each dimension by how much it contributes to top PCs
        scores = (top_K_eigvecs.pow(2) @ top_K_eigvals).float()  # [d]

        self.channel_scores = scores

        log = get_logger()
        log.section("SVD Channel Scoring")
        log.info(f"Top K: {K} of {d} principal components")
        log.info(f"Retained variance (top {K} PCs): "
                 f"{eigenvalues[-K:].sum().item() / eigenvalues.sum().item() * 100:.1f}%")
        log.info(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

        # Show top/bottom scored channels
        order = scores.argsort(descending=True)
        log.info(f"Top 5 channels: {order[:5].tolist()}")
        log.info(f"Bottom 5 channels: {order[-5:].tolist()}")

        return scores

    def get_channel_order(self) -> list[int]:
        """Return dimension indices sorted by importance (descending)."""
        if self.channel_scores is None:
            raise RuntimeError("Call score_channels() first.")
        return self.channel_scores.argsort(descending=True).tolist()


# ============================================================================
# Phase B: SVD Low-Rank Decomposition (Method 1)
# ============================================================================


class SVDDecomposer:
    """Decompose nn.Linear layers into low-rank DecomposedLinear form.

    For each weight matrix W [m, n], compute truncated SVD:
      W ≈ U_r @ S_r @ V_r^T
    where r = rank_factor * min(m, n) controls compression.
    """

    def __init__(self, rank_factor: float = 0.5):
        if not 0 < rank_factor <= 1:
            raise ValueError(f"rank_factor must be in (0, 1], got {rank_factor}")
        self.rank_factor = rank_factor

    def decompose_weight(self, weight: Tensor, name: str = "",
                          bias: Tensor | None = None) -> nn.Module:
        """Decompose a single weight matrix into DecomposedLinear.

        Args:
            weight: [out_features, in_features] weight matrix.
            name: Debug name for the weight.
            bias: Optional bias tensor [out_features].

        Returns:
            DecomposedLinear module with decomposed weights.
        """
        out_f, in_f = weight.shape
        max_rank = min(out_f, in_f)
        target_rank = max(1, int(max_rank * self.rank_factor))
        target_rank = min(target_rank, max_rank)

        # Use float32 for numerical stability
        w = weight.float()

        # Randomized SVD for large matrices
        if min(out_f, in_f) > 2048:
            U, S, Vt = torch.svd_lowrank(w, q=min(target_rank * 3, max_rank),
                                          niter=2)
        else:
            U, S, V = torch.svd(w)
            Vt = V.t()

        # Truncate to target rank
        U_r = U[:, :target_rank]           # [out_f, r]
        S_r = S[:target_rank]               # [r]
        Vt_r = Vt[:target_rank, :]          # [r, in_f]

        # Absorb S equally into both sides: W ≈ (U @ diag(sqrt(S))) @ (diag(sqrt(S)) @ V^T)
        S_sqrt = S_r.sqrt()
        W_out = U_r * S_sqrt.unsqueeze(0)      # [out_f, r]
        W_in = Vt_r * S_sqrt.unsqueeze(1)       # [r, in_f]

        mod = DecomposedLinear(in_f, out_f, target_rank,
                               bias=(bias is not None))
        mod.proj_in.weight.data.copy_(W_in)
        mod.proj_out.weight.data.copy_(W_out)
        if bias is not None:
            mod.proj_out.bias.data.copy_(bias)

        log = get_logger()
        original_params = out_f * in_f
        decomposed_params = target_rank * (out_f + in_f)
        log.info(f"  Decomposed {name}: {out_f}×{in_f} ({original_params:,} params) "
                 f"→ rank {target_rank} ({decomposed_params:,} params, "
                 f"{decomposed_params/original_params*100:.1f}%)")

        return mod


# ============================================================================
# Combined SVD Compressor
# ============================================================================


class SVDCompressor:
    """Build compressed model via SVD channel slicing + optional decomposition.

    Pipeline:
      1. Score hidden_size channels (SVDChannelScorer).
      2. Select top-d' channels, permute + slice all weights.
      3. Build standard HF model with reduced dimensions.
      4. Optionally: decompose remaining Linear layers via SVD.
    """

    def __init__(self, arch: ModelArchitecture,
                 scorer: SVDChannelScorer | None = None,
                 decomposer: SVDDecomposer | None = None):
        self.arch = arch
        self.scorer = scorer
        self.decomposer = decomposer
        self._channel_order: list[int] | None = None

    # ==================================================================
    # Channel selection + permutation mapping
    # ==================================================================

    def _build_channel_maps(self) -> tuple[dict[int, int], list[int]]:
        """Build forward and backward channel mapping for permutation+slice.

        Returns:
            fwd_map: old_dim → new_dim (or None if dropped)
            dropped: list of old dimensions being dropped
        """
        if self.scorer is None:
            raise RuntimeError("No scorer provided. Assign self.scorer first.")

        d = self.arch.hidden_size
        d_prime = self.arch.target_embed_dim

        if self._channel_order is None:
            self._channel_order = self.scorer.get_channel_order()

        kept = self._channel_order[:d_prime]
        fwd_map = {old: new for new, old in enumerate(kept)}
        dropped = self._channel_order[d_prime:]

        log = get_logger()
        log.section("Channel Selection")
        log.info(f"Target dims: {d_prime}/{d} ({d_prime/d*100:.0f}%)")
        log.info(f"Kept channels: {kept[:10]}...")
        log.info(f"Dropped channels: {dropped[:10]}...")

        return fwd_map, dropped

    # ==================================================================
    # Weight projection (permute + slice, no rotation)
    # ==================================================================

    def _get_layer_rule(self, key: str) -> str:
        """Determine how a state dict key relates to hidden_size dimension."""
        if "q_proj" in key or "k_proj" in key or "v_proj" in key:
            return "attn_input"       # [nh*hd, d] — col=d is residual
        if "o_proj" in key:
            return "attn_output"      # [d, nh*hd] — row=d is residual
        if "gate_proj" in key or "up_proj" in key:
            return "ffn_input"        # [ff, d] — col=d is residual
        if "down_proj" in key:
            return "ffn_output"       # [d, ff] — row=d is residual
        if "embed_tokens" in key or "lm_head" in key:
            return "embed"             # [V, d] — col=d is residual
        if "q_norm" in key or "k_norm" in key:
            return "head_norm"         # [nh*hd] or [hd]
        if "norm" in key:
            return "norm"              # [d] 1D
        return "skip"

    def _slice_weight(self, weight: Tensor, key: str,
                       fwd_map: dict[int, int]) -> Tensor:
        """Permute + slice a single weight tensor.

        For 'attn_input' / 'ffn_input' / 'embed': slice columns (dim=1)
        For 'attn_output' / 'ffn_output': slice rows (dim=0)
        For 'norm': slice entries
        For 'head_norm' / 'skip': copy unchanged
        """
        rule = self._get_layer_rule(key)

        if rule == "skip":
            return weight.clone()

        if weight.dim() == 1:
            if rule == "head_norm":
                hd = self.arch.head_dim
                if weight.numel() == hd:
                    return weight.clone()
                # Per-head weight [nh*hd]: slice to keep top heads
                nh = self.arch.num_attention_heads
                if weight.numel() == nh * hd:
                    w = weight.reshape(nh, hd)
                    w = w[:self.arch.target_num_heads, :]
                    return w.reshape(-1)
                # kv-head weight [nk*hd]
                nk = self.arch.num_kv_heads
                if weight.numel() == nk * hd:
                    w = weight.reshape(nk, hd)
                    w = w[:self.arch.target_num_kv_heads, :]
                    return w.reshape(-1)
                return weight.clone()
            if rule == "norm":
                d_prime = self.arch.target_embed_dim
                result = weight.new_zeros(d_prime)
                for old_idx, new_idx in fwd_map.items():
                    result[new_idx] = weight[old_idx]
                return result
            if rule == "attn_input":
                return self._slice_output_dim_bias(weight, key)
            if rule == "attn_output":
                return self._slice_residual_1d(weight, fwd_map)
            if rule == "ffn_input":
                return self._slice_output_dim_bias(weight, key)
            if rule == "ffn_output":
                return self._slice_residual_1d(weight, fwd_map)
            return weight.clone()

        # 2D weights
        if rule in ("attn_input", "ffn_input", "embed"):
            # Slice columns (dim=-1, the input/residual dimension)
            out_dim = weight.shape[0]
            in_dim = weight.shape[1]
            d_prime = self.arch.target_embed_dim

            if rule in ("attn_input", "ffn_input"):
                # Also need to slice the output dimension
                result = weight.new_zeros(out_dim, d_prime)
                for old_idx, new_idx in fwd_map.items():
                    result[:, new_idx] = weight[:, old_idx]
                # Slice output dim
                result = self._slice_output_dim(result, key, rule)
            else:  # embed
                result = weight.new_zeros(out_dim, d_prime)
                for old_idx, new_idx in fwd_map.items():
                    result[:, new_idx] = weight[:, old_idx]

            return result

        if rule in ("attn_output", "ffn_output"):
            # Slice rows (dim=0, the output/residual dimension)
            out_dim = weight.shape[0]
            in_dim = weight.shape[1]
            d_prime = self.arch.target_embed_dim

            result = weight.new_zeros(d_prime, in_dim)
            for old_idx, new_idx in fwd_map.items():
                result[new_idx, :] = weight[old_idx, :]
            # Slice input dim
            result = self._slice_input_dim(result, key, rule)

            return result

        return weight.clone()

    def _slice_output_dim(self, weight: Tensor, key: str,
                           rule: str) -> Tensor:
        """After slicing residual dim, also slice the non-residual output dim."""
        t = self.arch
        hd = self.arch.head_dim

        if "q_proj" in key:
            return weight[:t.target_num_heads * hd, :]

        if "k_proj" in key or "v_proj" in key:
            return weight[:t.target_num_kv_heads * hd, :]

        if rule == "ffn_input":
            return weight[:t.target_intermediate_size, :]

        return weight

    def _slice_input_dim(self, weight: Tensor, key: str,
                          rule: str) -> Tensor:
        """After slicing residual dim, also slice the non-residual input dim."""
        t = self.arch
        hd = self.arch.head_dim

        if "o_proj" in key:
            # Always slice to target_num_heads
            target_head_dim = t.target_num_heads * hd
            return weight[:, :target_head_dim]

        if rule == "ffn_output":
            return weight[:, :t.target_intermediate_size]

        return weight

    def _slice_output_dim_bias(self, weight: Tensor, key: str) -> Tensor:
        """Slice bias of input-rule weight (q/k/v/gate/up bias)."""
        t = self.arch
        hd = self.arch.head_dim
        if "q_proj" in key:
            return weight[:t.target_num_heads * hd]
        if "k_proj" in key or "v_proj" in key:
            return weight[:t.target_num_kv_heads * hd]
        if "gate_proj" in key or "up_proj" in key:
            return weight[:t.target_intermediate_size]
        return weight.clone()

    def _slice_residual_1d(self, weight: Tensor,
                            fwd_map: dict[int, int]) -> Tensor:
        """Slice 1D tensor on the residual stream dimension (o_proj/down bias)."""
        d_prime = self.arch.target_embed_dim
        result = weight.new_zeros(d_prime)
        for old_idx, new_idx in fwd_map.items():
            if old_idx < weight.numel():
                result[new_idx] = weight[old_idx]
        return result

    # ==================================================================
    # Build compressed model (Phase A: channel slicing)
    # ==================================================================

    @torch.no_grad()
    def compress(self, model: nn.Module,
                  original_config,
                  fwd_map: dict[int, int] | None = None) -> nn.Module:
        """Build compressed model via channel permutation + slicing.

        No PCA rotation. No norm absorption. Permutation preserves RMSNorm.

        Args:
            model: Original HF model.
            original_config: Original HF config.
            fwd_map: Pre-computed mapping (old_idx → new_idx).

        Returns:
            Compressed HF model with reduced hidden_size.
        """
        if fwd_map is None:
            fwd_map, _ = self._build_channel_maps()

        t = self.arch
        state_dict = model.state_dict()

        # Move original to CPU
        original_device = model.device if hasattr(model, 'device') else 'cpu'
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build target config + model
        target_config = self._build_target_config(original_config)
        with torch.no_grad():
            compressed_model = AutoModelForCausalLM.from_config(
                target_config,
                torch_dtype=model.dtype,
            )
        target_sd = compressed_model.state_dict()

        skipped: list[str] = []
        for key, weight in tqdm(state_dict.items(), desc="Slicing weights"):
            if key not in target_sd:
                skipped.append(key)
                continue
            try:
                sliced = self._slice_weight(weight.cpu(), key, fwd_map)
                target_sd[key].copy_(sliced.to(target_sd[key].dtype))
            except Exception as e:
                print(f"  Warning: skipping {key} ({e})")
                skipped.append(key)

        if skipped:
            print(f"  Skipped {len(skipped)} keys: {skipped}")

        # Handle tied embeddings
        if getattr(original_config, "tie_word_embeddings", False) \
                and "lm_head.weight" in target_sd:
            target_sd["lm_head.weight"].copy_(
                target_sd["model.embed_tokens.weight"]
            )

        compressed_model = compressed_model.to(original_device)
        return compressed_model

    # ==================================================================
    # Phase B: SVD low-rank decomposition
    # ==================================================================

    @torch.no_grad()
    def decompose_model(self, model: nn.Module) -> nn.Module:
        """Replace all attention + FFN Linear layers with DecomposedLinear.

        Iterates through transformer layers and replaces:
          - q_proj, k_proj, v_proj, o_proj
          - gate_proj, up_proj, down_proj

        Embedding layers and norms are left unchanged.
        Head dimensions (q_norm, k_norm) are left unchanged.

        Preserves bias on o_proj.
        """
        if self.decomposer is None:
            raise RuntimeError("No decomposer set. Assign self.decomposer first.")

        nl = model.config.num_hidden_layers
        log = get_logger()
        log.section(f"SVD Decomposition (rank_factor={self.decomposer.rank_factor})")

        original_params = 0
        decomposed_params = 0

        for i in tqdm(range(nl), desc="Decomposing layers"):
            layer = model.model.layers[i]

            # Attention projections
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                orig_linear = getattr(layer.self_attn, name)
                if not isinstance(orig_linear, nn.Linear):
                    continue
                w = orig_linear.weight.data
                has_bias = orig_linear.bias is not None
                bias = orig_linear.bias.data.clone() if has_bias else None

                decomp = self.decomposer.decompose_weight(
                    w, f"L{i}.attn.{name}", bias=bias)
                setattr(layer.self_attn, name, decomp)

                original_params += w.numel()
                decomposed_params += sum(p.numel() for p in decomp.parameters())

            # FFN projections
            for name in ["gate_proj", "up_proj", "down_proj"]:
                if not hasattr(layer.mlp, name):
                    continue
                orig_linear = getattr(layer.mlp, name)
                if not isinstance(orig_linear, nn.Linear):
                    continue
                w = orig_linear.weight.data

                decomp = self.decomposer.decompose_weight(w,
                                                          f"L{i}.mlp.{name}")
                setattr(layer.mlp, name, decomp)

                original_params += w.numel()
                decomposed_params += sum(p.numel() for p in decomp.parameters())

        total_orig = original_params
        total_decomp = decomposed_params
        if total_orig > 0:
            log.info(f"Decomposition summary: {total_orig:,} → {total_decomp:,} params "
                     f"({total_decomp/total_orig*100:.1f}%)")

        return model

    def _build_target_config(self, original_config):
        """Create config for sliced model."""
        t = self.arch
        cfg = copy.deepcopy(original_config)

        cfg.hidden_size = t.target_embed_dim
        cfg.num_attention_heads = t.target_num_heads
        cfg.num_key_value_heads = t.target_num_kv_heads
        cfg.head_dim = t.target_head_dim
        cfg.intermediate_size = t.target_intermediate_size

        if hasattr(cfg, "tie_word_embeddings"):
            cfg.tie_word_embeddings = False

        return cfg


# ============================================================================
# Build SVD-decomposed export model (custom architecture for Phase B output)
# ============================================================================


class DraftDecoderLayer(nn.Module):
    """Custom decoder layer that wraps decomposed linear layers.

    Mirrors the structure of standard llama/qwen2 decoder layers but
    uses DecomposedLinear for all weight matrices.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        hd = config.head_dim if hasattr(config, 'head_dim') else (
            config.hidden_size // config.num_attention_heads
        )
        nh = config.num_attention_heads
        nk = config.num_key_value_heads
        ff = config.intermediate_size
        d = config.hidden_size

        self.self_attn = _DraftAttention(config, layer_idx)
        self.mlp = _DraftMLP(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids,
                                     past_key_value=past_key_value,
                                     output_attentions=output_attentions,
                                     use_cache=use_cache)
        if isinstance(attn_output, tuple):
            attn_output, *rest = attn_output
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class _DraftAttention(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        d = config.hidden_size
        nh = config.num_attention_heads
        nk = config.num_key_value_heads
        hd = config.head_dim if hasattr(config, 'head_dim') else (d // nh)

        self.q_proj = nn.Linear(d, nh * hd, bias=False)
        self.k_proj = nn.Linear(d, nk * hd, bias=False)
        self.v_proj = nn.Linear(d, nk * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)

        self.num_heads = nh
        self.num_kv_heads = nk
        self.head_dim = hd
        self.num_kv_groups = nh // nk

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class _DraftMLP(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        d = config.hidden_size
        ff = config.intermediate_size
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


# --- RoPE helpers (compatible with llama/qwen2/qwen3) ---


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat key/value states for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding with broadcasting."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ============================================================================
# Verification
# ============================================================================


def verify_svd_consistency(model: nn.Module,
                            tokenizer,
                            device: str = "cuda") -> bool:
    """Verify that the SVD-compressed model produces valid outputs."""
    model.eval()
    model_device = next(model.parameters()).device
    dummy_input = torch.randint(0, min(tokenizer.vocab_size, 1000),
                                 (1, 16), device=model_device)

    with torch.no_grad():
        output = model(dummy_input, output_hidden_states=True)

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states.")

    hidden_dim = model.config.hidden_size
    for i, hs in enumerate(hidden_states):
        assert hs.shape[-1] == hidden_dim, \
            f"Layer {i} hidden state dim {hs.shape[-1]} != {hidden_dim}"

    for i in range(1, len(hidden_states)):
        assert hidden_states[i-1].shape[-1] == hidden_states[i].shape[-1], \
            f"Residual dim mismatch at layer {i}"

    assert output.logits.shape[-1] == model.config.vocab_size, \
        f"Logits dim {output.logits.shape[-1]} != vocab_size {model.config.vocab_size}"

    return True
