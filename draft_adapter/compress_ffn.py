"""Swift-SVD: activation-aware FFN + attention head pruning.

No PCA rotation of residual stream. No global Q.
Architecture-preserving — same model_type, vLLM compatible.

Two compression ops:
  1. FFN: per-neuron importance → prune intermediate_size
  2. Attention: per-head magnitude → prune num_heads (head_dim frozen)

Why this is lower-risk than SliceGPT:
  - Residual stream basis unchanged — no rotation
  - head_dim frozen — RoPE untouched
  - Per-weight importance, not global PCA
  - Each weight compressed independently
"""

import copy
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from .debug_log import get_logger
from .inspect import ModelArchitecture


class SwiftSVDCompressor:
    """Activation-aware pruning: FFN neurons + attention heads."""

    def __init__(self, arch: ModelArchitecture):
        self.arch = arch
        self.ffn_importance: dict[int, Tensor] = {}    # layer → [intermediate]
        self.head_importance: dict[int, Tensor] = {}    # layer → [num_heads]
        self.kv_importance: dict[int, Tensor] = {}     # layer → [num_kv_heads]

    # =================================================================
    # Phase 1a: Attention head importance (output magnitude)
    # =================================================================
    @torch.no_grad()
    def compute_head_importance(self, model, input_ids: Tensor,
                                chunk_size: int = 4) -> None:
        """Score each attention head by its output contribution magnitude."""
        nl = self.arch.num_layers
        nh = self.arch.num_attention_heads
        nk = self.arch.num_kv_heads
        hd = self.arch.head_dim
        device = input_ids.device

        q_accum = {i: torch.zeros(nh, device=device) for i in range(nl)}
        kv_accum = {i: torch.zeros(nk, device=device) for i in range(nl)}

        for start in tqdm(range(0, input_ids.shape[0], chunk_size),
                          desc="Head importance"):
            end = min(start + chunk_size, input_ids.shape[0])
            batch = input_ids[start:end]

            # Hook o_proj input → measure per-head contribution
            o_in: dict[int, list[Tensor]] = {i: [] for i in range(nl)}
            v_out: dict[int, list[Tensor]] = {i: [] for i in range(nl)}

            def hook_o(idx):
                def fn(m, inp, out):
                    o_in[idx].append(inp[0].detach())
                return fn
            def hook_v(idx):
                def fn(m, inp, out):
                    v_out[idx].append(out.detach())
                return fn

            hooks = []
            for i in range(nl):
                hooks.append(model.model.layers[i].self_attn.o_proj
                            .register_forward_hook(hook_o(i)))
                hooks.append(model.model.layers[i].self_attn.v_proj
                            .register_forward_hook(hook_v(i)))

            model(batch)
            for h in hooks:
                h.remove()

            for i in range(nl):
                for x in o_in[i]:
                    # [B, S, nh*hd] → [B, S, nh, hd]
                    a = x.reshape(-1, nh, hd)
                    q_accum[i] += a.float().norm(dim=-1).mean(dim=[0, 1])
                for x in v_out[i]:
                    a = x.reshape(-1, nk, hd)
                    kv_accum[i] += a.float().norm(dim=-1).mean(dim=[0, 1])

        log = get_logger()
        log.section("Attention Head Importance")
        for i in range(nl):
            self.head_importance[i] = q_accum[i]
            self.kv_importance[i] = kv_accum[i]
            if i < 3:
                top_q = q_accum[i].topk(min(3, nh)).indices.tolist()
                top_kv = kv_accum[i].topk(min(2, nk)).indices.tolist()
                log.info(f"L{i}: Q top-heads {top_q}, KV top-heads {top_kv}")

    # =================================================================
    # Phase 1b: FFN neuron importance (activation × output weight norm)
    # =================================================================
    @torch.no_grad()
    def compute_ffn_importance(self, model, input_ids: Tensor,
                               chunk_size: int = 4) -> None:
        """Swift-SVD style: activation magnitude × output column norm."""
        nl = self.arch.num_layers
        ff = self.arch.intermediate_size
        device = input_ids.device
        accum = {i: torch.zeros(ff, device=device) for i in range(nl)}

        for start in tqdm(range(0, input_ids.shape[0], chunk_size),
                          desc="FFN importance"):
            end = min(start + chunk_size, input_ids.shape[0])
            batch = input_ids[start:end]
            gate_out: dict[int, list[Tensor]] = {i: [] for i in range(nl)}

            def hook(idx):
                def fn(m, inp, out):
                    gate_out[idx].append(out.detach())
                return fn

            hooks = []
            for i in range(nl):
                hooks.append(model.model.layers[i].mlp.gate_proj
                            .register_forward_hook(hook(i)))
            model(batch)
            for h in hooks:
                h.remove()

            for i in range(nl):
                for out in gate_out[i]:
                    accum[i] += out.float().pow(2).mean(dim=[0, 1])

        log = get_logger()
        log.section("FFN Neuron Importance (Swift-SVD)")
        for i in range(nl):
            imp = accum[i]
            # Weight by down_proj column norm
            down_w = model.model.layers[i].mlp.down_proj.weight.data
            col_norm = down_w.float().norm(dim=0)
            col_norm = col_norm / col_norm.max().clamp(min=1e-8)
            imp = imp * col_norm
            self.ffn_importance[i] = imp
            if i < 3:
                top3 = imp.topk(3).indices.tolist()
                log.info(f"L{i}: top neurons {top3}, "
                         f"scores {[f'{imp[j]:.4f}' for j in top3]}")

    # =================================================================
    # Phase 2: Prune
    # =================================================================
    def prune(self, model, original_config) -> tuple:
        """Build compressed model: slice FFN + attention by importance."""
        t = self.arch
        nl = self.arch.num_layers
        t_ff = t.target_intermediate_size
        t_nh = t.target_num_heads
        t_nk = t.target_num_kv_heads
        hd = self.arch.head_dim  # FROZEN
        nh = self.arch.num_attention_heads
        nk = self.arch.num_kv_heads

        cfg = copy.deepcopy(original_config)
        cfg.intermediate_size = t_ff
        cfg.num_attention_heads = t_nh
        cfg.num_key_value_heads = t_nk
        if hasattr(cfg, 'tie_word_embeddings'):
            cfg.tie_word_embeddings = False

        # Move original model to CPU to free GPU memory
        target_device = model.device
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cm = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=model.dtype).to(target_device)
        src = model.state_dict()
        dst = cm.state_dict()
        log = get_logger()
        log.section(f"Prune: FFN {self.arch.intermediate_size}→{t_ff}, "
                    f"heads {nh}→{t_nh}/{nk}→{t_nk}")

        for key in src:
            if key not in dst:
                continue

            parts = key.split(".")
            # --- Global weights: copy directly ---
            if "layers" not in parts:
                dst[key].copy_(src[key])
                continue

            lidx = int(parts[2])

            # --- Attention weights ---
            if "self_attn" in key:
                if "q_proj.weight" in key:
                    imp = self.head_importance.get(lidx)
                    idx = imp.topk(t_nh).indices.sort()[0] if imp is not None \
                        else torch.arange(t_nh, device=src[key].device)
                    w = src[key].reshape(nh, hd, -1)[idx]
                    dst[key].copy_(w.reshape(t_nh * hd, -1))
                    continue

                if ("k_proj.weight" in key or "v_proj.weight" in key):
                    imp = self.kv_importance.get(lidx)
                    idx = imp.topk(t_nk).indices.sort()[0] if imp is not None \
                        else torch.arange(t_nk, device=src[key].device)
                    w = src[key].reshape(nk, hd, -1)[idx]
                    dst[key].copy_(w.reshape(t_nk * hd, -1))
                    continue

                if "o_proj.weight" in key:
                    imp = self.head_importance.get(lidx)
                    idx = imp.topk(t_nh).indices.sort()[0] if imp is not None \
                        else torch.arange(t_nh, device=src[key].device)
                    w = src[key].reshape(-1, nh, hd)[:, idx]
                    dst[key].copy_(w.reshape(-1, t_nh * hd))
                    continue

                if "q_norm.weight" in key:
                    if src[key].numel() == hd:
                        dst[key].copy_(src[key])
                    else:
                        imp = self.head_importance.get(lidx)
                        idx = imp.topk(t_nh).indices.sort()[0] if imp is not None \
                            else torch.arange(t_nh, device=src[key].device)
                        w = src[key].reshape(nh, hd)[idx].reshape(-1)
                        dst[key].copy_(w)
                    continue

                if "k_norm.weight" in key:
                    if src[key].numel() == hd:
                        dst[key].copy_(src[key])
                    else:
                        imp = self.kv_importance.get(lidx)
                        idx = imp.topk(t_nk).indices.sort()[0] if imp is not None \
                            else torch.arange(t_nk, device=src[key].device)
                        w = src[key].reshape(nk, hd)[idx].reshape(-1)
                        dst[key].copy_(w)
                    continue

            # --- FFN weights ---
            if "mlp" in key and "weight" in key:
                imp = self.ffn_importance.get(lidx)
                idx = imp.topk(t_ff).indices.sort()[0] if imp is not None \
                    else torch.arange(t_ff, device=src[key].device)
                if "down_proj" in key:
                    dst[key].copy_(src[key][:, idx])
                else:
                    dst[key].copy_(src[key][idx])
                continue

            # --- Everything else (bias, norm, rotary) ---
            dst[key].copy_(src[key])

        if "lm_head.weight" in dst and "model.embed_tokens.weight" in dst:
            dst["lm_head.weight"].copy_(dst["model.embed_tokens.weight"])

        return cm, {}
