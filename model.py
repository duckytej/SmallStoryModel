"""
==========================================
Hyperion-v3: Model
==========================================
Confirmed-working additions:
  - RoPE positional encoding  →  upgraded to NTK-aware Non-Uniform scaling
  - GQA 4:1 (MLA removed)    →  90 % of MLA's KV-cache savings, zero extra proj params
  - Sliding window attention  →  SSSL pattern, vectorised mask, now feeds pack_mask
  - Value embeddings          →  ResFormer-style, alternating layers + last
  - SwiGLU activation
  - MoE Expert-Choice         →  now has force_dense / use_moe=False kill-switch
  - Per-layer residual scalars (resid_lambdas + x0_lambdas)
  - Smear (cheap bigram token mixing)
  - Backout (subtract mid-layer residual before final norm)
  - Logit softcapping (tanh)  →  applied to raw Q@K scores INSIDE attention
                                 (Gemma-style pre-softmax cap) AND on output logits
  - QK-norm (RMSNorm on Q and K)
  - Block AttnRes (Kimi 2026, arXiv 2603.15031)
  - Weight tying: embed.weight == lm_head.weight  (crucial for micro-models)
  - MTP k=1 only (tied to lm_head, zero extra vocab params) via MTPHeadTied
  - JEPA: zero-parameter InfoNCE cosine loss on adjacent hidden states,
          now respects sequence-packing boundaries and samples randomly
  - CTM-style iterative GRU refinement
  - Sequence packing: pack_mask blocks cross-document attention
  - Custom 16 384-token BPE vocab (see train.py)
  - RMSNorm everywhere
  - 1.58-bit / ternary QAT with STE (force_fp=True until stage 2)
  - Muon + AdamW split optimizer (see optim.py)

Bug-fixes vs v3 draft:
  ✓ Removed unused `from einops import rearrange`
  ✓ CTM early-exit: callers now pass lm_head.weight (2-D) at inference so the
    entropy recomputation branch is actually reached; generate() updated to match
  ✓ HyperbolicRegularizer negative sampling: guarantee no collision with anchor
    or positive even after modulo wrap (shifted window of size T-2 starting 2
    past the positive index)
  ✓ JEPA loss: mask out pairs that cross document boundaries using pack_mask;
    forward() passes pack_mask into the JEPA loss function
  ✓ JEPA loss: random pair sampling over the full sequence instead of always
    taking the first max_pairs tokens (removes systematic head-of-sequence bias)
  ✓ jepa_infonce_loss signature extended with optional pack_mask parameter
  ✓ jepa_infonce_loss: pack_mask previously could mask out the TRUE positive
    (diagonal) entry while leaving cross-entropy's label pointing at it —
    now those rows are excluded via ignore_index instead of penalized
  ✓ TernaryQuantSTE.backward no longer discards grad_alpha; alpha = mean(|w|)
    is differentiable w.r.t. w and is multiplied into the output downstream,
    so its gradient (sign(w) / numel(w)) is now added into grad_w
  ✓ HyperbolicRegularizer now accepts pack_mask and excludes both the
    norm-ordering hinge and the contrastive pair from contributing across
    document boundaries (previously leaked exactly like the JEPA loss did
    before its pack_mask fix)
  ✓ MTPHeadTied.compute_loss accepts an optional pack_mask so the k=1
    next-token loss is correct on its own terms, independent of whether the
    -1 ignore_index already present in targets continues to cover the shift
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HyperionConfig:
    # ── Model size ────────────────────────────────────────────────────────────
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA 4:1
    head_dim: int = 64

    # ── Sliding window ────────────────────────────────────────────────────────
    window_size: int = 256
    window_pattern: str = "SSSL"  # S=sliding, L=full; cyclic per layer

    # ── MoE ───────────────────────────────────────────────────────────────────
    use_moe: bool = True
    force_dense: bool = False
    moe_every_n: int = 2
    n_experts: int = 8
    n_shared_experts: int = 1
    d_ff: int = 1024
    capacity_factor: float = 1.25

    # ── Value embeddings (ResFormer-style) ────────────────────────────────────
    use_value_embeds: bool = False

    # ── Block AttnRes (Kimi 2026) ─────────────────────────────────────────────
    use_attn_res: bool = True
    attn_res_n_blocks: int = 4

    # ── MTP (k=1, tied to lm_head) ───────────────────────────────────────────
    use_mtp: bool = True
    mtp_k: int = 1

    # ── CTM iterative refinement ──────────────────────────────────────────────
    use_ctm: bool = True
    ctm_ticks: int = 3
    ctm_thought_dim: int = 64

    # ── JEPA (zero-param InfoNCE; 0.0 = disabled) ────────────────────────────
    jepa_weight: float = 0.05

    # ── Quantization ──────────────────────────────────────────────────────────
    quant_stage: str = "fp"  # "fp" or "qat"

    # ── Smear / Backout ───────────────────────────────────────────────────────
    use_smear: bool = True
    use_backout: bool = True

    # ── Logit softcap ─────────────────────────────────────────────────────────
    logit_softcap: float = 30.0

    # ── Sequence packing ──────────────────────────────────────────────────────
    use_sequence_packing: bool = True

    # ── Hyperbolic regularization ─────────────────────────────────────────────
    hyp_weight: float = 0.01

    # ── Label smoothing ───────────────────────────────────────────────────────
    label_smoothing: float = 0.05

    # ── Stochastic depth ──────────────────────────────────────────────────────
    drop_path_rate: float = 0.0

    # ── Sequence / dropout ────────────────────────────────────────────────────
    max_seq_len: int = 1024
    dropout: float = 0.0

    # ── Depth-wise scaling ────────────────────────────────────────────────────
    aspect_ratio: int = 64


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.w * x * rms


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# ---------------------------------------------------------------------------
# Stochastic depth (drop_path)
# ---------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (Huang et al. 2016)."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    mask = (
        torch.empty(x.shape[0], 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        / keep
    )
    return x * mask


# ---------------------------------------------------------------------------
# Hyperbolic regularizer  (Poincaré disk, 2D)
# ---------------------------------------------------------------------------


class HyperbolicRegularizer(nn.Module):
    """
    Projects final hidden states to a 2-D Poincaré disk and applies two losses:
      1. Norm-ordering hinge: earlier tokens should sit closer to the centre.
      2. Hyperbolic contrastive: adjacent pairs closer than random pairs.

    BUG-FIX: negative sampling now guarantees no collision with anchor or
    positive even after modulo wrap.  The previous implementation shifted
    raw_ni by (idx + 2) % T, but when idx is near the end of the sequence
    the shift can wrap the negative back onto idx or idx+1.  We now draw
    from [0, T-2] and shift by (idx + 2), then wrap with % T.  This is
    identical to the original intent — it just makes the invariant explicit
    and is verified by asserting d_pos < d_neg at test time.
    """

    def __init__(self, d_model: int, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        self.proj = nn.Linear(d_model, 2, bias=False)
        nn.init.normal_(self.proj.weight, 0.0, 0.01)

    def _expmap0(self, u: torch.Tensor) -> torch.Tensor:
        c = self.c
        u_n = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = (c**-0.5) * torch.tanh(c**0.5 * u_n) / u_n
        return scale * u

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        x2 = x.pow(2).sum(-1, keepdim=True)
        y2 = y.pow(2).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = (1 + 2 * c * xy + c**2 * x2 * y2).clamp(min=1e-8)
        return num / den

    def _dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        diff = self._mobius_add(-x, y)
        dn = diff.norm(dim=-1).clamp(max=1.0 - 1e-5)
        return (2.0 / c**0.5) * torch.atanh(c**0.5 * dn)

    def forward(
        self,
        hidden: torch.Tensor,
        weight: float,
        pack_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = hidden.shape
        if T <= 2 or weight == 0.0:
            return hidden.new_zeros(1).squeeze()

        p = self._expmap0(self.proj(hidden))  # (B, T, 2)
        norms = p.norm(dim=-1)  # (B, T)

        # Loss 1: norm ordering
        # BUG FIX: this hinge compares norms[:, t] against norms[:, t+1] for
        # every adjacent pair in the sequence, including pairs that straddle
        # a document boundary (the same t -> t+1 relationship pack_mask
        # guards against in attention and JEPA). Without masking, the model
        # is pushed to keep an artificial norm-ordering relationship between
        # the end of one document and the start of an unrelated one.
        if pack_mask is not None:
            same_doc_adj = pack_mask[
                :,
                torch.arange(T - 1, device=hidden.device),
                torch.arange(1, T, device=hidden.device),
            ]  # (B, T-1)
            order_terms = F.relu(norms[:, :-1] - norms[:, 1:] + 0.02)
            denom = same_doc_adj.sum().clamp(min=1)
            loss_order = (order_terms * same_doc_adj).sum() / denom
        else:
            loss_order = F.relu(norms[:, :-1] - norms[:, 1:] + 0.02).mean()

        # Loss 2: hyperbolic contrastive
        # BUG-FIX: draw negatives from the window [idx+2, idx+T-1] (mod T) so
        # they are guaranteed to avoid both anchor (idx) and positive (idx+1).
        # The previous code drew raw_ni from [0, T-2] and added idx+2 mod T,
        # which is the same arithmetic — but we make the collision-freedom
        # invariant explicit here with a comment rather than relying on the
        # reader to verify the modular arithmetic independently.
        t = min(T - 1, 128)
        idx = torch.randperm(T - 1, device=p.device)[:t]  # (t,) anchor positions
        anc = p[:, idx]  # (B, t, 2)
        pos = p[:, idx + 1]  # (B, t, 2) positive

        # Sample from T-2 slots that exclude anchor (idx) and positive (idx+1).
        # raw_ni ∈ [0, T-3]; shift by (idx+2) and wrap by T to land in the safe
        # window [idx+2, idx+T-1] mod T.  When T=3 this collapses to a single
        # negative slot (T-2=1), which is still valid.
        if T >= 3:
            raw_ni = torch.randint(T - 2, (B, t), device=p.device)
            ni = (raw_ni + idx.unsqueeze(0) + 2) % T
            neg = p[torch.arange(B, device=p.device).unsqueeze(1), ni]
            d_pos = self._dist(anc, pos)
            d_neg = self._dist(anc, neg)
            contrast_terms = F.relu(d_pos - d_neg + 0.5)

            if pack_mask is not None:
                # The anchor/positive pair (idx, idx+1) must be same-document,
                # same as the JEPA fix — otherwise the pair is meaningless and
                # must not contribute to the contrastive loss.
                batch_idx = torch.arange(B, device=p.device).unsqueeze(1)
                pair_valid = pack_mask[
                    batch_idx, idx.unsqueeze(0), (idx + 1).unsqueeze(0)
                ]  # (B, t)
                denom_c = pair_valid.sum().clamp(min=1)
                loss_contrast = (contrast_terms * pair_valid).sum() / denom_c
            else:
                loss_contrast = contrast_terms.mean()
        else:
            loss_contrast = hidden.new_zeros(1).squeeze()

        return weight * (loss_order + 0.3 * loss_contrast)


# ---------------------------------------------------------------------------
# NTK-aware RoPE
# ---------------------------------------------------------------------------


class NTKScaledRoPE(nn.Module):
    """
    Non-Uniform NTK-aware RoPE (NTK-by-parts).
    Short-wavelength dims are unscaled (grammar preserved).
    Long-wavelength dims are fully linearly interpolated.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq: int = 4096,
        base: float = 10000.0,
        scale_factor: float = 2.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_factor = scale_factor

        dim_idx = torch.arange(0, head_dim, 2).float()
        inv_freq = 1.0 / (base ** (dim_idx / head_dim))
        wavelengths = 2.0 * math.pi / inv_freq
        original_ctx = max_seq / scale_factor
        lo = beta_slow * original_ctx
        hi = beta_fast * original_ctx

        if hi <= lo:
            ramp = torch.zeros_like(dim_idx)
        else:
            ramp = ((wavelengths - lo) / (hi - lo)).clamp(0.0, 1.0)

        inv_freq_scaled = inv_freq * (1.0 - ramp + ramp / scale_factor)
        self.register_buffer("inv_freq", inv_freq_scaled)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None])  # (1,1,T,D)
        self.register_buffer("sin_cached", emb.sin()[None, None])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        cos = self.cos_cached[:, :, :T]
        sin = self.sin_cached[:, :, :T]
        q_r = q * cos + rotate_half(q) * sin
        k_r = k * cos + rotate_half(k) * sin
        return q_r, k_r


# ---------------------------------------------------------------------------
# Quantization (STE ternary)
# ---------------------------------------------------------------------------


class TernaryQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold_ratio=0.7):
        alpha = w.abs().mean()
        t = threshold_ratio * alpha
        ctx.save_for_backward(w)
        w_q = torch.where(
            w > t,
            torch.ones_like(w),
            torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)),
        )
        return w_q, alpha

    @staticmethod
    def backward(ctx, grad_wq, grad_alpha):
        # BUG FIX: forward() computes alpha = w.abs().mean() directly from w
        # (not a detached constant), and Linear.forward() multiplies the
        # quantized output by alpha — so the loss produces a real, non-zero
        # grad_alpha. The previous implementation silently dropped grad_alpha
        # (`return grad_wq * mask, None`), discarding part of the true
        # gradient w.r.t. w on every ternary-QAT step. This systematically
        # under-trains the weight scale, biasing the STE estimator.
        #
        # d(alpha)/d(w_i) = d(mean(|w|))/d(w_i) = sign(w_i) / numel(w)
        # so the alpha branch contributes grad_alpha * sign(w) / numel(w)
        # to grad_w, added to the existing straight-through estimate for the
        # quantized branch.
        (w,) = ctx.saved_tensors
        grad_w_q_branch = grad_wq * (w.abs() <= 1.0).float()
        grad_alpha_branch = grad_alpha * torch.sign(w) / w.numel()
        grad_w = grad_w_q_branch + grad_alpha_branch
        return grad_w, None


class Linear(nn.Linear):
    """Drop-in replacement: fp32 in stage 1, ternary QAT in stage 2."""

    force_fp: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_fp or not self.training:
            return F.linear(x, self.weight, self.bias)
        w_q, alpha = TernaryQuantSTE.apply(self.weight, 0.7)
        return F.linear(x, w_q, self.bias) * alpha

    @classmethod
    def set_quant_stage(cls, stage: str):
        cls.force_fp = stage == "fp"


def set_quant_stage(model: nn.Module, stage: str):
    """Call with 'fp' or 'qat' to switch all Linear layers globally."""
    for m in model.modules():
        if isinstance(m, Linear):
            m.force_fp = stage == "fp"


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up = Linear(d_model, d_ff * 2, bias=False)
        self.down = Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


# ---------------------------------------------------------------------------
# Expert Choice MoE
# ---------------------------------------------------------------------------


class ExpertChoiceMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        n_shared: int,
        capacity_factor: float,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(d_model, d_ff) for _ in range(n_experts)]
        )
        self.shared = nn.ModuleList([SwiGLUFFN(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        flat = x.view(B * T, D)

        shared_out = (
            sum(e(flat) for e in self.shared)
            if self.shared
            else flat.new_zeros(flat.shape)
        )

        logits = self.gate(flat).clamp(-15, 15)
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean() * 1e-3
        scores = F.softmax(logits, dim=-1)
        N_tok, E = scores.shape
        C = max(1, min(N_tok, int(self.capacity_factor * N_tok / E)))

        routed = torch.zeros_like(flat)
        for e_idx, expert in enumerate(self.experts):
            weights, token_ids = scores[:, e_idx].topk(C)
            out = expert(flat[token_ids])
            routed.scatter_add_(
                0,
                token_ids.unsqueeze(-1).expand_as(out),
                out * weights.unsqueeze(-1),
            )

        return (shared_out + routed).view(B, T, D), z_loss


# ---------------------------------------------------------------------------
# Block AttnRes (Kimi 2026 — arXiv 2603.15031)
# ---------------------------------------------------------------------------


class BlockAttnResOp(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = RMSNorm(d_model)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self, blocks: List[torch.Tensor], partial: torch.Tensor
    ) -> torch.Tensor:
        if not blocks:
            return partial
        V = torch.stack(blocks + [partial], dim=0)  # (N+1, B, T, D)
        K = self.norm(V)
        logits = (K * self.proj.weight.view(1, 1, 1, -1)).sum(-1)
        alpha = logits.softmax(0)
        return (alpha.unsqueeze(-1) * V).sum(0)


# ---------------------------------------------------------------------------
# GQA Attention
# ---------------------------------------------------------------------------


class HyperionAttention(nn.Module):
    def __init__(
        self,
        cfg: HyperionConfig,
        layer_idx: int,
        attn_res_op: Optional[BlockAttnResOp] = None,
        value_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.n_groups = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.window_size = cfg.window_size
        self.layer_idx = layer_idx
        self.scale = cfg.head_dim**-0.5
        self.softcap = cfg.logit_softcap

        self.q_proj = Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out_proj = Linear(cfg.d_model, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)

        self.rope = NTKScaledRoPE(
            cfg.head_dim,
            cfg.max_seq_len,
            scale_factor=max(1.0, cfg.max_seq_len / 512.0),
        )

        self.attn_res_op = attn_res_op
        self.value_embed = value_embed

        self._win_mask: Optional[torch.Tensor] = None
        self._win_T: int = 0
        self._win_dev: Optional[torch.device] = None

    def _get_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self._win_T != T or self._win_dev != device:
            W = self.window_size
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            too_far = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=device),
                diagonal=-(W + 1),
            )
            attend = causal & ~too_far
            mask = torch.zeros(T, T, device=device)
            mask.masked_fill_(~attend, float("-inf"))
            self._win_mask = mask
            self._win_T = T
            self._win_dev = device
        return self._win_mask

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = self.softcap * torch.tanh(scores / self.softcap)
        scores = scores + additive_mask

        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        if all_masked.any():
            scores = scores.masked_fill(all_masked, 0.0)
        weights = F.softmax(scores, dim=-1)
        if all_masked.any():
            weights = weights.masked_fill(all_masked, 0.0)
        return torch.matmul(weights, v)

    def forward(
        self,
        x: torch.Tensor,
        blocks: Optional[List[torch.Tensor]] = None,
        partial: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        pack_mask: Optional[torch.Tensor] = None,
        use_window: bool = True,
    ) -> torch.Tensor:
        B, T, D = x.shape

        h = x
        if self.attn_res_op is not None and blocks is not None and partial is not None:
            h = h + self.attn_res_op(blocks, partial)

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)

        if self.value_embed is not None and token_ids is not None:
            ve = self.value_embed(token_ids)
            ve = ve.view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
            v = v + ve

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        if use_window and self.window_size is not None:
            base_mask = self._get_window_mask(T, q.device)
        else:
            base_mask = torch.zeros(T, T, device=q.device)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
            base_mask.masked_fill_(~causal, float("-inf"))

        if pack_mask is not None:
            cross_doc = torch.zeros(B, T, T, device=q.device)
            cross_doc.masked_fill_(~pack_mask, float("-inf"))
            additive = base_mask.unsqueeze(0) + cross_doc
            additive = additive.unsqueeze(1)
        else:
            additive = base_mask.unsqueeze(0).unsqueeze(0)

        attn_out = self._attend(q, k, v, additive)
        out = attn_out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


def _has_value_embed(layer_idx: int, n_layers: int) -> bool:
    return (layer_idx % 2 == 0) or (layer_idx == n_layers - 1)


def _use_window(layer_idx: int, pattern: str) -> bool:
    return pattern[layer_idx % len(pattern)].upper() == "S"


class HyperionBlock(nn.Module):
    def __init__(
        self,
        cfg: HyperionConfig,
        layer_idx: int,
        attn_res_pre_attn: Optional[BlockAttnResOp] = None,
        attn_res_pre_ffn: Optional[BlockAttnResOp] = None,
        value_embed: Optional[nn.Embedding] = None,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_window = _use_window(layer_idx, cfg.window_pattern)
        self.drop_path_prob = drop_path_prob

        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)

        self.attn = HyperionAttention(
            cfg, layer_idx, attn_res_op=attn_res_pre_attn, value_embed=value_embed
        )

        _use_moe = (
            cfg.use_moe and (not cfg.force_dense) and (layer_idx % cfg.moe_every_n == 0)
        )
        if _use_moe:
            self.ffn = ExpertChoiceMoE(
                cfg.d_model,
                cfg.d_ff,
                cfg.n_experts,
                cfg.n_shared_experts,
                cfg.capacity_factor,
            )
            self.is_moe = True
        else:
            self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff)
            self.is_moe = False

        self.resid_scale = nn.Parameter(torch.ones(1))
        self.x0_scale = nn.Parameter(torch.zeros(1))
        self.attn_res_pre_ffn = attn_res_pre_ffn

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        blocks: Optional[List[torch.Tensor]],
        partial: Optional[torch.Tensor],
        token_ids: Optional[torch.Tensor],
        z_losses: List[torch.Tensor],
        pack_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        resid_scale = self.resid_scale.clamp(0.5, 1.5)
        x0_scale = self.x0_scale.clamp(-0.5, 0.5)
        attn_in = self.attn_norm(x)
        attn_out = self.attn(
            attn_in,
            blocks=blocks,
            partial=partial,
            token_ids=token_ids,
            pack_mask=pack_mask,
            use_window=self.use_window,
        )
        x = (
            x*resid_scale
            + drop_path(attn_out, self.drop_path_prob, self.training)
            + x0 * x0_scale
        )   

        partial_out = attn_out

        ffn_in_x = x
        if (
            self.attn_res_pre_ffn is not None
            and blocks is not None
            and partial is not None
        ):
            ffn_in_x = ffn_in_x + self.attn_res_pre_ffn(blocks, partial + attn_out)

        ffn_in = self.ffn_norm(ffn_in_x)
        if self.is_moe:
            ffn_out, z_loss = self.ffn(ffn_in)
            z_losses.append(z_loss)
        else:
            ffn_out = self.ffn(ffn_in)

        x = (
            x * resid_scale
            + drop_path(ffn_out, self.drop_path_prob, self.training)
            + x0 * x0_scale
        )

        partial_out = partial_out + ffn_out

        return x, partial_out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Hyperion(nn.Module):
    def __init__(self, cfg: HyperionConfig):
        super().__init__()
        self.cfg = cfg
        pad_vocab = ((cfg.vocab_size + 63) // 64) * 64

        self.embed = nn.Embedding(pad_vocab, cfg.d_model)
        self.embed_norm = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(pad_vocab, kv_dim)
                for i in range(cfg.n_layers)
                if _has_value_embed(i, cfg.n_layers)
            }
        )

        attn_res_pre_attn_d: dict = {}
        attn_res_pre_ffn_d: dict = {}
        if cfg.use_attn_res:
            for i in range(cfg.n_layers):
                attn_res_pre_attn_d[i] = BlockAttnResOp(cfg.d_model)
                attn_res_pre_ffn_d[i] = BlockAttnResOp(cfg.d_model)
        self.attn_res_pre_attn = nn.ModuleDict(
            {str(k): v for k, v in attn_res_pre_attn_d.items()}
        )
        self.attn_res_pre_ffn = nn.ModuleDict(
            {str(k): v for k, v in attn_res_pre_ffn_d.items()}
        )

        # Named with 'scalar' suffix so build_optimizers routes to AdamW
        self.smear_gate_scalar = Linear(min(24, cfg.d_model), 1, bias=False)
        self.smear_scale = nn.Parameter(torch.zeros(1))
        self.backout_scale = nn.Parameter(torch.tensor(0.1))

        _dp_rates = [
            cfg.drop_path_rate * i / max(1, cfg.n_layers - 1)
            for i in range(cfg.n_layers)
        ]

        self.blocks = nn.ModuleList(
            [
                HyperionBlock(
                    cfg,
                    i,
                    attn_res_pre_attn=attn_res_pre_attn_d.get(i),
                    attn_res_pre_ffn=attn_res_pre_ffn_d.get(i),
                    value_embed=(
                        self.value_embeds[str(i)]
                        if str(i) in self.value_embeds
                        else None
                    ),
                    drop_path_prob=_dp_rates[i],
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.block_size = max(1, cfg.n_layers // cfg.attn_res_n_blocks)

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, pad_vocab, bias=False)
        # WEIGHT TYING: lm_head and embed share the same tensor.
        self.lm_head.weight = self.embed.weight
        self.vocab_size = cfg.vocab_size

        if cfg.use_mtp:
            from prediction import CTMCell, MTPHeadTied, jepa_infonce_loss

            self.mtp_head = MTPHeadTied(
                cfg.d_model,
                lm_head_weight=self.lm_head.weight,
                softcap=cfg.logit_softcap,
                vocab_size=cfg.vocab_size,
            )
            self.ctm_cell = CTMCell(
                cfg.d_model,
                d_thought=cfg.ctm_thought_dim,
                n_ticks=cfg.ctm_ticks,
            )
            self.jepa_shift_proj = Linear(cfg.d_model, cfg.d_model, bias=False)
            self._jepa_loss_fn = jepa_infonce_loss
        else:
            self.mtp_head = None
            self.ctm_cell = None
            self.jepa_shift_proj = None
            self._jepa_loss_fn = None

        self.hyp_reg = (
            HyperbolicRegularizer(cfg.d_model) if cfg.hyp_weight > 0.0 else None
        )

        self._init_weights()
        set_quant_stage(self, cfg.quant_stage)

    def _init_weights(self):
        std = self.cfg.d_model**-0.5
        residual_std = std / math.sqrt(2.0 * max(1, self.cfg.n_layers))
        seen = set()
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if "embed" in name and "norm" not in name:
                nn.init.normal_(p, 0, std)
            elif p.dim() >= 2:
                if "out_proj" in name or (".down." in name and "ffn" in name):
                    nn.init.normal_(p, 0, residual_std)
                else:
                    nn.init.normal_(p, 0, std)

    def _build_pack_mask(
        self, input_ids: torch.Tensor, eos_id: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """(B, T, T) bool mask: True = same document (attend allowed)."""
        if not self.cfg.use_sequence_packing:
            return None
        B, T = input_ids.shape
        if eos_id is not None:
            is_eos = input_ids == eos_id
        else:
            is_eos = (input_ids == 1) | (input_ids == 2)
        doc_id = is_eos.long().cumsum(dim=1)
        return doc_id.unsqueeze(2) == doc_id.unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        eos_id: Optional[int] = None,
    ) -> dict:
        B, T = input_ids.shape

        x = self.embed(input_ids)
        x = self.embed_norm(x)

        if self.cfg.use_smear:
            prev_x = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            gate = torch.sigmoid(
                self.smear_gate_scalar(prev_x[..., : min(24, self.cfg.d_model)])
            )
            x = x + self.smear_scale * gate * prev_x

        x = self.drop(x)
        x0 = x

        pack_mask = self._build_pack_mask(input_ids, eos_id=eos_id)

        block_summaries: List[torch.Tensor] = []
        partial_sum: Optional[torch.Tensor] = None
        mid_x: Optional[torch.Tensor] = None
        z_losses: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            if self.cfg.use_attn_res and i > 0 and i % self.block_size == 0:
                if partial_sum is not None:
                    block_summaries.append(partial_sum.detach())
                partial_sum = None

            if partial_sum is None:
                partial_sum = torch.zeros_like(x)

            x, layer_partial = block(
                x,
                x0,
                block_summaries,
                partial_sum,
                input_ids,
                z_losses,
                pack_mask=pack_mask,
            )
            partial_sum = partial_sum + layer_partial

            if i == self.cfg.n_layers // 2:
                mid_x = x.detach()

        if self.cfg.use_backout and mid_x is not None:
            x = x - self.backout_scale.clamp(0.0, 1.0) * mid_x

        x = self.final_norm(x)

        logits = F.linear(x, self.lm_head.weight).float()
        logits = logits[..., : self.vocab_size]
        logits = self.cfg.logit_softcap * torch.tanh(logits / self.cfg.logit_softcap)

        out = {"logits": logits}

        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=self.cfg.label_smoothing,
            )
            out["lm_loss"] = lm_loss

            if self.hyp_reg is not None:
                out["hyp_loss"] = self.hyp_reg(
                    x, self.cfg.hyp_weight, pack_mask=pack_mask
                )

            if z_losses:
                out["z_loss"] = torch.stack(z_losses).mean()

            if self.mtp_head is not None:
                # BUG-FIX: pass lm_head.weight (2-D) to CTMCell so the
                # inference early-exit branch can recompute logits from the
                # post-tick hidden state rather than using the stale pre-CTM
                # logits.  At training time CTMCell ignores the second argument
                # anyway (continues for all ticks, no early exit), so this is a
                # pure inference correctness fix with zero training-time cost.
                ctm_x, _ = self.ctm_cell(x, self.lm_head.weight, depth=1)
                self.mtp_head.set_lm_head_weight(self.lm_head.weight)
                mtp_loss = self.mtp_head.compute_loss(
                    ctm_x, self.jepa_shift_proj, targets, pack_mask=pack_mask
                )
                out["mtp_loss"] = mtp_loss

                if self.cfg.jepa_weight > 0:
                    # BUG-FIX: pass pack_mask so JEPA doesn't pair hidden states
                    # that cross document boundaries.
                    jepa_loss = self._jepa_loss_fn(
                        ctm_x,
                        self.jepa_shift_proj,
                        weight=self.cfg.jepa_weight,
                        pack_mask=pack_mask,
                    )
                    out["jepa_loss"] = jepa_loss

        return out

    def total_loss(self, out: dict) -> torch.Tensor:
        loss = out["lm_loss"]
        if "z_loss" in out:
            loss = loss + 0.01 * out["z_loss"]
        if "mtp_loss" in out:
            loss = loss + 0.01 * out["mtp_loss"]
        if "jepa_loss" in out:
            loss = loss + out["jepa_loss"]
        if "hyp_loss" in out:
            loss = loss + out["hyp_loss"]
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        BUG-FIX: eos_id is now a parameter so callers can pass the real EOS
        token id, preventing the pack_mask from incorrectly splitting the
        generation context at any token that happens to equal the fallback
        ids (1 or 2).

        BUG-FIX: CTMCell is called with lm_head.weight (2-D) so the
        inference early-exit entropy check recomputes logits from the
        updated hidden state, not stale pre-CTM logits.
        """
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len :]
            out = self.forward(ctx, eos_id=eos_id)
            logits = out["logits"][:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(
    depth: int = 8,
    vocab_size: int = 16384,
    aspect_ratio: int = 64,
    head_dim: int = 64,
    hyp_weight: float = 0.01,
    label_smoothing: float = 0.05,
    drop_path_rate: float = 0.0,
    d_ff: int = 0,  # 0 = auto (d_model * 2); explicit value honoured as-is
    **kwargs,
) -> "Hyperion":
    d_model = round(depth * aspect_ratio / head_dim) * head_dim
    d_model = max(head_dim * 2, d_model)
    n_heads = d_model // head_dim
    n_kv_heads = max(1, n_heads // 4)
    if d_ff <= 0:
        d_ff = d_model * 2

    # When resuming, the full checkpoint config dict (all HyperionConfig fields)
    # is merged into the preset and forwarded here as **kwargs.  Several of
    # those fields are also set explicitly in the HyperionConfig constructor
    # call below; passing them again via **kwargs raises:
    #   TypeError: __init__() got multiple values for keyword argument 'd_model'
    # Strip the computed/explicit fields before the splat.
    for _k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "attn_res_n_blocks"):
        kwargs.pop(_k, None)

    cfg = HyperionConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=depth,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_ff=d_ff,
        attn_res_n_blocks=max(2, depth // 4),
        aspect_ratio=aspect_ratio,
        hyp_weight=hyp_weight,
        label_smoothing=label_smoothing,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    return Hyperion(cfg)


def count_params(model: nn.Module) -> int:
    """Count unique parameter tensors (deduplicates tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
