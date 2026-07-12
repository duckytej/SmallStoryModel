"""
Hyperion-v3: Optimizers
========================
Muon + AdamW split — nanochat confirmed best practice:
  - 2-D weight matrices  →  Muon  (orthogonal gradient updates via Newton-Schulz)
  - embeddings, norms, biases, scalars  →  AdamW
  - All AdamW LRs scaled by sqrt(768 / d_model) for cross-scale consistency.

Change from v2:
  ✓ build_optimizers() deduplicates tied weights by tensor id.
    embed.weight and lm_head.weight are the SAME tensor after weight tying;
    without deduplication the parameter would appear in two optimizer groups,
    causing double gradient updates and incorrect effective LR.
  ✓ BUG FIX (tied-weight LR conflict): the tied embed.weight / lm_head.weight
    tensor used to be routed into embedding_params and updated with the full
    embedding_lr (default 0.2). That rate is calibrated for the INPUT side —
    an aggressive step size to spread out a sparse vocabulary table — but the
    same tensor also serves as the OUTPUT projection straight into the
    softmax, where that step size directly inflates logit variance and
    produces gradient spikes / instability. The tied tensor is now detected
    by identity (id(model.lm_head.weight)) and routed to its own optimizer
    group with a separate, more conservative tied_lm_head_lr, while any other
    genuinely non-tied embedding-like parameters keep the full embedding_lr.

Muon reference: "Muon is Scalable for LLM Training" (arXiv 2502.16982)
  The Newton-Schulz orthogonalisation produces RMS-normalised updates for all
  matrix parameters without per-parameter second-moment estimates (unlike Adam).
  This makes it particularly effective for the large weight matrices in attention
  and FFN layers while keeping memory cost minimal.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (5 iterations, ~1 ms on GPU)
# ---------------------------------------------------------------------------


def _ns5(G: torch.Tensor) -> torch.Tensor:
    """
    Orthogonalise gradient matrix G via 5 Newton-Schulz iterations.
    Operates in-place on a normalised copy; safe on GPU.
    Handles both tall (rows > cols) and wide (cols > rows) matrices.
    """
    assert G.ndim == 2, (
        f"_ns5 expects a 2-D matrix, got shape {tuple(G.shape)}. "
        "Route higher-rank tensors to AdamW via is_degenerate_matrix."
    )
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    if (
        G.shape[-2] < G.shape[-1]
    ):  # wide matrix → transpose, orthogonalise, transpose back
        X = X.T
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[-2] < G.shape[-1]:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Newton-Schulz orthogonal update for 2-D weight matrices.
    Single-GPU version (no distributed Nesterov all-gather).

    Steps per update:
      1. Accumulate Nesterov momentum buffer.
      2. Orthogonalise the update via _ns5.
      3. Scale by sqrt(max(shape)) to preserve RMS magnitude.
      4. Apply the update with learning rate lr.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=mom)
                else:
                    g = buf
                if g.ndim >= 2:
                    g = _ns5(g).mul_(max(g.shape) ** 0.5)
                p.add_(g, alpha=-lr)


# ---------------------------------------------------------------------------
# Factory: Muon + AdamW with tied-weight deduplication
# ---------------------------------------------------------------------------


def build_optimizers(
    model: nn.Module,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.2,
    scalar_lr: float = 0.02,
    gate_lr: float = 0.01,
    tied_lm_head_lr: Optional[float] = None,
    weight_decay: float = 0.0,
    adamw_betas: Tuple[float, float] = (0.9, 0.95),
    d_model: int = 512,
) -> List[torch.optim.Optimizer]:
    """
    Returns [muon_optimizer, adamw_optimizer].

    Routing rules:
      • 2-D matrices that are NOT embeddings or norms  → Muon
      • Embeddings (including tied lm_head.weight)     → AdamW embedding_lr
        EXCEPT: the tensor shared by embed.weight and lm_head.weight (weight
        tying) gets its own, lower tied_lm_head_lr group — see below.
      • Residual-stream gate scalars (resid_scale, x0_scale, smear_scale,
        backout_scale) and degenerate (1, d) gating matrices
        (BlockAttnResOp.proj, smear_gate_scalar)          → AdamW gate_lr
      • Everything else (RMSNorm weights, biases)         → AdamW scalar_lr

    BUG FIX — tied-weight learning rate conflict:
      With weight tying, embed.weight and lm_head.weight are literally the
      SAME tensor in memory, so it can only live in one optimizer group with
      one learning rate. embedding_lr (default 0.2) is calibrated for the
      INPUT side: a large, sparse lookup table that needs aggressive updates
      to spread out a big vocabulary. But that same tensor is also the
      OUTPUT projection straight into the softmax — every update directly
      rescales logit magnitude. Applying embedding_lr's aggressive step size
      there amplifies logit variance turn after turn and produces the loss
      spikes / instability described for tied-weight micro-models. We now
      identify the tied tensor specifically (the one parameter whose name
      contains both an embedding-like and lm_head-like reference, or — more
      robustly — whichever tensor object is reachable as model.lm_head.weight
      if that attribute exists) and route it to tied_lm_head_lr, a separate,
      more conservative rate. Any other, non-tied embedding-style parameters
      (e.g. value_embeds, which are NOT shared with lm_head) keep the full
      embedding_lr they actually need. If tied_lm_head_lr is left as None we
      default it to matrix_lr's scale rather than embedding_lr's, since the
      output-projection role is functionally closer to a regular weight
      matrix than to an input lookup table.

    BUG FIX — residual-stream gate scalars sharing scalar_lr with norms/biases:
      resid_scale and x0_scale multiply the ENTIRE residual stream and are
      each applied TWICE per transformer block (once after attention, once
      after the FFN) — 2 * n_layers compounding multiplications through the
      depth of the network. AdamW's update magnitude is roughly bounded by
      its lr regardless of gradient size, so the old scalar_lr=0.5 (scaled to
      ~0.7 for d_model=384) let a single step move resid_scale from 1.0 to
      well outside [0.5, 1.5] — which then compounds exponentially across the
      2*n_layers applications and produces a sudden, catastrophic activation
      blow-up. This is what produced a loss spike immediately after warmup
      completed and the AdamW LR group first reached full strength: a
      gradient-down-scaling spike guard cannot undo an already-applied bad
      optimizer step, so loss kept climbing instead of recovering.
      smear_scale, backout_scale, BlockAttnResOp.proj (a degenerate (1, d)
      gating matrix used 2*n_layers times for block-summary mixing), and
      smear_gate_scalar (also degenerate) carry the same risk and are now
      routed to a dedicated, conservative gate_lr group. gate_lr is NOT
      multiplied by the width-rescaling factor below — it's an absolute
      architectural safety bound, not a parameter-count-scaled embedding rate.
      scalar_lr's default was also lowered (0.5 → 0.02): even RMSNorm
      weights and biases don't need embedding-table-style step sizes.

    Deduplication:
      Tied weights (embed.weight == lm_head.weight) share the same tensor.
      We track seen tensor ids so the tensor is added to exactly one group.

    LR scaling:
      AdamW LRs (except gate_lr) × sqrt(768 / d_model) keeps effective step
      size consistent when comparing runs with different model widths.
    """
    scale = (768 / d_model) ** 0.5

    if tied_lm_head_lr is None:
        # Default: treat the tied tensor's effective LR as much closer to a
        # regular weight matrix's safe range than to the raw embedding_lr.
        tied_lm_head_lr = max(matrix_lr, embedding_lr * 0.1)

    # Identify the actual tied tensor (if the model exposes the conventional
    # attribute names) so we route by tensor identity, not by name-guessing.
    tied_tensor_id = None
    lm_head_mod = getattr(model, "lm_head", None)
    if lm_head_mod is not None and hasattr(lm_head_mod, "weight"):
        tied_tensor_id = id(lm_head_mod.weight)

    matrix_params: List[nn.Parameter] = []
    embedding_params: List[nn.Parameter] = []
    tied_lm_head_params: List[nn.Parameter] = []
    gate_params: List[nn.Parameter] = []
    scalar_params: List[nn.Parameter] = []

    seen_ids: set = set()

    # Known residual-stream / mixing-gate scalar names (see BUG FIX above).
    # Matched by substring so a future qualified name (e.g. a "blocks.3."
    # prefix) would still match.
    _GATE_NAME_HINTS = ("resid_scale", "x0_scale", "smear_scale", "backout_scale")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Deduplicate tied weights — same tensor must not appear in two groups.
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        is_embed_like = ("embed" in name) or ("lm_head" in name)
        is_norm_like = "norm" in name
        is_scalar_like = (
            "scalar" in name
        )  # e.g. smear_gate_scalar — degenerate row vectors
        # A matrix where either dim == 1 (e.g. BlockAttnResOp.proj: shape (1, d_model))
        # is degenerate for Newton-Schulz orthogonalisation — there is no pair of
        # row/column vectors to orthogonalise against, and the update is semantically
        # wrong (it's closer to a gate than a weight matrix). It is ALSO a
        # residual-stream mixing gate, so it goes to gate_params, not scalar_params.
        is_degenerate_matrix = p.ndim >= 2 and min(p.shape) == 1

        is_tied_lm_head = tied_tensor_id is not None and id(p) == tied_tensor_id

        is_gate_like = is_degenerate_matrix or any(h in name for h in _GATE_NAME_HINTS)

        if (
            p.ndim >= 2
            and not is_embed_like
            and not is_norm_like
            and not is_scalar_like
            and not is_degenerate_matrix
        ):
            matrix_params.append(p)  # → Muon
        elif is_tied_lm_head:
            tied_lm_head_params.append(p)  # → AdamW, conservative tied_lm_head_lr
        elif is_embed_like:
            embedding_params.append(p)  # → AdamW (high LR, no decay)
        elif is_gate_like:
            gate_params.append(
                p
            )  # → AdamW, conservative gate_lr (residual-stream safety)
        else:
            scalar_params.append(p)  # → AdamW (norms, biases)

    muon = Muon(matrix_params, lr=matrix_lr)

    adamw = AdamW(
        [
            {"params": embedding_params, "lr": embedding_lr * scale},
            {"params": tied_lm_head_params, "lr": tied_lm_head_lr * scale},
            {"params": gate_params, "lr": gate_lr},
            {"params": scalar_params, "lr": scalar_lr * scale},
        ],
        betas=adamw_betas,
        weight_decay=weight_decay,
    )

    return [muon, adamw]


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """
    Multiplicative LR scheduler applied uniformly to all optimizer param groups.
    Phase 1 (steps 0 → warmup_steps):    linear ramp from 0 → base_lr
    Phase 2 (steps warmup_steps → total): cosine decay to min_lr_frac × base_lr
    """

    def __init__(
        self,
        optimizers: List[torch.optim.Optimizer],
        warmup_steps: int,
        total_steps: int,
        min_lr_frac: float = 0.1,
    ):
        self.optimizers = optimizers
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_frac = min_lr_frac
        # BUG FIX: guard against misconfigured schedules.
        # If warmup_steps >= total_steps the cosine phase never runs and the
        # LR stays pegged at base_lr (or higher if step > total_steps during
        # the warmup ramp).  Catch this early rather than silently over-training.
        if warmup_steps >= total_steps:
            raise ValueError(
                f"WarmupCosineScheduler: warmup_steps ({warmup_steps}) must be "
                f"strictly less than total_steps ({total_steps})."
            )
        self._step = 0
        # Snapshot base LRs at construction time
        self._base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    def step(self):
        self._step += 1
        s, W, T = self._step, self.warmup_steps, self.total_steps

        if s <= W:
            frac = s / max(W, 1)
        else:
            progress = (s - W) / max(T - W, 1)
            frac = self.min_lr_frac + 0.5 * (1.0 - self.min_lr_frac) * (
                1.0 + math.cos(math.pi * progress)
            )

        for opt, base_lrs in zip(self.optimizers, self._base_lrs):
            for g, base_lr in zip(opt.param_groups, base_lrs):
                g["lr"] = base_lr * frac

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def current_frac(self) -> float:
        s, W, T = self._step, self.warmup_steps, self.total_steps
        if s <= W:
            return s / max(W, 1)
        progress = (s - W) / max(T - W, 1)
        return self.min_lr_frac + 0.5 * (1.0 - self.min_lr_frac) * (
            1.0 + math.cos(math.pi * progress)
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        """Return the scheduler state so it can be saved in a checkpoint."""
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_frac": self.min_lr_frac,
            "base_lrs": self._base_lrs,
        }

    def load_state_dict(self, sd: dict) -> None:
        """
        Restore scheduler state from a checkpoint dict.
        Also updates the live param-group LRs to match the restored step so
        the first optimizer step after resume uses the correct learning rate.
        """
        self._step = sd["step"]
        self.warmup_steps = sd["warmup_steps"]
        self.total_steps = sd["total_steps"]
        self.min_lr_frac = sd["min_lr_frac"]
        self._base_lrs = sd["base_lrs"]
        # Re-apply the LR that corresponds to the restored step
        frac = self.current_frac
        for opt, base_lrs in zip(self.optimizers, self._base_lrs):
            for g, base_lr in zip(opt.param_groups, base_lrs):
                g["lr"] = base_lr * frac
