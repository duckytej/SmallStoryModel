# Hyperion-v3: Prediction heads
# ==============================

# Three components, each carefully budget-constrained:

# 1. CTMCell  (unchanged from v2)
#    GRU-based iterative refinement with confidence-based early exit.
#    Parameters: 3 × (d_model × d_thought) + d_thought² + d_thought
#    These are NOT shared with the main model — the GRU is a tiny post-hoc
#    polisher on the final hidden states, not a core transformer layer.

# 2. MTPHeadTied  (replaces MTPHead)
#    Predicts 1 token ahead using the tied lm_head.weight — ZERO new vocab params.

#    Old MTPHead cost:
#      draft_transformer  ≈ 4 × d_head_inner²           (attention + FFN)
#      k independent heads ≈ k × d_head_inner × vocab_size
#      For k=4, d_head_inner=256, vocab=32000: ~33 M params → theft from core model

#    New MTPHeadTied cost:
#      shift_norm         = d_model scalars               (RMSNorm, negligible)
#      vocab projection   = ZERO — reuses lm_head.weight
#      shift_proj         = d_model × d_model (shared with JEPA, in Hyperion)
#      For d_model=384: 147 K params total (shift_proj is shared, not doubled)

#    How it works:
#      hidden → shift_proj (relational shift: "what comes next?") →
#      RMSNorm → lm_head.weight^T  →  logits →  softcap  →  CE loss

# 3. jepa_infonce_loss  (replaces JEPALightPredictor)
#    Zero-parameter InfoNCE / NT-Xent contrastive loss.

#    Old JEPALightPredictor cost:
#      TransformerEncoder + 3 ahead_projs ≈ 4 × d_latent²  (e.g. 64² × 5 ≈ 20 K)
#      Plus jepa_proj Linear(d_model → d_latent): d_model × d_latent more

#    New jepa_infonce_loss cost:
#      ZERO new parameters.
#      Reuses the shift_proj already registered in Hyperion as jepa_shift_proj.
#      anchor   = L2_norm( shift_proj( hidden[:, :-1] ) )
#      positive = L2_norm( hidden[:, 1:] ).detach()          ← stop-gradient
#      loss     = InfoNCE( anchor, positive, temperature=0.1 )
#      This forces the model to encode "what token comes next" in its
#      current hidden representation — exactly the JEPA objective — with
#      zero extra parameters beyond the shared shift projection.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Shared primitive
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.w * x * rms


# ---------------------------------------------------------------------------
# CTMCell — iterative GRU refinement  (unchanged from v2)
# ---------------------------------------------------------------------------

class CTMCell(nn.Module):
    """
    Runs up to n_ticks GRU steps on the final hidden states.
    Applies a small delta (scaled 0.1) per tick to avoid destabilising training.
    Early-exit during inference when token entropy is low AND gate confidence high.
    depth==0 short-circuits immediately (no-op path for ablation).
    """
    def __init__(self, d_model: int, d_thought: int = 64, n_ticks: int = 3):
        super().__init__()
        self.d_thought = d_thought
        self.n_ticks   = n_ticks
        self.to_thought   = nn.Linear(d_model,   d_thought)
        self.gru          = nn.GRUCell(d_thought, d_thought)
        self.from_thought = nn.Linear(d_thought, d_model)
        self.exit_gate    = nn.Linear(d_thought, 1)
        self.norm         = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, vocab_logits: torch.Tensor,
                depth: int = 1,
                max_ticks: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        if depth == 0:
            return x, 0

        B, T, D  = x.shape
        max_ticks = max_ticks or self.n_ticks
        x_flat    = x.reshape(B * T, D)
        thought   = torch.zeros(B * T, self.d_thought, device=x.device, dtype=x.dtype)
        ticks_used = 0

        # At inference, if the caller passes lm_head.weight (2-D) we recompute
        # logits from the updated hidden state so the entropy check reflects the
        # post-tick distribution rather than the stale pre-CTM logits.
        _lm_weight: Optional[torch.Tensor] = None
        if not self.training and vocab_logits.ndim == 2:
            _lm_weight = vocab_logits

        for _ in range(max_ticks):
            inp     = self.to_thought(self.norm(x_flat))
            thought = self.gru(inp, thought)
            delta   = self.from_thought(thought).reshape(B, T, D)
            x       = x + delta * 0.1
            x_flat  = x.reshape(B * T, D)
            ticks_used += 1

            if self.training:
                continue  # no early exit during training

            with torch.no_grad():
                cur_logits = x @ _lm_weight.T if _lm_weight is not None else vocab_logits
                probs   = F.softmax(cur_logits.float(), dim=-1)
                entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean()
                conf    = self.exit_gate(thought).sigmoid().mean()
                if entropy < 1.0 and conf > 0.8:
                    break

        return x, ticks_used


# ---------------------------------------------------------------------------
# MTPHeadTied — 1-step look-ahead, vocabulary projection tied to lm_head
# ---------------------------------------------------------------------------

class MTPHeadTied(nn.Module):
    """
    Parameter-efficient 1-step Multi-Token Prediction head.

    The key insight: predicting token T+1 from hidden[T] is exactly what the
    main lm_head.weight already encodes. We do NOT need a new output matrix.
    We only need a lightweight "shift" that transforms the current hidden state
    into a "next-step query" before projecting through the shared vocabulary.

    The shift_proj (d_model → d_model) is passed in from Hyperion at call time
    so that it can be shared with the JEPA loss — both need to answer the same
    question: "what does the next hidden state look like?"

    Flow:
      hidden  →  shift_proj  →  shift_norm  →  lm_head.weight^T  →  softcap  →  CE
    """
    def __init__(self, d_model: int,
                 lm_head_weight: torch.Tensor,
                 softcap: float = 30.0,
                 vocab_size: Optional[int] = None):
        super().__init__()
        self.softcap    = softcap
        self.shift_norm = RMSNorm(d_model)
        # True vocab size for logit slicing. lm_head.weight has shape
        # (pad_vocab, d_model) where pad_vocab is rounded up to a multiple of
        # 64; we must slice to the real vocabulary before computing CE loss.
        # Falls back to lm_head.weight.shape[0] (pad_vocab) if not provided,
        # which is safe but wastes a few softmax buckets on padding tokens.
        self._vocab_size: int = vocab_size if vocab_size is not None else lm_head_weight.shape[0]
        # Store a reference to the tied weight — NOT as a Parameter so it
        # isn't double-counted by count_params().
        object.__setattr__(self, '_lm_w', lm_head_weight)

    # Allow Hyperion to update the reference if the model is compiled / re-wrapped
    def set_lm_head_weight(self, w: torch.Tensor):
        object.__setattr__(self, '_lm_w', w)

    def compute_loss(self, hidden: torch.Tensor,
                     shift_proj: nn.Module,
                     targets:    torch.Tensor,
                     pack_mask:  Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden     : (B, T, D)  post-CTM hidden states
        shift_proj : Linear(D→D) shared with JEPA, registered in Hyperion
        targets    : (B, T)     token ids  (predict targets[:, 1:] from hidden[:, :-1])
        pack_mask  : (B, T, T)  bool, True = same document (optional)

        BUG FIX: pack_mask parameter added to signature (was referenced in body
        but missing from the parameter list, causing NameError at runtime).

        BUG FIX: MTP predicts hidden[:, t] → target token at position t+1.
        The valid transition check must confirm that positions t and t+1 belong
        to the same document, i.e. pack_mask[:, t, t+1]. The previous
        implementation incorrectly used idx_target = arange(2, T+1) (offset by
        2) which corresponds to a 2-step look-ahead relationship, not the actual
        1-step relationship being predicted, and then clamped/zeroed the last
        entry inconsistently.
        """
        B, T, D = hidden.shape
        if T <= 1:
            return hidden.new_zeros(1).squeeze()

        # Apply relational shift, then project through tied lm_head weight.
        # Slice to true vocab_size (lm_head.weight rows = pad_vocab >= vocab_size).
        V       = self._vocab_size
        shifted = self.shift_norm(shift_proj(hidden[:, :-1]))     # (B, T-1, D)
        logits  = F.linear(shifted, self._lm_w).float()[..., :V]  # (B, T-1, V)
        logits  = self.softcap * torch.tanh(logits / self.softcap)

        # targets[:, 1:] — token at position t+1 is what hidden[:, t] predicts
        target_ids = targets[:, 1:].reshape(-1)                   # (B*(T-1),)

        if pack_mask is not None:
            # MTP predicts hidden[:, t] → target[:, t+1].
            # Valid only when positions t and t+1 are in the same document,
            # i.e. pack_mask[:, t, t+1] is True.
            # Indices into the (T-1)-length shifted sequence: anchor=t, target=t+1.
            idx_anchor = torch.arange(T - 1, device=hidden.device)      # 0..T-2
            idx_target = torch.arange(1, T,   device=hidden.device)     # 1..T-1

            # pack_mask: (B, T, T)  →  valid_transitions: (B, T-1)
            valid_transitions = pack_mask[:, idx_anchor, idx_target]     # (B, T-1)

            # Mask out cross-document positions so CE ignores them
            # Flatten to (B*(T-1),) matching target_ids
            target_ids = target_ids.masked_fill(
                ~valid_transitions.reshape(-1), -1
            )

        return F.cross_entropy(
            logits.reshape(-1, V), target_ids, ignore_index=-1
        )


# ---------------------------------------------------------------------------
# jepa_infonce_loss — zero-parameter JEPA via InfoNCE contrastive loss
# ---------------------------------------------------------------------------

def jepa_infonce_loss(hidden:     torch.Tensor,
                      shift_proj: nn.Module,
                      weight:     float = 0.05,
                      temperature:float = 0.1,
                      max_pairs:  int   = 256,
                      pack_mask:  Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    BUG FIX: when pack_mask masks the true positive (diagonal) for a given
    anchor, the CE label still points at that diagonal position but the
    similarity has been set to -1e9 — the loss becomes arbitrarily large and
    incoherent rather than simply ignored.  We now detect rows where the
    diagonal (true positive) is masked and set those target labels to -1 so
    they are excluded via ignore_index.
    """
    B, T, D = hidden.shape
    if T <= 1 or weight == 0.0:
        return hidden.new_zeros(1).squeeze()

    t = min(T - 1, max_pairs)

    # 1. Random sampling instead of static [:t] slice
    if T - 1 > max_pairs:
        # Sort indices to maintain sequence order (good practice for memory access)
        idx, _ = torch.sort(torch.randperm(T - 1, device=hidden.device)[:t])
    else:
        idx = torch.arange(t, device=hidden.device)

    # 2. Gather representations at the chosen random indices
    idx_exp = idx.unsqueeze(0).unsqueeze(2).expand(B, t, D)
    anchors_pre = hidden.gather(1, idx_exp)
    positives   = hidden.gather(
        1, (idx + 1).unsqueeze(0).unsqueeze(2).expand(B, t, D)
    ).detach()

    anchors   = shift_proj(anchors_pre.reshape(B * t, D)).reshape(B, t, D)
    anchors   = F.normalize(anchors,   dim=-1)
    positives = F.normalize(positives, dim=-1)

    # (B, t, t)
    sim = torch.bmm(anchors, positives.transpose(1, 2)) / temperature

    # Labels: diagonal (i, i) — anchor i predicts positive i
    labels = torch.arange(t, device=hidden.device).unsqueeze(0).expand(B, -1)  # (B, t)

    # 3. Mask out cross-document pairs using pack_mask
    if pack_mask is not None:
        # idx_i[i,j] = anchor sequence position for pair (i,j)
        # idx_j[i,j] = positive sequence position for pair (i,j)
        idx_i = idx.unsqueeze(1).expand(t, t)            # anchor positions (t, t)
        idx_j = (idx + 1).unsqueeze(0).expand(t, t)      # positive positions (t, t)

        batch_idx = torch.arange(B, device=hidden.device).view(B, 1, 1)
        valid_mask = pack_mask[batch_idx,
                               idx_i.unsqueeze(0),
                               idx_j.unsqueeze(0)]       # (B, t, t)

        # Suppress cross-document similarities (off-diagonal negatives and
        # any stray cross-doc true-positives on the diagonal)
        sim = sim.masked_fill(~valid_mask, -1e9)

        # BUG FIX: if the TRUE POSITIVE (diagonal entry) is cross-document
        # (valid_mask[:, i, i] == False), the CE label i is now pointing at a
        # -1e9-masked slot — the loss would be enormous and wrong.  Exclude
        # those rows entirely via ignore_index=-1.
        diag_valid = valid_mask[:, torch.arange(t), torch.arange(t)]  # (B, t)
        labels = labels.masked_fill(~diag_valid, -1)

    loss = F.cross_entropy(
        sim.reshape(B * t, t),
        labels.reshape(-1),
        ignore_index=-1,
    )

    return weight * loss