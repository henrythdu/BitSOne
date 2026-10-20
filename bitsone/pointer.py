"""pointer — the pointer head, its losses, and the batch builders training uses.

Spec: experiments/ternary-s1/PREREG-run-4.md §2. A pointer head asks the torso to compare a
question slot against each option's own hidden state instead of choosing among 26 letters:

    logit_k = (LayerNorm(h_query) W_q) · (LayerNorm(h_key_k) W_k)ᵀ / sqrt(d)

so K is unbounded and the head holds no positional parameter at all. What it does NOT have: option
k's vector still comes from a causal pass, so it has read options 1..k and not the ones after it —
see `checks/serving_replay_check.py`, which measures how often that changes the answer. The head is
~1.3M params instead of a 128,256-way vocabulary read. It runs in fp32 even when the torso is bf16 —
a low-precision classifier is a needless source of calibration error.

The dump exists so Stage A is a single-variable test: hidden states are produced ONCE on the
deployment path (packed ternary, offline) and the head is then trained on cached vectors, with no
torso in the loop. What lives here is the head, its losses and the batch builders; the renderer it
consumes is `bitsone.corpus`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

HIDDEN = 2560  # BitNet b1.58 2B4T
D_HEAD = 256  # pointer dimension, fixed by the pre-registration


class PointerHead(nn.Module):
    """LayerNorm -> W_q / W_k -> scaled dot product. K-independent: no parameter knows the slot index.

    `checks/pointer_head_check.py` recomputes this formula from the weights. Each axis below is a
    pre-registered choice:

      * One LayerNorm for both sides. Separate norms give each side its own scale, and the dot
        product would then mix the match score with that scale. No measurement separates the two
        forms, so this form is pre-registered, not a measured claim.
      * Projection biases at the torch default (True). A bias is not redundant after a LayerNorm:
        (k+b_k)·(q+b_q) = k·q + k·b_q + b_k·q + b_k·b_q, and every added term depends on k, so bias
        changes the ranking.
      * Dropout 0.0 (Identity): the pre-registered head has none.
    """

    def __init__(self, hidden: int = HIDDEN, d: int = D_HEAD, dropout: float = 0.0, mid: int = 0) -> None:
        """`mid=0` is the pre-registered primary head. `mid>0` is the **H4e capacity control**: the
        same class, same norm, same shared-projection shape, with one extra `mid`-wide hidden layer
        before each projection, so capacity is the ONLY variable that changes. PREREG §3 pins it to
        `mid=512`; H4e is reachable only when H4a is not a PASS and can only ever weaken a FAIL.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.mid = mid
        if mid:
            self.q_pre = nn.Sequential(nn.Linear(hidden, mid), nn.GELU(), nn.Linear(mid, d))
            self.k_pre = nn.Sequential(nn.Linear(hidden, mid), nn.GELU(), nn.Linear(mid, d))
        else:
            self.q = nn.Linear(hidden, d)
            self.k = nn.Linear(hidden, d)
        self.scale = d**-0.5

    def forward(self, h_query: torch.Tensor, h_keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B,H) + (B,K,H) + (B,K) bool -> (B,K) fp32 logits, masked positions at -inf.

        The inputs are cast to fp32 first, not only the outputs. A bf16 torso passes bf16 hidden
        states: an fp32 Linear raises a dtype error outside autocast, and under autocast it computes
        the projection in bf16. This cast makes the head run in fp32 whatever the caller does.

        Masking is inside the head, so position correctness is tested in one place.
        """
        nq = self.dropout(self.norm(h_query.float()))
        nk = self.dropout(self.norm(h_keys.float()))
        if self.mid:
            q = self.q_pre(nq).float().unsqueeze(-1)
            k = self.k_pre(nk).float()
        else:
            q = self.q(nq).float().unsqueeze(-1)
            k = self.k(nk).float()
        logits = (k @ q).squeeze(-1) * self.scale
        return logits.masked_fill(~mask, float("-inf"))


def pointer_soft_ce(logits: torch.Tensor, gold_probs: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy over the K logits against the row's existing gold distribution.

    Same objective family as letter-based scoring, so the head is the only change under test. Rows are ragged in K, so each row's gold is truncated to its own K.
    """
    total = logits.new_zeros(())
    for i in range(logits.shape[0]):
        valid = ~torch.isinf(logits[i])
        k = int(valid.sum())
        # k's RANGE is a separate hole from k's POSITION, and the position assert below does not
        # cover it: an all-masked row gives k=0, the prefix assert passes vacuously, the empty slice
        # sums to 0.0 and the row silently contributes NOTHING to the loss behind a plausible curve
        # (measured: no raise, contribution 0.0). k wider than the gold row would slice past it.
        if k == 0 or k > gold_probs.shape[1]:
            raise ValueError(
                f"pointer_soft_ce row {i}: k={k} out of range for a gold row of width "
                f"{gold_probs.shape[1]} -- an all-masked row would contribute 0.0 silently"
            )
        # The [:k] slice below is correct ONLY while the valid slots are the LEADING k. A
        # right-padded row satisfies that; a left-padded one does not, and there k still counts
        # the valid slots correctly while [:k] cuts the WRONG entries -- the loss would then train
        # on masked positions behind a plausible-looking curve. Assert the prefix rather than
        # trust the convention. (head_probs needs no such assert: it selects with this same
        # boolean mask, which is position-agnostic.)
        if not bool(valid[:k].all()) or bool(valid[k:].any()):
            raise ValueError(
                f"pointer_soft_ce row {i}: valid slots are not the leading {k} "
                f"(valid={valid.tolist()}) -- the [:k] slice would cut the wrong entries"
            )
        logp = torch.log_softmax(logits[i, :k].float(), dim=-1)
        total = total - (gold_probs[i, :k].float() * logp).sum()
    return total / logits.shape[0]


def dense_batch(
    dump: dict[str, Any], idx: torch.Tensor, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ragged dump -> (q [B,H], k [B,maxK_batch,H], mask [B,maxK_batch]), non-mutating.

    DENSIFIES ONE BATCH ONLY (256 rows), never the whole split: transient peak is
    256 × 16 × 2560 × 4 B ≈ 42 MB. The dense alternative is N × maxK × H: at the corpus' 123,339 rows
    and maxK 16 that is ≈ 20.2 GB fp32 (18.8 GiB). The numbers are here so nobody "optimises" this
    into the version that does not fit.
    """
    off = dump["offset"]
    h = dump["k"].shape[1]
    ks = [dump["k"][int(off[i]) : int(off[i + 1])] for i in idx.tolist()]
    maxk = max(int(x.shape[0]) for x in ks)
    q = dump["q"][idx].to(device).float()
    k = torch.zeros(len(ks), maxk, h, device=device)
    mask = torch.zeros(len(ks), maxk, dtype=torch.bool, device=device)
    for j, x in enumerate(ks):
        n = int(x.shape[0])
        k[j, :n] = x.to(device).float()
        mask[j, :n] = True
    return q, k, mask


def gold_ragged(dump: dict[str, Any], idx: torch.Tensor, maxk: int, device: str = "cuda") -> torch.Tensor:
    """Ragged flat gold -> dense (B, maxk) aligned to dense_batch's own ragged slices.

    Read from the DUMP, never re-derived from the source rows: the dump's gold was normalised and
    permuted at dump time, so re-deriving it would be a second source of truth for the targets.
    """
    off = dump["offset"]
    g = dump["gold"]
    out = torch.zeros(len(idx), maxk, device=device)
    for j, i in enumerate(idx.tolist()):
        sl = g[int(off[i]) : int(off[i + 1])]
        out[j, : len(sl)] = sl.to(device)
    return out
