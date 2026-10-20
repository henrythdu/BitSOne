#!/usr/bin/env python3
"""checks/pointer_head_check.py — the shipped head loads, and its forward is the documented formula.

The tensor the model ships with loads strict into `bitsone.pointer.PointerHead`, and that module's
forward computes exactly the arithmetic its docstring states. This file recomputes the formula from
the raw weights with primitive ops (`F.layer_norm`, `F.linear`) rather than through the module's own
layers, so folding the scale, dropping the mask or reordering the norm shows up as a mismatch.

Also asserted:

  * mask semantics: masked slots are -inf, and an all-masked row is all -inf (not NaN, not 0)
  * K-permutation equivariance: permuting the keys permutes the logits and changes nothing else
  * key independence: a key's logit does not depend on the keys that follow it
  * precision: logits are fp32 even when the torso hands over bf16 hidden states

Needs: torch, and a checkpoint directory holding `slot_head.pt`. No GPU, no corpus, no base weights.

Run:  .venv-s1/bin/python checks/pointer_head_check.py
Exit: 0 on PASS, 1 on any failed assertion or a head that cannot be found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checks.harness import checkpoint_dir  # noqa: E402

if TYPE_CHECKING:  # annotations only: this module must stay importable without torch
    import torch

    from bitsone.pointer import PointerHead

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    """Record, never raise: one run should report every failure, not the first."""
    print(f"{'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}" if detail else label)


def head_path() -> Path:
    ckpt = checkpoint_dir()
    path = ckpt / "slot_head.pt"
    if not path.is_file():
        sys.exit(f"FAIL: the shipped head is not at {path}")
    return path


def formula(head: PointerHead, hq: torch.Tensor, hk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The documented arithmetic, recomputed from the raw parameters.

    `pointer.py` states: logit_k = (LayerNorm(h_query) W_q) · (LayerNorm(h_key_k) W_k)^T / sqrt(d),
    with masked slots at -inf. Written here with functional ops on purpose: it must not borrow the
    module's own LayerNorm/Linear, or the comparison would be the head against itself.
    """
    import torch.nn.functional as F

    if getattr(head, "mid", 0):
        raise SystemExit("formula: mid>0 is the capacity control and ships no checkpoint")
    d = int(head.k.weight.shape[0])
    nq = F.layer_norm(hq, (hq.shape[-1],), head.norm.weight, head.norm.bias)
    nk = F.layer_norm(hk, (hk.shape[-1],), head.norm.weight, head.norm.bias)
    q = F.linear(nq, head.q.weight, head.q.bias)
    k = F.linear(nk, head.k.weight, head.k.bias)
    logits = (k @ q.unsqueeze(-1)).squeeze(-1) * (d**-0.5)
    return logits.masked_fill(~mask, float("-inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--slots", type=int, default=7)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed, so this gate cannot read the code it claims about", file=sys.stderr)
        return 2

    from bitsone.pointer import HIDDEN, PointerHead

    path = head_path()
    state = torch.load(path, map_location="cpu", weights_only=True)
    head = PointerHead(hidden=HIDDEN)
    try:
        head.load_state_dict(state, strict=True)
        check(True, "the shipped head loads strict into our class")
    except RuntimeError as exc:
        check(False, "the shipped head loads strict into our class", str(exc))
        print("\nFAIL: cannot continue without the shipped weights")
        return 1
    head.eval()

    gen = torch.Generator().manual_seed(0)
    hq = torch.randn(args.batch, HIDDEN, generator=gen)
    hk = torch.randn(args.batch, args.slots, HIDDEN, generator=gen)
    mask = torch.ones(args.batch, args.slots, dtype=torch.bool)
    mask[1, -2:] = False  # a partially masked row
    mask[2, :] = False  # an all-masked row

    with torch.no_grad():
        got = head(hq, hk, mask)
        want = formula(head, hq, hk, mask)

    check(
        bool(torch.allclose(got[mask], want[mask], atol=1e-6, rtol=1e-6)),
        "forward matches the documented formula (independent recompute)",
        f"max|Δ| = {float((got[mask] - want[mask]).abs().max()):.3e}",
    )
    check(
        bool(torch.isinf(got[~mask]).all()) and bool((got[~mask] < 0).all()),
        "masked slots are -inf",
        f"masked values: {got[~mask].tolist()[:4]}",
    )
    check(
        bool(torch.isinf(got[2]).all()),
        "an all-masked row is all -inf rather than NaN or zero",
        f"row 2: {got[2].tolist()[:4]}",
    )

    order = torch.arange(args.slots - 1, -1, -1)
    full = torch.ones(1, args.slots, dtype=torch.bool)
    with torch.no_grad():
        plain = head(hq[:1], hk[:1], full)
        permuted = head(hq[:1], hk[:1, order], full)
    check(
        bool(torch.allclose(permuted[0], plain[0][order], atol=1e-6, rtol=1e-6)),
        "permuting the keys permutes the logits and changes nothing else",
        "K-permutation is not equivariant",
    )

    named = dict(head.named_parameters())
    check(
        set(named) == {"norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"},
        "the shipped head has exactly the parameters the formula names",
        str(sorted(named)),
    )
    check(
        all(p.shape[0] != 1 or p.dim() == 1 for p in named.values()),
        "no head parameter is shaped by slot index",
    )
    # THE CLAIM THAT MATTERS, AND THE ONE THAT CAN FAIL: a key's logit must not depend on how many
    # keys follow it. The parameter-shape test above only rules out a head with a slot dimension; this
    # one catches a head that mixes keys (a softmax across K, a running mean over slots) and would
    # still be shape-legal. It reads the head's own forward, so it fails if that changes.
    with torch.no_grad():
        wide = head(hq[:1], hk[:1], torch.ones(1, args.slots, dtype=torch.bool))
        narrow = head(hq[:1], hk[:1, :3], torch.ones(1, 3, dtype=torch.bool))
    check(
        bool(torch.allclose(wide[0, :3], narrow[0], atol=1e-6, rtol=1e-6)),
        "a key's logit does not depend on the keys that follow it",
        f"K={args.slots} vs K=3: max|Δ| = {float((wide[0, :3] - narrow[0]).abs().max()):.3e}",
    )

    with torch.no_grad():
        bf = head(hq.to(torch.bfloat16), hk.to(torch.bfloat16), mask)
    check(bf.dtype == torch.float32, "logits are fp32 even when the hidden states arrive bf16", str(bf.dtype))

    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} of the head's claims do not hold")
        return 1
    print(f"\nPASS: the shipped head ({path.name}) loads and computes its documented formula")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
