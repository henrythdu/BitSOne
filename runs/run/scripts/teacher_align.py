#!/usr/bin/env python3
"""run7_teacher_align.py — is the teacher file aligned to the corpus it addresses?

WHY THIS EXISTS AND WHY IT IS NOT THE WIDTH GUARD
-------------------------------------------------
`train.py:470-483` resolves a teacher row as `ex = train_examples[d["i"]]`, so a
teacher row is attached to a corpus row by POSITION. Nothing at runtime checks
that position means what it meant when the teacher was written.

The trainer does guard, at `:477-479`:

    if len(d["probs"]) != ex.n_options:
        raise ValueError(f"teacher row {d['i']} has {len(d['probs'])} probs for ...")

but that fires only when the mis-resolved row has a DIFFERENT option count.
Measured over the six-file pool, adjacent rows share an option count 80.91% of
the time pooled -- and 100.0% in `adequacy_hs2` and `adequacy_gen`, where an
off-by-one is invisible to it. Its power against an offset-+1 misalignment is
21.43%, against a 19.09% prediction from the adjacent-K rate.

So the guard's silence is weak evidence. This script uses a strong one: the
teacher's `probs` are a distribution over options, and the corpus row carries the
gold `label`. If the alignment is correct then `argmax(probs)` IS the label for
the rows the teacher agreed with gold on, which `distill` reported as 59,525 of
70,408. If the alignment is wrong, agreement collapses toward chance.

It measures itself by injection: it re-runs the same comparison at several
deliberate index offsets and REQUIRES them to collapse. A check that cannot
produce a non-zero reading is decoration.

USAGE
-----
  python3 runs/run/scripts/teacher_align.py
  python3 runs/run/scripts/teacher_align.py --self-test
  python3 runs/run/scripts/teacher_align.py --min-agree 0.90 --max-misaligned 0.60

Exit 0 only if: zero indices out of range, zero width mismatches, agreement at
offset 0 at or above `--min-agree`, and EVERY injected offset at or below
`--max-misaligned`. Any other result exits 1.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # runs/run/scripts/<this file>

SIX = [
    "train_v5",
    "multistep_v14",
    "generated_v16p",
    "generated_v18p",
    "adequacy_hs2",
    "adequacy_gen",
]
DEFAULT_DATA = Path(os.environ.get("BITSONE_CORPUS_DIR", REPO / "data"))
DEFAULT_TEACHER = REPO / "runs/run/teacher/teacher_v20.jsonl"
OFFSETS = (0, 1, -1, 2, -2, 137)

# Recorded 2026-10-14 over the six-file pool, at offset 0.
RECORDED_ROWS = 72_434
RECORDED_AGREE = 0.9790
RECORDED_CHANCE = 0.3220


def load_pool(data_dir: Path) -> tuple[list[int], list[int]]:
    """(option count, gold label) per row, in train_files order."""
    widths: list[int] = []
    labels: list[int] = []
    for name in SIX:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"FATAL: {path} is missing; the pool is not what was measured")
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SystemExit(f"FATAL: cannot read {path}: {exc!r}") from exc
        with fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception as exc:
                    raise SystemExit(f"FATAL: {path}:{n} does not parse: {exc!r}") from exc
                try:
                    widths.append(len(d["options"]))
                    labels.append(int(d["label"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise SystemExit(f"FATAL: {path}:{n} has no usable options/label: {exc!r}") from exc
    return widths, labels


def load_teacher(path: Path) -> list[tuple[int, list[float]]]:
    if not path.exists():
        raise SystemExit(f"FATAL: {path} is missing")
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"FATAL: cannot read {path}: {exc!r}") from exc
    out: list[tuple[int, list[float]]] = []
    with fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append((int(d["i"]), [float(x) for x in d["probs"]]))
            except Exception as exc:
                raise SystemExit(f"FATAL: {path}:{n} does not parse as a teacher row: {exc!r}") from exc
    return out


def score(teacher, widths, labels, offset: int) -> tuple[int, int, int]:
    """(width-guard passes, argmax==label, teacher rows usable at this offset)."""
    wok = agree = usable = 0
    for i, probs in teacher:
        j = i + offset
        if j < 0 or j >= len(widths):
            continue
        usable += 1
        if len(probs) != widths[j]:
            continue  # the guard at train.py:477-479
        wok += 1
        if max(range(len(probs)), key=probs.__getitem__) == labels[j]:
            agree += 1
    return wok, agree, usable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--min-agree", type=float, default=0.90, help="offset 0 must reach this")
    ap.add_argument("--max-misaligned", type=float, default=0.60, help="every injected offset must stay at or below")
    ap.add_argument("--self-test", action="store_true", help="require the injected offsets to collapse; exit 1 if not")
    args = ap.parse_args()

    widths, labels = load_pool(args.data_dir)
    teacher = load_teacher(args.teacher)

    # Chance floor: for a uniformly random row, P(pick the gold option) = sum p_K / K.
    ck = collections.Counter(widths)
    total = len(widths)
    chance = sum(c / total * (1.0 / k) for k, c in ck.items())

    print(f"  pool rows             {total:,}")
    print(f"  teacher rows          {len(teacher):,}")
    print(f"  chance floor          {chance:.4f}")
    print()
    print(f"  {'offset':>8}  {'guard passes':>13}  {'argmax==label':>14}  verdict")

    failures: list[str] = []
    at_zero = None
    for off in OFFSETS:
        wok, agree, usable = score(teacher, widths, labels, off)
        rate = agree / wok if wok else 0.0
        guard = wok / usable if usable else 0.0
        tag = "CORRECT" if off == 0 else "MISALIGNED (injected)"
        print(f"  {off:>+8}  {guard:>13.4f}  {rate:>14.4f}  {tag}")
        if off == 0:
            at_zero = rate
            if wok != len(teacher):
                failures.append(f"offset 0: only {wok:,} of {len(teacher):,} rows passed the width guard")
            if rate < args.min_agree:
                failures.append(f"offset 0 agreement {rate:.4f} < required {args.min_agree:.2f}")
        elif rate > args.max_misaligned:
            failures.append(f"offset {off:+d} agreement {rate:.4f} > {args.max_misaligned:.2f}: the test cannot fire")

    print()
    if at_zero is not None:
        print(f"  agreement at offset 0 = {at_zero:.4f}  ({RECORDED_AGREE:.4f} recorded 2026-10-14)")
        print(f"  chance                = {chance:.4f}  ({RECORDED_CHANCE:.4f} recorded)")

    if args.self_test:
        print()
        print("  SELF-TEST: the injected offsets above must ALL be at or below")
        print(f"  {args.max_misaligned:.2f}. If any is high, this check has no power.")

    if failures:
        print()
        for f in failures:
            print(f"  FATAL: {f}")
        return 1

    print()
    print("  PASS  the teacher is aligned to the pool, and the test fires under injection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
