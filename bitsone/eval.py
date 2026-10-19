#!/usr/bin/env python3
"""eval — the sealed, read-once evaluation of a checkpoint.

One request per row through the same path serving uses (`bitsone.serve.Engine.answer`): the
renderer, the key-token rule and the pointer head. Nothing here re-implements scoring, so a number
from this file cannot disagree with what the model actually answers.

Read-once is enforced, not promised: the first run of a split writes a receipt under
`runs/eval/`, and a second run of the same split refuses unless `--allow-reread` gives a reason,
which is recorded in the new receipt. A split read a second time without that reason would make
every later number on it partly a training signal, so refusing is the point.

Run:
    .venv-s1/bin/python -m bitsone.eval --split "Training Data/<split>.jsonl" --limit 200
    .venv-s1/bin/python -m bitsone.eval --split <file> --allow-reread "why"

Needs: GPU + the base weights + a checkpoint (published artifacts by default). Not part of CI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bitsone import corpus

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "runs" / "eval"
BINS = 10


def ece(pairs: list[tuple[float, float]]) -> float:
    """Expected calibration error over `BINS` equal-width confidence buckets."""
    if not pairs:
        return float("nan")
    total, acc = 0.0, 0.0
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        bucket = [(conf, hit) for conf, hit in pairs if lo <= conf < hi or (b == BINS - 1 and conf == 1.0)]
        if bucket:
            conf_mean = sum(c for c, _ in bucket) / len(bucket)
            hit_mean = sum(h for _, h in bucket) / len(bucket)
            total += abs(conf_mean - hit_mean) * len(bucket)
            acc += len(bucket)
    return total / acc if acc else float("nan")


def receipt_path(split: Path) -> Path:
    return RECEIPTS / f"{split.stem}.read.json"


def claim(split: Path, allow_reread: str | None) -> None:
    """Refuse a second read of a split unless a reason is given; record either way."""
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    path = receipt_path(split)
    prior: list[dict[str, Any]] = []
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8")).get("reads", [])
        except (OSError, ValueError):
            prior = []
        if not allow_reread:
            raise SystemExit(
                f"{split.name} has already been read ({len(prior)} time(s)): {path}\n"
                '  A second read is a decision, not a rerun. Pass --allow-reread "<why>".'
            )
    prior.append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": allow_reread or "first read",
        }
    )
    path.write_text(json.dumps({"split": str(split), "reads": prior}, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-once evaluation through the serving path.")
    ap.add_argument("--split", required=True, help="JSONL of rows: state, kind, instructions, options, label")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    ap.add_argument("--checkpoint", default=None, help="local checkpoint dir (default: published artifacts)")
    ap.add_argument("--base", default=None, help="local base weights (default: bitsone.paths.base_model)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="report path (default: runs/eval/<split>.report.json)")
    ap.add_argument("--allow-reread", default=None, help="reason a second read is legitimate")
    args = ap.parse_args()

    split = Path(args.split)
    if not split.is_file():
        raise SystemExit(f"{split}: not a file")
    claim(split, args.allow_reread)

    from bitsone.serve import load

    engine = load(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        base=Path(args.base) if args.base else None,
        device=args.device,
    )

    rows = hits = 0
    nll = 0.0
    conf_all: list[tuple[float, float]] = []
    per_kind: dict[str, Counter[str]] = {}
    per_kind_conf: dict[str, list[tuple[float, float]]] = {}
    for row in corpus.open_jsonl(split):
        if args.limit and rows >= args.limit:
            break
        rows += 1
        resp = engine.answer(row["state"], row["kind"], row["instructions"], row["options"])
        gold = int(row["label"])
        probs = resp.probs
        best = max(range(len(probs)), key=probs.__getitem__) if probs else -1
        hit = best == gold
        hits += int(hit)
        nll -= resp.log_probs[gold] if 0 <= gold < len(resp.log_probs) else 0.0
        conf = max(probs) if probs else 0.0
        conf_all.append((conf, float(hit)))
        kind = str(row["kind"])
        per_kind.setdefault(kind, Counter())["n"] += 1
        per_kind[kind]["hits"] += int(hit)
        per_kind_conf.setdefault(kind, []).append((conf, float(hit)))

    if not rows:
        raise SystemExit(f"{split}: no rows read")

    by_kind: dict[str, dict[str, float]] = {
        k: {"n": float(c["n"]), "accuracy": c["hits"] / c["n"], "ece": ece(per_kind_conf.get(k, []))}
        for k, c in per_kind.items()
    }
    report: dict[str, Any] = {
        "split": str(split),
        "rows": rows,
        "accuracy": hits / rows,
        "nll": nll / rows,
        "ece": ece(conf_all),
        "mean_confidence": sum(c for c, _ in conf_all) / rows,
        "by_kind": by_kind,
        "checkpoint": str(args.checkpoint or "published artifacts"),
        "reread": args.allow_reread is not None,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else RECEIPTS / f"{split.stem}.report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(
        f"rows {rows}   accuracy {report['accuracy']:.5f}   nll {report['nll']:.4f}   "
        f"ece {report['ece']:.4f}   mean conf {report['mean_confidence']:.4f}"
    )
    for kind, stats in sorted(by_kind.items()):
        print(f"  {kind:8} n={stats['n']:<6} acc {stats['accuracy']:.4f}   ece {stats['ece']:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
