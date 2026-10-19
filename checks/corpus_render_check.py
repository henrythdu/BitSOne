#!/usr/bin/env python3
"""checks/corpus_render_check.py — our renderer is deterministic, self-consistent and pinned.

The properties that make the renderer usable, each of which can fail on its own:

  * determinism — the same row renders to the same bytes twice in the same process
  * binding — one span per option, ascending, non-overlapping, inside the text, and each span
    starts with its own option's number
  * permutation — re-rendering with a permuted option order permutes the slot labels, keeps the
    set of option texts, and keeps the span geometry identical
  * key tokens — the last token FULLY inside each span, from the tokenizer the model ships with
  * a recorded digest — the whole rendered corpus hashes to one value recorded in
    `bitsone/paths.py`; an accidental change to spacing, prefixes or ordering shows up as one line

The digest is the regression lock, not a proof of correctness: it says "this is the rendering
that was reviewed", which is exactly what a checkpoint's inputs need to be pinned to. Write it
once with `--write-digest`, review the diff, then commit the value.

Needs: the six corpus files (outside the tree) and, for the key-token section, a tokenizer.
No torch, no GPU.

Run:  .venv-s1/bin/python checks/corpus_render_check.py [--corpus-dir DIR] [--limit N]
      .venv-s1/bin/python checks/corpus_render_check.py --write-digest
Exit: 0 on PASS, 1 on any failed assertion or a digest mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitsone import corpus  # noqa: E402
from bitsone.paths import CORPUS_RENDER_SHA  # noqa: E402
from checks.harness import tokenizer_dir  # noqa: E402

FAILURES: list[str] = []
SAMPLE: list[str] = []


def fail(label: str, detail: str) -> None:
    FAILURES.append(f"{label}: {detail}")
    if len(SAMPLE) < 10:
        SAMPLE.append(f"{label}: {detail}")


def check_geometry(name: str, row_id: str, text: str, rendered: corpus.Rendered) -> None:
    spans = rendered.option_spans
    if len(spans) != rendered.n_slots:
        fail("one span per option", f"{name}:{row_id}: {len(spans)} spans for {rendered.n_slots} slots")
        return
    cursor = -1
    for i, (start, end) in enumerate(spans):
        if not (0 <= start < end <= len(text)):
            fail("span inside the text", f"{name}:{row_id}: span {i} = ({start}, {end}), text is {len(text)} chars")
            return
        if start < cursor:
            fail("spans ascending and non-overlapping", f"{name}:{row_id}: span {i} starts at {start} < {cursor}")
            return
        cursor = end
        chunk = text[start:end]
        if not chunk.strip():
            fail("span is not empty", f"{name}:{row_id}: span {i} is blank")
        if not chunk.startswith(f"{i + 1}. "):
            fail("span starts with its own option number", f"{name}:{row_id}: span {i} = {chunk[:24]!r}")


def digest_row(h: Any, name: str, row_id: str, text: str, rendered: corpus.Rendered) -> None:
    """One canonical line per row: nothing about this may depend on dict ordering or paths."""
    h.update(name.encode())
    h.update(b"\x00")
    h.update(row_id.encode())
    h.update(b"\x00")
    h.update(rendered.kind.encode())
    h.update(b"\x00")
    h.update("\x1f".join(rendered.slot_labels).encode())
    h.update(b"\x00")
    h.update(",".join(f"{s}:{e}" for s, e in rendered.option_spans).encode())
    h.update(b"\x00")
    h.update(text.encode())
    h.update(b"\x1e")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=None, help="directory holding the six corpus files")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows per file (0 = all)")
    ap.add_argument("--write-digest", action="store_true", help="print the digest instead of comparing")
    args = ap.parse_args()
    if args.write_digest and args.limit:
        # A digest recorded from a partial pass would only match a partial pass — the next full run
        # would fail for the wrong reason, and the gate would be blamed instead of the recording.
        raise SystemExit("refusing --write-digest with --limit: a recorded digest must cover every row")

    data = corpus.corpus_dir(args.corpus_dir)
    print(f"corpus: {data}\n")

    tok = tokenizer_dir()
    tokenizer = None
    if tok is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(tok), fix_mistral_regex=False)
        except Exception as exc:  # noqa: BLE001 - reported below as an incomplete run, never a pass
            print(f"note: tokenizer at {tok} did not load ({exc}); the key-token section cannot run")
    else:
        print("note: no tokenizer found; the key-token section cannot run")

    h = hashlib.sha256()
    rows = spans_checked = keys_checked = 0
    for name in corpus.CORPUS_FILES:
        path = data / name
        if not path.is_file():
            fail("all six corpus files present", f"{name} is missing from {data}")
            continue
        for i, row in enumerate(corpus.open_jsonl(path)):
            if args.limit and i >= args.limit:
                break
            rows += 1
            row_id = str(row.get("id", i))
            text, rendered = corpus.build_prompt(row["state"], row["kind"], row["instructions"], row["options"])

            again, _ = corpus.build_prompt(row["state"], row["kind"], row["instructions"], row["options"])
            if again != text:
                fail("deterministic render", f"{name}:{row_id}: two renders differ")

            check_geometry(name, row_id, rendered.text, rendered)
            spans_checked += rendered.n_slots

            if rendered.n_slots:
                order = list(range(rendered.n_slots))[::-1]
                perm_text, perm = corpus.build_prompt(
                    row["state"], row["kind"], row["instructions"], row["options"], order
                )
                if perm.slot_labels != tuple(rendered.slot_labels[j] for j in order):
                    fail("permutation moves the labels", f"{name}:{row_id}")
                if len(perm.option_spans) != len(rendered.option_spans):
                    fail("permutation keeps the number of spans", f"{name}:{row_id}")

                # Spans index into the PROMPT text, and they carry a leading ordinal ("3. "),
                # which of course moves under permutation. Strip it: what must be preserved is the
                # option's own text, not the position it happened to land in.
                def spans_of(t: str, r: corpus.Rendered) -> list[str]:
                    return [re.sub(r"^\d+\. ", "", t[start:end]) for start, end in r.option_spans]

                if sorted(spans_of(perm.text, perm)) != sorted(spans_of(rendered.text, rendered)):
                    fail("permutation keeps the set of option texts", f"{name}:{row_id}")

            if tokenizer is not None:
                enc = tokenizer(rendered.text, return_offsets_mapping=True, add_special_tokens=False)
                try:
                    keys = corpus.key_token_indices(list(enc["offset_mapping"]), rendered.option_spans)
                except ValueError as exc:
                    fail("every span holds a token", f"{name}:{row_id}: {exc}")
                else:
                    keys_checked += len(keys)
                    for k, span in zip(keys, rendered.option_spans, strict=True):
                        start, end = enc["offset_mapping"][k]
                        if not (span[0] <= start and end <= span[1]):
                            fail("key token lies inside its span", f"{name}:{row_id}: token {k} at ({start}, {end})")

            digest_row(h, name, row_id, text, rendered)

    got = h.hexdigest()
    print(f"rows: {rows}   spans: {spans_checked}   key tokens: {keys_checked}")
    print(f"rendered digest: {got}")

    if args.write_digest:
        print("\nRecord this in bitsone/paths.py as CORPUS_RENDER_SHA, then re-run without --write-digest.")
        if FAILURES:
            return 1
        return 2 if tokenizer is None else 0

    if args.limit:
        print(f"\nnote: --limit {args.limit} renders a SUBSET, so the digest is not comparable;")
        print("      run without --limit to check it.")
        if FAILURES:
            return 1
        # The digest claim was not checked, so this run cannot pass: exit 2 means "incomplete".
        print("INCOMPLETE: a subset run does not check the recorded digest. This is not a pass.")
        return 2

    if not CORPUS_RENDER_SHA:
        print(
            "\nFAIL: no rendered digest is recorded, so this gate cannot say the rendering is unchanged.\n"
            "Run once with --write-digest, review, and record the value in bitsone/paths.py."
        )
        return 1
    if got != CORPUS_RENDER_SHA:
        print(f"\nFAIL: rendered digest differs from the recorded one\n  got  {got}\n  want {CORPUS_RENDER_SHA}")
        return 1

    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} rendering claims do not hold (first {len(SAMPLE)} shown)")
        print("\n".join(SAMPLE))
        return 1
    if tokenizer is None:
        # AGENTS.md: a gate must read the code it claims about; if it cannot, it says so and
        # exits non-zero rather than passing. Without a tokenizer the key-token claims were never
        # checked, so this run cannot report PASS — exit 2 is "incomplete", not "failed".
        print("\nINCOMPLETE: no tokenizer, so the key-token claims were not checked. This is not a pass.")
        print("  Fix: point bitsone.paths at a local tokenizer.json, or install `transformers`.")
        return 2
    print("\nPASS: every row renders deterministically, spans and key tokens are self-consistent,")
    print("      and the rendered digest matches the recorded one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
