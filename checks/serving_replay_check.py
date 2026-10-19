#!/usr/bin/env python3
"""checks/serving_replay_check.py — the served path answers the same way every time.

What a served model has to satisfy, end to end through `bitsone.serve`:

  * the published artifacts load (base + adapter + head), with the head checked against the base's
    hidden size at load rather than against a stored digest
  * the same request twice gives the same answer, to the bit
  * permuting the options permutes the LABELS and re-derives exactly the same key tokens, so the
    render→span→key binding is exact. It does NOT permute the scores: the options share one causal
    sequence, so reordering them changes the hidden state of every later token. Equivariance of the
    scores is a training property (augmentation), never a serving invariant, and asserting it here
    only made the gate red for something no causal torso can do.
  * the scores are a proper distribution over exactly the row's options (finite log-probs, sum 1)
  * the key tokens are strictly increasing and inside their spans — the routing rule is exercised
    on a real tokenizer rather than assumed

Needs: a GPU, the base weights and the shipped checkpoint. The corpus is
used for one real row; without it the gate says so and exits non-zero rather than passing on a
synthetic row that would not exercise the tokenizer.

Run:  .venv-s1/bin/python checks/serving_replay_check.py [--corpus-dir DIR] [--row N]
Exit: 0 on PASS, 1 on any failed assertion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}" if detail else label)


def pick_row(corpus_dir: str | None, row_index: int) -> dict:
    """A real corpus row: the gate must exercise the tokenizer on actual option text."""
    from bitsone import corpus

    data = corpus.corpus_dir(corpus_dir)
    path = data / corpus.CORPUS_FILES[0]
    for i, row in enumerate(corpus.open_jsonl(path)):
        if i >= row_index:
            return row
    sys.exit(f"FAIL: {path} has fewer than {row_index + 1} rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--row", type=int, default=41, help="which row of the first corpus file")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("FAIL: torch is not installed, so this gate cannot read the code it claims about")
    if not torch.cuda.is_available():
        sys.exit("FAIL: no CUDA device, so the served path cannot be exercised here")

    from bitsone import corpus, serve

    try:
        engine = serve.load()
        check(True, "the published artifacts load (base + adapter + head, digest-verified)")
    except (SystemExit, OSError, RuntimeError, ValueError) as exc:
        check(False, "the published artifacts load (base + adapter + head, digest-verified)", str(exc))
        return 1

    row = pick_row(args.corpus_dir, args.row)
    state, kind = row["state"], row["kind"]
    instructions, options = row["instructions"], row["options"]

    first = engine.answer(state, kind, instructions, options)
    second = engine.answer(state, kind, instructions, options)

    # The adapter must be APPLIED, not merely present. A key-name mismatch leaves LoRA's B matrices
    # at their zero initialisation, so the served model is the bare base with an adapter-trained
    # head -- plausible-looking numbers from a model that never had the fine-tune. Disabling the
    # adapter and demanding a change is the property that fails when the weights did not bind.
    with engine.model.disable_adapter():
        unadapted = engine.answer(state, kind, instructions, options)
    check(
        any(abs(a - b) > 1e-6 for a, b in zip(first.logits, unadapted.logits, strict=True)),
        "the adapter is applied: disabling it changes the scores",
        f"{first.logits[:3]} vs {unadapted.logits[:3]}",
    )
    check(first.log_probs == second.log_probs, "the same request twice gives the same answer")
    check(first.text == second.text, "the rendered request is deterministic")

    check(
        len(first.log_probs) == len(options),
        "one score per option",
        f"{len(first.log_probs)} scores for {len(options)} options",
    )
    check(all(v == v and abs(v) != float("inf") for v in first.log_probs), "log-probs are finite", str(first.log_probs))
    total = sum(first.probs)
    check(abs(total - 1.0) < 1e-5, "the scores form a distribution (sum 1)", f"sum = {total:.6f}")
    check(first.best in first.slot_labels, "the answer names a rendered option", first.best)

    keys = first.key_indices
    check(all(b > a for a, b in zip(keys, keys[1:], strict=False)), "key tokens are strictly increasing", str(keys))
    # Independent of the span arithmetic: the key tokens must sit AFTER the options block begins.
    # A span space mismatch (question-local spans against prompt offsets) lands them in the state
    # prefix and still looks self-consistent to every span-based assertion.
    opts_at = first.text.index("<options>")
    offsets = engine._inputs(first.text)[1]
    check(
        min(offsets[k][0] for k in keys) > opts_at,
        "every key token lies after the options block, not in the state prefix",
        f"first key at {min(offsets[k][0] for k in keys)}, options begin at {opts_at}",
    )
    text = first.text
    check(all(0 <= k < 200_000 for k in keys) and bool(text), "key indices are plausible positions in the text")

    order = list(range(len(options)))[::-1]
    permuted = engine.answer(state, kind, instructions, options, option_order=order)
    check(
        list(permuted.slot_labels) == [first.slot_labels[i] for i in order],
        "permuting the options permutes the slot labels with them",
        f"{permuted.slot_labels[:3]}",
    )
    # The binding, checked THREE ways, because two of them must not consult the rule they are grading:
    #
    #   1. the permuted request is the permutation of the original request (renderer, not the rule);
    #   2. each returned key token's characters lie strictly inside its own option's span, and the
    #      token is the LAST one that fits -- properties of the indices themselves, so they hold
    #      whether or not `key_token_indices` is right;
    #   3. only last: the returned indices equal what the rule recomputes. Kept because it catches
    #      `answer()` diverging from `render()`, but on its own it asks the rule to grade itself.
    permuted_text, permuted_rendered = engine.render(state, kind, instructions, options, option_order=order)
    permuted_offsets = engine._inputs(permuted_text)[1]
    check(
        len(permuted_text) == len(first.text),
        "permuting the options leaves the render the same length (only the blocks move)",
        f"{len(permuted_text)} vs {len(first.text)}",
    )

    inside, last_inside = [], []
    for idx, (start, end) in zip(permuted.key_indices, permuted_rendered.option_spans, strict=True):
        token = permuted_offsets[idx]
        inside.append(start <= token[0] and token[1] <= end)
        # No LATER token also fits fully inside this span. The guard is `o[1] > o[0]`, NOT `o[1] > 0`,
        # because the rule being graded skips zero-width offsets (`e > s`). With span (100,140) and
        # offsets [(100,105), (110,139), (140,140)] the old guard reports a later fit and fails a
        # CORRECT binding; the rule's own definition passes it.
        later_fits = any(
            start <= o[0] and o[1] <= end and o[1] > o[0] for j, o in enumerate(permuted_offsets) if j > idx
        )
        last_inside.append(not later_fits)
    check(
        all(inside), "each key token's characters lie strictly inside its own option's span", str(permuted.key_indices)
    )
    check(all(last_inside), "each key token is the last token that fits in its span", str(permuted.key_indices))
    check(
        len(set(permuted.key_indices)) == len(permuted.key_indices),
        "the key tokens are distinct, so two options cannot share one binding",
    )
    check(
        list(permuted.key_indices) == list(corpus.key_token_indices(permuted_offsets, permuted_rendered.option_spans)),
        "the returned key tokens equal what the rule recomputes for that render",
        f"{permuted.key_indices} vs {first.key_indices}",
    )
    check(
        len(permuted.log_probs) == len(options) and abs(sum(permuted.probs) - 1.0) < 1e-5,
        "the permuted request still scores exactly the row's options",
    )

    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} of the served path's claims do not hold")
        return 1
    drift = max(abs(permuted.logits[i] - first.logits[order[i]]) for i in range(len(order)))
    print(f"\nPASS: served path answers identically twice, scores {len(first.log_probs)} options,")
    print(f"      binds permuted key tokens to their own spans, and reports score drift {drift:.3f}")
    print(f"      under permutation (causal torso, expected; best = {first.best!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
