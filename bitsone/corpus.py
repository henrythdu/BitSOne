"""corpus — the six corpus files, read and rendered here.

This is the data half of the pipeline: it reads the six JSONL files in the order the teacher's index
space was built on, and renders one row into the prompt the pointer head consumes. It also owns
where those six files are found (`corpus_dir`), so the gates and the trainers resolve one fact once.

The contract it implements (verified over all 123,339 rows by `checks/corpus_render_check.py`,
which pins the rendered text with one recorded digest):

  * one option is exactly ONE line, numbered from 1, so slot ``k`` is the line reading
    ``k+1.``; the pointer head scores option k from the hidden state at the LAST token of
    that line — the token that has just read the whole option under causal attention;
  * ``option_spans`` are CHARACTER spans, so a caller maps them to tokens with the
    tokeniser's offset mapping rather than re-deriving the prompt segment by segment;
  * descriptions collapse to a single line, joined to the name by an em dash;
  * ``noul`` has exactly two slots, labels ``('false', 'true')``, and falls back to default
    descriptions only for labels the row does not supply — a supplied empty string WINS,
    which is why the merge is by presence, not by truthiness;
  * ``score`` labels are the level indices as strings;
  * option order is a per-epoch augmentation, so a permutation is applied to the RENDERED
    list and the resulting slot -> label binding is returned; the caller never tracks it.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# The six files, in the order run 7's `train_files` names them. The order IS the teacher's
# index space (teacher row i aligns with concatenated corpus row i), so it is not cosmetic.
CORPUS_FILES: tuple[str, ...] = (
    "train_v5.jsonl",
    "multistep_v14.jsonl",
    "generated_v16p.jsonl",
    "generated_v18p.jsonl",
    "adequacy_hs2.jsonl",
    "adequacy_gen.jsonl",
)


def corpus_dir(explicit: str | Path | None = None, *, near: Path | None = None) -> Path:
    """The directory holding all six files. They are third-party, so they live outside this repo.

    Order: `explicit` (the `--corpus-dir` flag) wins, then `$BITSONE_CORPUS_DIR`, then the
    repository's own `data/` directory, then the `data/` sibling of `near`. A directory counts only if ALL six files are in it, so a partial
    corpus fails here rather than half-way through a run.

    Raises `SystemExit` rather than a plain exception because all four callers are CLIs or gates
    whose contract is "say why, exit non-zero": that costs one line here instead of a try/except in
    each `main()`. A caller that wants to catch it can — `SystemExit` is an ordinary exception.
    """
    cands: list[str | Path | None] = [
        explicit,
        os.environ.get("BITSONE_CORPUS_DIR"),
        Path(__file__).resolve().parents[1] / "data",
        Path(near).parent / "data" if near else None,
    ]
    tried: list[str] = []
    nearest: tuple[Path, list[str]] | None = None
    for cand in cands:
        if not cand:
            continue
        path = Path(cand)
        tried.append(str(path))
        missing = [f for f in CORPUS_FILES if not (path / f).is_file()]
        if not missing:
            return path.resolve()
        if len(missing) < len(CORPUS_FILES) and (nearest is None or len(missing) < len(nearest[1])):
            nearest = (path, missing)

    lines = [
        "the six corpus files are not available: they are third-party and are not in this repository;",
        "point --corpus-dir (or BITSONE_CORPUS_DIR) at the directory holding all six.",
    ]
    if nearest is not None:
        lines.append(f"  nearest: {nearest[0]} is missing {', '.join(nearest[1])}")
    if tried:
        lines.append(f"  tried: {', '.join(dict.fromkeys(tried))}")
    raise SystemExit("\n".join(lines))


NOUL_SLOT_LABELS: tuple[str, str] = ("false", "true")
NOUL_DEFAULT_CRITERIA: dict[str, str] = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}
HEADERS: dict[str, str] = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}
EM_DASH = " \u2014 "


@dataclass(frozen=True)
class Rendered:
    """A question flattened to text, plus the binding needed to read an answer back."""

    text: str
    slot_labels: tuple[str, ...]
    slot_descriptions: tuple[str, ...]
    kind: str
    option_spans: tuple[tuple[int, int], ...]

    @property
    def n_slots(self) -> int:
        return len(self.slot_labels)


def render_content(content: Any) -> str:
    """Flatten a state, instruction or description into text, stably.

    Dicts and lists are emitted as indented JSON rather than ``str()`` so key order and
    unicode are deterministic: the same state must always tokenise identically.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


def render_state(state: Any) -> str:
    """The shared prefix: identical across the questions of one request, so it caches once."""
    return f"<state>\n{render_content(state)}\n</state>\n"


def option_block(pairs: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Number the options from 1 and return each line's span within the block.

    A name must not contain a line break. A description is collapsed to one line. Therefore one
    option is exactly one line, and a span can mark it without ambiguity.
    """
    for name, _ in pairs:
        if "\n" in name or "\r" in name:
            raise ValueError(f"option name contains a line break: {name!r}")
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        line = f"{i + 1}. {name}" + (f"{EM_DASH}{desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # the joining newline
    return "\n".join(lines), spans


def pairs_of(kind: str, options: list[list[str]]) -> list[tuple[str, str]]:
    """The (name, description) pairs of one row, before rendering.

    `noul` ignores the row's names and uses the fixed label order; `score` uses the level
    index as the name; `choice` keeps the row's order.
    """
    if kind == "noul":
        supplied = {name: desc for name, desc in options}
        merged = {**NOUL_DEFAULT_CRITERIA, **supplied}  # a supplied "" wins over the default
        return [(label, merged.get(label, "")) for label in NOUL_SLOT_LABELS]
    if kind == "choice":
        return [(name, desc) for name, desc in options]
    if kind == "score":
        return [(str(i), desc) for i, (_name, desc) in enumerate(options)]
    raise ValueError(f"unknown kind {kind!r}")


def render_question(
    kind: str,
    instructions: Any,
    options: list[list[str]],
    option_order: list[int] | None = None,
) -> Rendered:
    """Render one row's question. `option_order` permutes the rendered list (augmentation)."""
    if kind not in HEADERS:
        raise ValueError(f"unknown kind {kind!r}")
    pairs = pairs_of(kind, options)
    # choice descriptions are rendered (they may be structured); noul too; score rubrics are
    # plain strings and are normalised inside `option_block`.
    if kind != "score":
        pairs = [(name, render_content(desc)) for name, desc in pairs]
    if option_order is not None:
        if sorted(option_order) != list(range(len(pairs))):
            raise ValueError("option_order must be a permutation of the option indices")
        pairs = [pairs[i] for i in option_order]

    block, block_spans = option_block(pairs)
    prefix = f'<question type="{kind}">\n{HEADERS[kind]}\n{render_content(instructions)}\n<options>\n'
    text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return Rendered(
        text=text,
        slot_labels=tuple(name for name, _ in pairs),
        slot_descriptions=tuple(str(desc or "").strip() for _, desc in pairs),
        kind=kind,
        option_spans=tuple((base + s, base + e) for s, e in block_spans),
    )


def build_prompt(
    state: Any, kind: str, instructions: Any, options: list[list[str]], option_order: list[int] | None = None
) -> tuple[str, Rendered]:
    """The full prompt: the shared state prefix followed by the question.

    `option_spans` come back in PROMPT coordinates -- shifted by the state prefix -- so a caller
    tokenises `text` and uses them directly. `render_question` returns question-local spans; the
    shift happens here, once, because a caller that forgets it does not fail: it resolves the key
    tokens to the state region and still returns plausible scores.
    """
    rendered = render_question(kind, instructions, options, option_order)
    state_text = render_state(state)
    base = len(state_text)
    return state_text + rendered.text, Rendered(
        text=state_text + rendered.text,
        slot_labels=rendered.slot_labels,
        slot_descriptions=rendered.slot_descriptions,
        kind=rendered.kind,
        option_spans=tuple((s + base, e + base) for s, e in rendered.option_spans),
    )


def permutation(n: int, rng: random.Random) -> list[int]:
    """A uniform option permutation, the way the collator draws one per epoch."""
    order = list(range(n))
    rng.shuffle(order)
    return order


def all_instructions(row: dict[str, Any]) -> list[str]:
    """Canonical phrasing first, then the stored variants."""
    return [row["instructions"], *(row.get("instruction_variants") or [])]


def key_token_indices(offsets: list[tuple[int, int]], spans: tuple[tuple[int, int], ...]) -> list[int]:
    """The last token FULLY inside each option span — the pointer head's key position.

    "Fully inside" is the rule, not "overlapping": a line-final '.', '!' or ',' merges with the
    following newline into ONE token (``.\\n``), which starts inside the span and ends outside
    it, so the key is the token before it. A space before the newline breaks that merge and the
    punctuation stays inside. Raises when a span contains no token at all, rather than silently
    returning a neighbour.

    ZERO-WIDTH offsets are skipped: a token covering no characters is not inside a span, and
    counting it would let a tokenizer's bookkeeping decide the key position. `checks/corpus_render_check.py`
    recomputes this rule over every row of all six files and asserts the key tokens land inside their
    spans, so this is checked rather than trusted.
    """
    keys: list[int] = []
    for start, end in spans:
        inside = [i for i, (s, e) in enumerate(offsets) if e > s and s >= start and e <= end]
        if not inside:
            raise ValueError(f"no token lies inside option span ({start}, {end})")
        keys.append(inside[-1])
    return keys


def split_examples(
    rows: list[dict[str, Any]], *, val_fraction: float = 0.03, seed: int = 0, group_by_task: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train/val split, stratified by task.

    Stratified by task, because the corpus mixes tasks with very different sizes and an unstratified
    slice can miss a small one entirely. Per task, in SORTED task order on ONE shared rng: shuffle,
    cut at `max(1, int(n * (1 - val_fraction)))` when n > 1, else keep the single row in train. Then
    both sides are shuffled. Every one of those choices is observable: a different order, a different
    cut, or a second rng gives a different split, and the split decides what `val` means.
    """
    rng = random.Random(seed)
    if not group_by_task:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1 - val_fraction))
        return shuffled[:cut], shuffled[cut:]

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task", "unknown")), []).append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for task in sorted(by_task):
        items = by_task[task]
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_fraction))) if len(items) > 1 else 1
        train.extend(items[:cut])
        val.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def parse_row(line: str, path: Path, lineno: int) -> dict[str, Any]:
    """Parse one JSONL line, naming the file and line if it is malformed.

    A bare `json.loads` raises `JSONDecodeError` with the offset inside the LINE, which on a six-file,
    123,339-row corpus says nothing about where to look. Every failure path (bad JSON, not an object)
    is re-raised with the coordinate that identifies the row.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{lineno}: not JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
    return row


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Read JSONL rows (gzip-aware) in file order. The concatenation IS the index space."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if line:
                    rows.append(parse_row(line, path, lineno))
    return rows


def open_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows without holding the corpus in memory (the render gate walks all of it)."""
    opener: Any = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                yield parse_row(line, path, lineno)
