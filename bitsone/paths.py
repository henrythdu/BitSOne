"""The filesystem locations this repository uses, in one place.

The base model path and the checkpoint location are defined here, not in each caller. A location
that appears in several modules changes in one place, and `checkpoint_dir` raises when the files
are absent instead of guessing.

`CORPUS_RENDER_SHA` is the one digest this repository keeps. It is not a provenance claim. It locks
the renderer: a change to `corpus.py` that alters the prompt text fails
`checks/corpus_render_check.py`.

This module uses the standard library only, so CI can import it without torch.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where build evidence may live. Ordered: the repo's own run tree first, then the exported copy.
EVIDENCE_ROOTS: tuple[Path, ...] = (ROOT / "runs", ROOT / "System One Models")

# Where the base weights live, and what its config must declare. The declaration is a guard: a
# directory that points at a differently quantised model is refused.
BASE_DIR = ROOT / "BitNet Models" / "bitnet-b1.58-2b-4t"
BASE_CONFIG_REQUIRED = {"quantization_mode": "offline", "linear_class": "autobitlinear"}

# The two files a shipped checkpoint must contain. These are names, not hashes: `bitsone.serve.load`
# checks the head against the base's hidden size.
HEAD_NAME = "slot_head.pt"
ADAPTER_DIR = "lora"
ADAPTER_NAME = "adapter_model.safetensors"

# The rendered corpus, one digest. `checks/corpus_render_check.py --write-digest` records it and
# each run compares against it. It pins this repository's renderer.
CORPUS_RENDER_SHA = "f67859b0c66cc3459e91d66b9de5c4b7ccfd57a24ffadb59d082bdb097faab55"


def candidates(name: str) -> list[Path]:
    """Every place `name` could be, in preference order. Existence is checked by the caller."""
    return sorted(p for root in EVIDENCE_ROOTS for p in root.glob(f"**/{name}"))


def find(name: str) -> Path:
    """The first file called `name` under the evidence roots; exits non-zero if there is none."""
    hits = candidates(name)
    if not hits:
        roots = ", ".join(str(r) for r in EVIDENCE_ROOTS)
        raise SystemExit(f"{name}: not found under {roots}")
    return hits[0]


def checkpoint_dir() -> Path:
    """The directory that holds a shipped head and its adapter.

    The test is structural: a `slot_head.pt` and a sibling `lora/` that holds the adapter file.
    `bitsone.serve.load` then checks the head against the base's hidden size.
    """
    for path in candidates(HEAD_NAME):
        if (path.parent / ADAPTER_DIR / ADAPTER_NAME).is_file():
            return path.parent
    raise SystemExit(
        f"{HEAD_NAME} with a sibling {ADAPTER_DIR}/{ADAPTER_NAME}: no shipped checkpoint found "
        f"under {', '.join(str(r) for r in EVIDENCE_ROOTS)}"
    )


def base_model() -> Path:
    """The base that the adapter and head attach to, after its config declarations are checked.

    A directory that points at a differently quantised base is refused here.
    """
    import json

    weights = BASE_DIR / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"base weights not found at {weights}")
    cfg_path = BASE_DIR / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"base config not found at {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"base config at {cfg_path} is unreadable: {exc}") from exc
    # The declaration is under `quantization_config` in a real base config. The flat form is also
    # accepted, because a hand-written config may put it at the top level.
    declared = {**cfg, **cfg.get("quantization_config", {})}
    wrong = {k: (declared.get(k), v) for k, v in BASE_CONFIG_REQUIRED.items() if declared.get(k) != v}
    if wrong:
        detail = "; ".join(f"{k}={got_v!r}, want {want!r}" for k, (got_v, want) in wrong.items())
        raise SystemExit(f"base config contradicts the run: {detail}")
    return BASE_DIR


def tokenizer_dir() -> Path | None:
    """A local tokenizer for the key-token section, or None to skip that section.

    Order: the base model's tokenizer first, then a checkpoint's copy, then any other file with that
    name. A candidate counts only if its `tokenizer.json` parses as a tokenizer.
    """
    import json

    def usable(path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(data, dict) and "model" in data

    for path in [BASE_DIR / "tokenizer.json", *candidates("tokenizer.json")]:
        if path.is_file() and usable(path):
            return path.parent
    return None
