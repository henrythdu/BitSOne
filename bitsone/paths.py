"""Every filesystem fact this repository knows, in one place.

Why this module exists: the base model's path used to be written out literally in six modules, and
the shipped checkpoint was found three different ways — one literal path, one glob, one two-root
glob. Several answers to "where does the run live" made a rename a six-file edit with one missed
site, and a gate that silently checked nothing.

The rule here: one place knows where things are, and it refuses to guess. The single exception is
`CORPUS_RENDER_SHA`, which is not a provenance claim: it is the renderer's regression lock, so a
stray space in `corpus.py` fails `checks/corpus_render_check.py` instead of quietly changing what
the shipped checkpoint was trained on.

Kept stdlib-only on purpose — nothing in this module may import torch, so CI can use it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where build evidence may live. Ordered: the repo's own run tree first, then the exported copy.
EVIDENCE_ROOTS: tuple[Path, ...] = (ROOT / "runs", ROOT / "System One Models")

# Where the base weights live, and what its config must declare. The declaration is the guard:
# without it, a directory pointing at a differently quantised model trains happily and is wrong.
BASE_DIR = ROOT / "BitNet Models" / "bitnet-b1.58-2b-4t"
BASE_CONFIG_REQUIRED = {"quantization_mode": "offline", "linear_class": "autobitlinear"}

# The two files a shipped checkpoint must contain. Names, not hashes: a wrong-but-same-shaped
# head is caught structurally in `bitsone.serve.load`, which checks the head against the base.
HEAD_NAME = "slot_head.pt"
ADAPTER_DIR = "lora"
ADAPTER_NAME = "adapter_model.safetensors"

# The rendered corpus, one digest. This is the renderer's regression lock -- recorded by
# `checks/corpus_render_check.py --write-digest` and re-asserted on every run -- and the only
# digest this repository keeps. It pins OUR renderer, not anyone else's.
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
    """The directory holding a shipped head and its adapter.

    Structural, not cryptographic: a `slot_head.pt` with a sibling `lora/` that carries the adapter
    file. `bitsone.serve.load` then checks the head's shapes against the base, which is what
    actually catches a head from another run.
    """
    for path in candidates(HEAD_NAME):
        if (path.parent / ADAPTER_DIR / ADAPTER_NAME).is_file():
            return path.parent
    raise SystemExit(
        f"{HEAD_NAME} with a sibling {ADAPTER_DIR}/{ADAPTER_NAME}: no shipped checkpoint found "
        f"under {', '.join(str(r) for r in EVIDENCE_ROOTS)}"
    )


def base_model() -> Path:
    """The base the adapter and head attach to, with its config declarations asserted.

    No digest: the declaration check is what survives. A placeholder pointing at a differently
    quantised base would otherwise train happily and be wrong, and nothing downstream would say so.
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
    # The declaration lives under `quantization_config` in a real base config; accept the flat form
    # too, because a hand-written config may hoist it. Reading only the top level is what made this
    # guard reject a correct base.
    declared = {**cfg, **cfg.get("quantization_config", {})}
    wrong = {k: (declared.get(k), v) for k, v in BASE_CONFIG_REQUIRED.items() if declared.get(k) != v}
    if wrong:
        detail = "; ".join(f"{k}={got_v!r}, want {want!r}" for k, (got_v, want) in wrong.items())
        raise SystemExit(f"base config contradicts the run: {detail}")
    return BASE_DIR


def tokenizer_dir() -> Path | None:
    """A local tokenizer the key-token fixture can use; None means that section is skipped.

    Order matters: the base model's own tokenizer first (that is the one the shipped model
    tokenises with), then a checkpoint's copy, and only then whatever else carries the name —
    and a candidate counts only if its `tokenizer.json` actually parses as one. A stray file that
    happens to be called `tokenizer.json` used to be picked up here and silently disabled the
    section, which is the same class of failure as a gate that passes without reading anything.
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
