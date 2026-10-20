"""The plumbing the gates share. Not a gate: no `main`, no exit status of its own.

Why it exists: each gate resolved artifacts itself, so one question had several answers that could
drift apart. This module holds one implementation, and the gates import it.

The import direction is one-way on purpose: `checks` → `bitsone`, never the reverse. That keeps
`import bitsone` torch-free, which is what lets CI run without weights or a GPU.

Facts live in `bitsone/paths.py` (the resolution order). This module only re-exports
them under the names the gates already use, so a gate reads as "check the claim", not "know where
things are".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bitsone.paths import (  # noqa: E402
    base_model,
    checkpoint_dir,
    find,
    tokenizer_dir,
)

__all__ = [
    "base_model",
    "checkpoint_dir",
    "find",
    "tokenizer_dir",
]
