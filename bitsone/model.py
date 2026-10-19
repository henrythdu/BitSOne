"""model — load the base the adapter and head attach to.

One function, because that is all the live path needs: `bitsone.serve.load` and both trainers load
the same base the same way, and the one thing worth asserting at load time is that the quantisation
mode in effect is the one asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

try:  # transformers ships the BitNet config under different paths across versions
    from transformers import BitNetQuantConfig
except ImportError:  # pragma: no cover - only on older transformers
    from transformers.utils.quantization_config import BitNetQuantConfig  # type: ignore[no-redef]


def load_bitnet(repo: str, mode: str) -> Any:
    """Load a BitNet repo and REQUIRE the effective quantization mode to be `mode`.

    The effective mode is decided by the REPO's own config.json, because the loader prefers a
    checkpoint's existing quantization_config over the argument passed here. So a call site that
    passes a mode its repo does not carry is SILENTLY IGNORED: the argument looks like it selects
    the mode and does not. Asserting on the EFFECTIVE mode, read after load and before any forward,
    is what makes the wrong version impossible rather than merely unlikely.

    Repo modes in this tree, measured from the bytes:
      bitnet-b1.58-2b-4t       -> offline   (the packed artifact: what run 5 trained and what ships)
      bitnet-b1.58-2b-4t-bf16  -> online    (the bf16 masters)
    """
    cfg = BitNetQuantConfig(linear_class="autobitlinear", quantization_mode=mode)
    model: Any = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda", quantization_config=cfg
    )
    # NOTE: config.quantization_config is a BitNetQuantConfig OBJECT, not a dict — subscripting it
    # raises TypeError, which would make the guard fail for the WRONG reason and look like it fired.
    effective = model.config.quantization_config.quantization_mode
    assert effective == mode, (
        f"mode mismatch: asked for {mode!r} but the loaded model is {effective!r}. "
        f"Repo {Path(repo).name!r} carries its own quantization_config and the loader prefers it "
        f"over the argument, so the request was IGNORED. A CALL SITE MUST NOT PASS A MODE ITS "
        f"REPO DOES NOT CARRY."
    )
    return model
