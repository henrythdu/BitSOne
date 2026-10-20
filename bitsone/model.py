"""model — load the base that the adapter and the head attach to.

`bitsone.serve.load` and both trainers call `load_bitnet`, so the base is loaded the same way in
all three places. The load checks one property: the quantisation mode in effect is the mode asked
for.
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
    """Load a BitNet repo on the CUDA device and require the effective quantization mode to be `mode`.

    The repo's own config.json decides the effective mode. The loader uses the checkpoint's
    existing quantization_config before the argument passed here, so the argument is ignored when
    the repo carries its own. This function reads the effective mode after load and before any
    forward pass, and raises when it differs from `mode`.

    Modes in this tree, read from the files:
      bitnet-b1.58-2b-4t       -> offline   (the packed artifact: what run 5 trained and what ships)
      bitnet-b1.58-2b-4t-bf16  -> online    (the bf16 masters)
    """
    cfg = BitNetQuantConfig(linear_class="autobitlinear", quantization_mode=mode)
    model: Any = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda", quantization_config=cfg
    )
    # `config.quantization_config` is a BitNetQuantConfig object, not a dict. Subscripting it raises
    # TypeError, which would report the wrong cause.
    effective = model.config.quantization_config.quantization_mode
    # Raise, do not assert: `python -O` removes an assert, and this is the only check that the mode
    # in effect is the requested mode.
    if effective != mode:
        raise RuntimeError(
            f"mode mismatch: asked for {mode!r} but the loaded model is {effective!r}. "
            f"Repo {Path(repo).name!r} carries its own quantization_config and the loader prefers it "
            f"over the argument, so the request was IGNORED. A CALL SITE MUST NOT PASS A MODE ITS "
            f"REPO DOES NOT CARRY."
        )
    return model
