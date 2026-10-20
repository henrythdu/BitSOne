"""serve — one entry point: render the request, read the option keys, score them.

The three steps must agree. `Engine.answer` runs all three in one call, so a caller cannot bind an
option to the wrong token.

  1. `bitsone.corpus.build_prompt` renders the request. It is deterministic.
  2. `bitsone.corpus.key_token_indices` selects, for each option, the last token fully inside that
     option's span.
  3. `bitsone.pointer.PointerHead` scores each option's hidden state against the query position in
     fp32. Masked slots are -inf.

The torso comes from `transformers` and `peft`. All code above the torso is ours.

`bitsone.paths.checkpoint_dir` finds the checkpoint by its structure. The loader reads `lora/`,
`slot_head.pt` and the tokenizer. It does not read the run config file, whose name belongs to the
published artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitsone import corpus

HIDDEN = 2560


@dataclass
class Response:
    """What one request produced: the scored options and the bytes they were scored from."""

    text: str
    slot_labels: tuple[str, ...]
    key_indices: list[int]
    logits: list[float]
    log_probs: list[float]

    @property
    def best(self) -> str:
        """The option label with the highest score."""
        best = max(range(len(self.logits)), key=self.logits.__getitem__)
        return self.slot_labels[best]

    @property
    def probs(self) -> list[float]:
        import math

        return [math.exp(lp) for lp in self.log_probs]


class Engine:
    """A loaded served model: packed base + adapter + pointer head + tokenizer."""

    def __init__(self, tokenizer: Any, model: Any, head: Any, device: str = "cuda") -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.head = head
        self.device = device
        self.torso = _torso(model)

    def render(
        self, state: Any, kind: str, instructions: Any, options: list[list[str]], option_order: list[int] | None = None
    ) -> tuple[str, corpus.Rendered]:
        """The prompt text and the binding to the options."""
        return corpus.build_prompt(state, kind, instructions, options, option_order)

    def _inputs(self, text: str) -> tuple[Any, list[tuple[int, int]]]:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        return {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}, offsets

    def answer(
        self, state: Any, kind: str, instructions: Any, options: list[list[str]], option_order: list[int] | None = None
    ) -> Response:
        """Score every option for one request. Raises when a key position cannot be found."""
        import torch

        text, rendered = self.render(state, kind, instructions, options, option_order)
        inputs, offsets = self._inputs(text)
        keys = corpus.key_token_indices(offsets, rendered.option_spans)
        query = int(inputs["input_ids"].shape[1]) - 1

        with torch.no_grad():
            out = self.torso(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1] if out.hidden_states is not None else out.last_hidden_state
            hq = hidden[0, query].unsqueeze(0)
            hk = hidden[0, keys].unsqueeze(0)
            mask = torch.ones(1, len(keys), dtype=torch.bool, device=hq.device)
            logits = self.head(hq, hk, mask)[0]

        finite = torch.isfinite(logits)
        log_probs = torch.log_softmax(logits[finite].float(), dim=-1)
        return Response(
            text=text,
            slot_labels=rendered.slot_labels,
            key_indices=keys,
            logits=[float(v) for v in logits[finite].tolist()],
            log_probs=[float(v) for v in log_probs.tolist()],
        )




def _torso(model: Any) -> Any:
    """The decoder stack without the vocabulary projection.

    A full forward computes `lm_head` (128,256 x 2,560) for every prompt token, and this module never
    reads that output. With the shipped checkpoint, hidden states and head logits are identical with
    and without it, and one request is about 3% faster.

    The stack is found by attribute, so the plain model and the adapter-wrapped model both work. If
    no stack is found, the full model is returned and only the projection is wasted.
    """
    inner = getattr(model, "base_model", None)  # PeftModel -> LoraModel
    inner = getattr(inner, "model", None) or getattr(model, "model", None)
    stack = getattr(inner, "model", None)
    return stack if stack is not None and hasattr(stack, "layers") else model


def load_adapter_weights(model: Any, path: Path) -> None:
    """Load the adapter's tensors into `model`, and FAIL if any of them cannot be placed.

    The published adapter names its tensors `base_model.model.layers.N...lora_A.weight`, while PEFT
    expects `base_model.model.model.layers.N...lora_A.default.weight`. Handed straight to
    `PeftModel.from_pretrained`, that mismatch is a WARNING: LoRA's B matrices keep their zero
    initialisation and the model silently serves the bare base with an adapter-trained head. That is
    how the first evaluation of this checkpoint produced chance accuracy and saturated logits.

    So: translate each key to the name the model actually has, and refuse to return while any B
    matrix is still zero -- a zero B means the adapter contributes nothing, whatever the file says.
    """
    from safetensors.torch import load_file

    if not path.is_file():
        raise SystemExit(f"{path}: no adapter weights")
    state = load_file(str(path))
    expected = {name for name, _ in model.named_parameters() if "lora_" in name}

    def candidates(key: str) -> list[str]:
        out = [key]
        for base in list(out):
            if base.startswith("base_model.model.") and not base.startswith("base_model.model.model."):
                out.append(base.replace("base_model.model.", "base_model.model.model.", 1))
        out += [k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight") for k in list(out)]
        return out

    placed, unmatched = {}, []
    for key, tensor in state.items():
        match = next((c for c in candidates(key) if c in expected), None)
        if match is None:
            unmatched.append(key)
        else:
            placed[match] = tensor
    if unmatched:
        raise SystemExit(
            f"{path}: {len(unmatched)} adapter tensor(s) do not fit this model, e.g. {unmatched[0]}.\n"
            "  Refusing to serve: a partly placed adapter is indistinguishable from no adapter."
        )
    # COVERAGE IN THE OTHER DIRECTION. Unmatched tensors mean the file carries things this model
    # cannot use; a parameter with no tensor means PEFT's own initialisation survives -- a random A or
    # a zero B that the file never overwrote. The zero-B check below catches the second; nothing caught
    # a missing A, which leaves random features feeding a trained projection.
    missing = sorted(expected - set(placed))
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} adapter parameter(s) have no tensor in this file, e.g. {missing[0]}.\n"
            "  Refusing to serve: those keep PEFT's initialisation, not the trained values."
        )

    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, placed)
    dead = [n for n, p in model.named_parameters() if "lora_B" in n and float(p.float().abs().sum()) == 0.0]
    if dead:
        raise SystemExit(
            f"{path}: {len(dead)} LoRA B matrices are zero after loading, e.g. {dead[0]}.\n"
            "  A zero B is a no-op adapter: the served model would be the bare base."
        )


def load(
    checkpoint: Path | None = None,
    base: Path | None = None,
    device: str = "cuda",
    dtype: Any = None,
) -> Engine:
    """Load the shipped model: adapter + head on the packed base.

    Both paths default through `bitsone.paths`, which locates a checkpoint structurally (a
    `slot_head.pt` with a sibling `lora/`). Identity is then checked where it can be: the head is
    built for the base's hidden size, so a head from a differently shaped run is refused here
    rather than served silently. Two runs sharing a shape cannot be told apart by shape alone,
    which this repository no longer keeps.
    """
    import torch
    from transformers import AutoTokenizer

    from bitsone.model import load_bitnet
    from bitsone.paths import base_model as _base_model
    from bitsone.paths import checkpoint_dir
    from bitsone.pointer import PointerHead

    ckpt = Path(checkpoint) if checkpoint else checkpoint_dir()
    base_path = Path(base) if base else _base_model()
    model = load_bitnet(str(base_path), "offline")
    # The head's width is read from the BASE config, before PEFT wraps the model: after wrapping,
    # `model.config` is the LoRA config, which has no `hidden_size`, and a default there would build a
    # 2560-wide head for any base and turn a base/head mismatch into a shape error at load_state_dict.
    base_hidden = int(getattr(model.config, "hidden_size", 0))
    if base_hidden <= 0:
        raise SystemExit(f"{base_path}: config carries no hidden_size, so the head's width cannot be checked")
    # Build an EMPTY adapter shell, then place every tensor ourselves: `PeftModel.from_pretrained`
    # warns about a name mismatch and then serves the bare base, which is the failure this loader
    # exists to make impossible. One load path, and it fails closed.
    from peft import LoraConfig, get_peft_model

    adapter_cfg = LoraConfig.from_pretrained(str(ckpt / "lora"))
    model = get_peft_model(model, adapter_cfg)
    load_adapter_weights(model, ckpt / "lora" / "adapter_model.safetensors")
    if device != "cuda":
        # The loader places the model on CUDA. Another device is a move after load.
        model = model.to(device)
    model.eval()

    tok_dir = ckpt if (ckpt / "tokenizer.json").is_file() else base_path
    tok = AutoTokenizer.from_pretrained(str(tok_dir), fix_mistral_regex=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hidden = base_hidden
    head = PointerHead(hidden=hidden)
    state = torch.load(ckpt / "slot_head.pt", map_location=device, weights_only=True)
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"{ckpt}/slot_head.pt does not fit this base ({exc})") from exc
    head = head.to(device).float().eval()
    if dtype is not None:
        model = model.to(dtype)
    return Engine(tok, model, head, device)
