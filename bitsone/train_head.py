"""train_head — train the pointer head on cached hidden states (the staged half).

The head can be trained with the torso out of the loop: dump the hidden states once, then fit on the
cache. This module wires exactly that half:

    the six corpus files  ->  bitsone/corpus render  ->  cached hidden states  ->  train_head_only

What it does, and what it deliberately does not:

* **Does**: dumps query/key hidden states from the packed base — optionally with a shipped adapter
  attached, in which case the features are the deployed ones — then trains a fresh head on those
  cached vectors and reports train/val cross-entropy per epoch.
* **Does not** touch the torso. Nothing here has gradients outside the head: the dump runs under
  `no_grad`, and `train_head_only` optimises head parameters only. This is a STAGED trainer; the
  shipped head came from the joint path, `train_joint.py`.
* **Does not** apply the two collator augmentations (ordinal smoothing, reverse-score). They belong
  to the joint path; this trainer fits a one-hot target, and says so rather than implying otherwise.

**Gold, and why the permutation cannot desynchronise it.** Corpus rows carry a single `label` index
into the canonical option order. Rendering may permute the options, so the target is a one-hot at
`order.index(label)` in the rendered order, and `corpus.render_question` returns the slot -> label
binding where that identity is checked rather than believed. The dump stores the target ALIGNED to
its own ragged slices (the same contract `pointer.dump_rows` uses), so training cannot re-derive a
different alignment from the rows.

The head it produces is NEW — random init, trained here — and is not the shipped head. Nothing this
module computes is comparable to run 7's number, and no number of run 7's may be quoted from it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from bitsone import corpus
from bitsone.corpus import corpus_dir
from bitsone.model import load_bitnet
from bitsone.pointer import PointerHead, dense_batch, gold_ragged, pointer_soft_ce

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "BitNet Models" / "bitnet-b1.58-2b-4t"
SHIPPED = ROOT / "runs" / "run" / "out" / "fetched" / "checkpoints"


def load_base(base: Path, adapter: Path | None, tokenizer_from: Path | None = None) -> tuple[Any, Any]:
    """The packed base in `offline` mode (the artifact runs 5-7 trained), plus a tokenizer.

    With `adapter`, the shipped LoRA is attached and frozen, so the cached features are the deployed
    model's. `load_bitnet` asserts the EFFECTIVE quantization mode, because the repo's own config wins
    over the argument and a wrong mode would otherwise be silent.
    """
    from transformers import AutoTokenizer

    model = load_bitnet(str(base), "offline")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
    tok_dir = tokenizer_from or (adapter.parent if adapter else base)
    tok = AutoTokenizer.from_pretrained(str(tok_dir), fix_mistral_regex=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def one_hot_slot(row: dict[str, Any], slot_labels: tuple[str, ...], order: list[int] | None) -> int:
    """The rendered slot that carries the row's canonical label.

    The permutation moves options, so the target moves with them; the identity is asserted against
    the label the renderer reports for that slot, not assumed from the arithmetic.
    """
    label = int(row["label"])
    n = len(row["options"])
    if not 0 <= label < n:
        raise ValueError(f"label {label} out of range for {n} options")
    canonical = order.index(label) if order else label
    expect = str(corpus.pairs_of(row["kind"], row["options"])[label][0])
    if slot_labels[canonical] != expect:
        raise ValueError(
            f"permutation desynchronised the target: slot {canonical} carries "
            f"{slot_labels[canonical]!r}, the label expects {expect!r}"
        )
    return canonical


@torch.no_grad()
def dump_corpus(
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    *,
    seed: int = 0,
    batch: int = 4,
    device: str = "cuda",
    permute: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Corpus rows -> the ragged dump `train_head_only` consumes.

    Same shape as `pointer.dump_rows` (q, k, offset, ids, stored_dtype, gold, perm), different source:
    the prompt comes from `corpus.build_prompt` and the key positions from
    `corpus.key_token_indices` (the fully-inside rule).
    """
    rng = random.Random(seed)
    qs: list[torch.Tensor] = []
    ks: list[torch.Tensor] = []
    golds: list[float] = []
    offsets: list[int] = [0]
    ids: list[str] = []
    perms: list[list[int]] = []

    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        prompts, keys_per_row, slots, orders = [], [], [], []
        for i, row in enumerate(chunk):
            order = corpus.permutation(len(row["options"]), rng) if (permute and len(row["options"]) > 1) else None
            text, rendered = corpus.build_prompt(
                row["state"], row["kind"], row["instructions"], row["options"], option_order=order
            )
            prompts.append(text)
            slots.append(one_hot_slot(row, rendered.slot_labels, order))
            keys_per_row.append(rendered)
            orders.append(order if order is not None else list(range(len(row["options"]))))
        enc = tok(
            prompts,
            return_tensors="pt",  # without this the whole batch comes back as Python lists
            return_offsets_mapping=True,
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        offsets_map = enc.pop("offset_mapping")
        # Only tensors move to the device: a slow tokenizer hands `offset_mapping` back as a list of
        # pairs, and `.to()` on it is an AttributeError, not a fallback.
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        am = enc["attention_mask"]
        # NO pad_len shift here, unlike pointer.gather_batch: those offsets came from character spans and
        # were therefore unpadded-space, while these come from the tokenizer and are ALREADY padded-space
        # (measured: offsets shape == input_ids shape, real tokens in the last columns, pads offset (0,0)).
        # Adding the pad length double-counts it and points past the sequence.

        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]
        if not torch.isfinite(h[am.bool()]).all():
            raise ValueError("non-finite hidden state at a real position — the forward failed")

        for i, row in enumerate(chunk):
            rendered = keys_per_row[i]
            raw = offsets_map[i]
            offs = [(int(a), int(b)) for a, b in (raw if isinstance(raw, list) else raw.tolist())]
            # SPAN SPACE: `build_prompt` returns spans in PROMPT coordinates, which is what we
            # tokenized here, so no shift is added. Containment is asserted anyway: an unshifted
            # span does not fail loudly, it resolves to the STATE region and scores nonsense.
            spans = rendered.option_spans
            key_idx = corpus.key_token_indices(offs, spans)
            for ti, (a, b) in zip(key_idx, spans):
                ts, te = offs[ti]
                if not (ts >= a and te <= b):
                    raise ValueError(f"row {start + i}: key token {(ts, te)} is outside option span {(a, b)}")
            q_pos = int(am.shape[1]) - 1  # right edge: the prompt ends at '<answer>'
            # raises, not asserts: these guard the dump's integrity and would vanish under `python -O`
            if not bool(am[i, q_pos]):
                raise ValueError(f"row {start + i}: query position {q_pos} is padding")
            k_pos = list(key_idx)
            if not all(0 <= p < int(am.shape[1]) for p in k_pos):
                raise ValueError(f"row {start + i}: key position outside the batch width {int(am.shape[1])}")
            if not all(bool(am[i, p]) for p in k_pos):
                raise ValueError(f"row {start + i}: a key position falls on padding")

            qs.append(h[i, q_pos].cpu())
            ks.append(h[i, torch.tensor(k_pos, device=device)].cpu())
            offsets.append(offsets[-1] + len(k_pos))
            # one-hot in the RENDERED order, stored flat against this row's own ragged slice
            golds.extend(1.0 if j == slots[i] else 0.0 for j in range(len(k_pos)))
            perms.append(orders[i])
            ids.append(f"{row.get('task', '?')}:{start + i}")
        if progress and start % (batch * 200) == 0:
            print(f"    dumped {start}/{len(rows)}", flush=True)

    q, k = torch.stack(qs), torch.cat(ks)
    if q.dtype != k.dtype:
        raise ValueError(f"q is {q.dtype} but k is {k.dtype}: the dump has mixed dtypes")
    return {
        "q": q,
        "k": k,
        "offset": torch.tensor(offsets, dtype=torch.int64),
        "ids": ids,
        "stored_dtype": str(q.dtype),
        "gold": torch.tensor(golds, dtype=torch.float32),
        "perm": perms,
    }


@torch.no_grad()
def mean_ce(head: PointerHead, dump: dict[str, Any], device: str = "cuda", batch: int = 64) -> float:
    """Mean cross-entropy of the head over a dump, per row (its own options only)."""
    n = len(dump["ids"])
    total, count = 0.0, 0
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n))
        q, k, mask = dense_batch(dump, idx, device=device)
        maxk = k.shape[1]
        gold = gold_ragged(dump, idx, maxk, device=device)
        total += float(pointer_soft_ce(head(q, k, mask), gold).item()) * len(idx)
        count += len(idx)
    return total / max(count, 1)


def snapshot(params: Any, sample: int = 8) -> list[torch.Tensor]:
    """Clones of a deterministic sample of tensors, so a later check can prove what moved.

    Exact comparison rather than a fingerprint: `moved()` asks whether the bytes are equal, which is
    the question a loss curve cannot answer. Sampling keeps it cheap on a 2.4 GB torso.
    """
    named = sorted(params)
    step = max(1, len(named) // sample) if named else 1
    return [p.detach().to("cpu", torch.float32).clone() for _, p in named[::step][:sample]]


def moved(before: list[torch.Tensor], params: Any) -> bool:
    """True when the sampled tensors differ from `before` (including a changed shape or count)."""
    now = snapshot(params)
    return len(before) != len(now) or any(not torch.equal(a, b) for a, b in zip(before, now))


def train_head_only(
    dump: dict[str, Any],
    d: int = 256,
    mid: int = 0,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[PointerHead, list[float]]:
    """Train the head on CACHED vectors — no torso in the loop. Returns (head, loss curve).

    lr 1e-3 is ten times the LoRA rate, because the head is trained from scratch on frozen
    features. `mid` selects the capacity control (see PointerHead); it is the only difference
    between the two arms, which is what makes a capacity comparison interpretable.
    """
    torch.manual_seed(seed)
    n = len(dump["ids"])
    head = PointerHead(hidden=dump["k"].shape[1], d=d, mid=mid).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    curve: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(n)
        total, seen = 0.0, 0
        for i in range(0, n, 256):
            idx = order[i : i + 256]
            q, k, mask = dense_batch(dump, idx, device=device)
            gold = gold_ragged(dump, idx, k.shape[1], device=device)
            opt.zero_grad()
            loss = pointer_soft_ce(head(q, k, mask), gold)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        curve.append(total / max(seen, 1))
    return head, curve


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train the pointer head on cached features (staged, head only)."
    )
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--base-model", default=str(BASE))
    ap.add_argument("--adapter", default=None, help="LoRA dir to attach (e.g. the shipped run-7 adapter)")
    ap.add_argument("--limit", type=int, default=512, help="rows per split (0 = all)")
    ap.add_argument("--val-fraction", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-options", type=int, default=0, help="skip rows with more options than this")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="output dir for the head + report")
    ap.add_argument("--no-permute", action="store_true", help="render in canonical order (no augmentation)")
    args = ap.parse_args()
    # SEED FIRST: the head below is randomly initialised, so a seed set after its construction leaves
    # the starting point — and therefore the 'untrained' held-out CE — different on every run. Peer
    # review saw exactly that in the first two controls (1.4214 vs 1.4387); `trainer.train_head_only`
    # already seeds before constructing, which is why its own controls reproduced to 4 decimals.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = corpus_dir(args.corpus_dir)
    rows: list[dict[str, Any]] = []
    for name in corpus.CORPUS_FILES:
        rows.extend(corpus.open_jsonl(data / name))
        if args.limit and len(rows) >= args.limit * 20:
            break
    if args.max_options:
        rows = [r for r in rows if len(r["options"]) <= args.max_options]
    train_rows, val_rows = corpus.split_examples(rows, val_fraction=args.val_fraction, seed=args.seed)
    if args.limit:
        train_rows, val_rows = train_rows[: args.limit], val_rows[: max(1, args.limit // 8)]
    print(f"corpus: {data}\ntrain rows: {len(train_rows)}   val rows: {len(val_rows)}")

    model, tok = load_base(
        Path(args.base_model),
        Path(args.adapter) if args.adapter else None,
        tokenizer_from=Path(args.adapter) if args.adapter else None,
    )
    print(f"base: {args.base_model}{'  + adapter ' + args.adapter if args.adapter else '  (no adapter)'}")

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"refusing to run: {len(trainable)} torso params require grad, e.g. {trainable[0][0]}")
    torso_before = snapshot((n_p for n_p in model.named_parameters() if n_p[1].dim() >= 2))

    t0 = time.time()
    train_dump = dump_corpus(model, tok, train_rows, seed=args.seed, device=args.device, permute=not args.no_permute)
    val_dump = dump_corpus(
        model, tok, val_rows, seed=args.seed + 1, device=args.device, permute=not args.no_permute, progress=False
    )
    print(f"dumped in {time.time() - t0:.1f}s  stored_dtype={train_dump['stored_dtype']}")

    head = PointerHead(hidden=train_dump["q"].shape[1], d=256, mid=0).to(args.device)
    head_before = snapshot(head.named_parameters())
    n_head = sum(p.numel() for p in head.parameters())
    before = mean_ce(head, val_dump, device=args.device)
    t1 = time.time()
    head, curve = train_head_only(train_dump, epochs=args.epochs, lr=args.lr, seed=args.seed, device=args.device)
    after = mean_ce(head, val_dump, device=args.device)
    head_moved = moved(head_before, head.named_parameters())
    torso_moved = moved(torso_before, (n_p for n_p in model.named_parameters() if n_p[1].dim() >= 2))
    print("loss curve:", " ".join(f"{x:.4f}" for x in curve))
    print(f"val CE: untrained {before:.4f} -> trained {after:.4f}   ({time.time() - t1:.1f}s)")
    print(f"head params: {n_head:,}  moved={head_moved}   torso unchanged={not torso_moved}")
    if not head_moved:
        raise SystemExit("the head did not move: nothing was trained")
    if torso_moved:
        raise SystemExit("the torso moved: this is a staged trainer and it must not touch the torso")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), out / "head.pt")
        (out / "report.json").write_text(
            json.dumps(
                {
                    "corpus": str(data),
                    "base": args.base_model,
                    "adapter": args.adapter,
                    "rows": {"train": len(train_rows), "val": len(val_rows)},
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "seed": args.seed,
                    "stored_dtype": train_dump["stored_dtype"],
                    "head_params": n_head,
                    "head_moved": head_moved,
                    "torso_unchanged": not torso_moved,
                    "loss_curve": curve,
                    "val_ce": {"untrained": before, "trained": after},
                    "not_claimed": "a NEW head trained on cached features; not run 7's head, not joint training",
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}/head.pt and {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
