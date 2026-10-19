"""train_joint — train LoRA and the pointer head together, in one loop.

`train_head.py` trains the head on frozen cached features. Here the torso adapter and the head train
together, on the packed offline base:

    the six corpus files -> renderer (pinned by `checks/corpus_render_check.py`)
                         -> LoRA r16/a32 on the seven target modules, + pointer head
                         -> pointer soft-CE, plus optional teacher KL

What this file is not: run 7's arm, and not comparable to 0.60475. Run 7 ran outside this tree; this
is a different implementation of a similar shape, and the head it produces is new. No run-7 number
may be quoted from anything this file prints.

Two details make a joint trainer work and both cost real debugging to find, so they are stated where
they happen: the CHARACTER span shift (option spans index the question, and the prompt is state +
question, so spans shift by the state prefix), and the fact that token positions are ALREADY in
padded space, so they take no pad-length shift.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from bitsone import corpus
from bitsone.corpus import corpus_dir
from bitsone.model import load_bitnet
from bitsone.pointer import PointerHead, pointer_soft_ce
from bitsone.train_head import moved, one_hot_slot, snapshot

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "BitNet Models" / "bitnet-b1.58-2b-4t"

# Seven targets, measured on the packed base's own safetensors header: 30 layers,
# `self_attn.q/k/v/o_proj` and `mlp.gate/up/down_proj` at 60 tensors each (A and B), and
# `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj` at ZERO. Those five exist only on a
# hybrid-attention base, where linear-attention layers add them; this base has no recurrent layers
# at all, so the seven reach all thirty and the five would hit nothing.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_joint_model(base: Path, device: str, grad_ckpt: bool = True) -> Any:
    """Packed OFFLINE base (trits frozen) + trainable LoRA on the seven targets.

    `load_bitnet(..., "offline")` asserts the EFFECTIVE mode, because the repo's own config.json wins
    over the argument and a silently wrong mode would train something else. The base stays frozen:
    only adapters and the head receive gradients.
    """
    from peft import LoraConfig, get_peft_model

    model = load_bitnet(str(base), "offline")
    # A wrong base must be LOUD. PEFT either raises on an unmatched target or silently widens the
    # config, and both outcomes hide the real mistake: someone pointed this at a hybrid-attention
    # base, where the seven names reach only the full-attention layers.
    present = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    missing = [t for t in LORA_TARGETS if t not in present]
    if missing:
        raise ValueError(
            f"{base.name}: base has no {missing}. {LORA_TARGETS} is the seven-target set measured on\n"
            "the packed BitNet b1.58 2B-4T base (30 full-attention layers). A recipe written for a\n"
            "hybrid-attention base lists twelve, adding in_proj_qkv / in_proj_z / in_proj_a /\n"
            "in_proj_b / out_proj for its linear-attention layers; that list is not this one."
        )
    # PRESENCE IS NOT COVERAGE (peer review). A hybrid-attention base has all seven names too — on its
    # full-attention layers only — so a presence check passes while LoRA silently adapts part of the
    # torso. Counting matched modules against the layer count is what makes the refusal real: this
    # base has 30 of each; a hybrid base with 24 layers and 6 full-attention ones would have 6.
    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    matched = {t: sum(1 for n, _ in model.named_modules() if n.rsplit(".", 1)[-1] == t) for t in LORA_TARGETS}
    if n_layers and any(count < n_layers for count in matched.values()):
        raise ValueError(
            f"{base.name}: target coverage {matched} is short of {n_layers} layers. These seven names\n"
            "exist on every layer of a dense-attention base (BitNet b1.58 2B-4T: 30 of each). On a\n"
            "hybrid-attention base (24 layers, 6 of them full attention) they cover only part of the\n"
            "torso, and the twelve-name list with in_proj_qkv / in_proj_z /\n"
            "in_proj_a / in_proj_b / out_proj is the one that base needs."
        )
    for p in model.parameters():
        p.requires_grad_(False)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # PEFT + checkpointing compatibility
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    return model


def collate(
    rows: list[dict[str, Any]],
    tok: Any,
    device: str,
    *,
    rng: random.Random,
    permute: bool = True,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Corpus rows -> one training batch, with the option permutation drawn per row.

    Right padding: `pool_last_token` finds the last real token by mask
    length and infer.py's shared-prefix cache assumes the prompt starts at column 0.
    """
    prompts, spans_per_row, slots, orders = [], [], [], []
    for row in rows:
        order = corpus.permutation(len(row["options"]), rng) if (permute and len(row["options"]) > 1) else None
        text, rendered = corpus.build_prompt(
            row["state"], row["kind"], row["instructions"], row["options"], option_order=order
        )
        prompts.append(text)
        slots.append(one_hot_slot(row, rendered.slot_labels, order))
        orders.append(order if order is not None else list(range(len(row["options"]))))
        spans_per_row.append(rendered.option_spans)  # already in prompt coordinates

    enc = tok(
        prompts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        padding_side="right",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    offsets_map = enc.pop("offset_mapping")
    input_ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    # A truncated prompt can lose whole options, and a key position then falls outside the real
    # tokens — the head would score a neighbour's hidden state, which is silently wrong. Refuse
    # instead of training through it.
    # EXACT FIT IS NOT TRUNCATION. `>= max_length` raised on a prompt that was exactly `max_length`
    # tokens and had lost nothing, because the guard could not tell "hit the cap" from "fit exactly".
    # Only a prompt whose UNTRUNCATED length exceeds the cap actually lost tokens, so at-cap prompts are
    # re-tokenized without truncation (they are rare) and only real truncation refuses (peer review).
    at_cap = (am.sum(dim=1) >= max_length).nonzero().flatten().tolist()
    if at_cap:
        cut = [i for i in at_cap if len(tok(prompts[i], add_special_tokens=False)["input_ids"]) > max_length]
        if cut:
            raise ValueError(
                f"{len(cut)} prompt(s) exceed max_length={max_length} and would be truncated, possibly "
                f"losing whole options (batch row indices {cut}). Raise --max-length."
            )
    width = max(len(o) for o in orders)
    opt_idx = torch.full((len(rows), width), -1, dtype=torch.long, device=device)
    gold = torch.zeros((len(rows), width), dtype=torch.float32, device=device)

    for i in range(len(rows)):
        raw = offsets_map[i]
        offs = [(int(a), int(b)) for a, b in (raw if isinstance(raw, list) else raw.tolist())]
        keys = corpus.key_token_indices(offs, spans_per_row[i])
        # Token positions are ALREADY padded-space (offsets are aligned with input_ids), so no
        # pad-length shift is added here — unlike pointer.gather_batch, whose positions came from
        # character spans. Asserted against the mask rather than assumed.
        for j, p in enumerate(keys):
            if not (0 <= p < int(am.shape[1])):
                raise ValueError(f"row {i}: key position {p} outside batch width {int(am.shape[1])}")
            if not bool(am[i, p]):
                raise ValueError(f"row {i}: key position {p} falls on padding")
            opt_idx[i, j] = p
        gold[i, slots[i]] = 1.0

    return {"input_ids": input_ids, "attention_mask": am, "opt_idx": opt_idx, "gold": gold, "orders": orders}


def gather_head_inputs(
    hidden: torch.Tensor, am: torch.Tensor, opt_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Final-layer hidden states -> (q [B,H], k [B,K,H], mask [B,K]) for the pointer head.

    `q` is the LAST REAL token (right padding: index = mask length - 1), the token that has just read
    the whole question and the `<answer>` slot. `k` is each option's own key token, gathered from the
    same forward — that is what makes this joint: the adapters see the head's gradient.
    """
    b = hidden.shape[0]
    q_pos = am.sum(dim=1) - 1
    if bool((q_pos < 0).any()):
        raise ValueError("an empty sequence reached the head")
    q = hidden[torch.arange(b, device=hidden.device), q_pos]
    mask = opt_idx >= 0
    safe = opt_idx.clamp_min(0)
    k = hidden[torch.arange(b, device=hidden.device).unsqueeze(1), safe]
    return q, k, mask


def _autocast(model: Any) -> Any:
    """bf16 autocast on the device the model actually lives on.

    Autocast is required, not cosmetic: the base runs bf16 while PEFT keeps the adapters in fp32, and
    a plain forward meets them in a matmul whose dtypes disagree.

    The device type is DERIVED rather than hard-coded. `device_type="cuda"` made `--device cpu` die
    inside autocast with a torch error instead of behaving like the flag says — the flag was threaded
    through everything else, so it has to be honest here too (peer review).
    """
    return torch.autocast(device_type=str(next(model.parameters()).device.type), dtype=torch.bfloat16)


def batch_loss(
    model: Any,
    head: PointerHead,
    batch: dict[str, Any],
    teacher: torch.Tensor | None,
    teacher_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One joint step's loss: pointer soft-CE against the row's gold, plus optional teacher KL.

    Autocast is required, not cosmetic: the base runs bf16 while PEFT keeps the adapters in fp32, and
    a plain forward meets them in a matmul whose dtypes disagree.
    """
    with _autocast(model):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    q, k, mask = gather_head_inputs(out.hidden_states[-1].float(), batch["attention_mask"], batch["opt_idx"])
    logits = head(q, k, mask)
    loss = pointer_soft_ce(logits, batch["gold"])
    parts = {"gold_ce": float(loss.item())}
    # `logits_absmax` is printed because a divergent joint run is otherwise unreadable: the curve is a
    # SUM, so without the components one cannot tell a blown-up softmax from a bad target.
    parts["logits_absmax"] = float(logits.abs().max().item())
    if teacher is not None and teacher_weight > 0:
        # A row the teacher never scored carries a ZERO row, not a uniform one: a uniform target would
        # pull the head toward flatness, which is a real (if small) bias rather than a missing value.
        # Unscored rows are skipped, not filled with a uniform target: uniformity would pull the head
        # toward flatness, which is a bias rather than a missing value.
        covered = teacher.sum(dim=1) > 0
        if bool(covered.any()):
            tloss = pointer_soft_ce(logits[covered], teacher[covered])
            loss = loss + teacher_weight * tloss
            parts["teacher_ce"] = float(tloss.item())
            parts["teacher_rows"] = float(int(covered.sum()))
    return loss, parts


@torch.no_grad()
def mean_ce(
    model: Any, head: PointerHead, rows: list[dict[str, Any]], tok: Any, device: str, *, batch: int = 8
) -> float:
    """Held-out mean soft-CE, over the LIVE model (features are not cached in the joint path)."""
    rng = random.Random(0)
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(rows), batch):
        b = collate(rows[i : i + batch], tok, device, rng=rng, permute=False)
        with _autocast(model):
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        q, k, mask = gather_head_inputs(out.hidden_states[-1].float(), b["attention_mask"], b["opt_idx"])
        logits = head(q, k, mask)
        total += float(pointer_soft_ce(logits, b["gold"]).item()) * len(b["input_ids"])
        count += len(b["input_ids"])
    model.train()
    return total / max(count, 1)


def lora_grad_report(model: Any) -> dict[str, float | int]:
    """Grad observables: a joint run is only joint if the adapters actually received gradient."""
    grads = [(n, p.grad) for n, p in model.named_parameters() if "lora_" in n]
    with_grad = [(n, g) for n, g in grads if g is not None]
    nonzero = [(n, g) for n, g in with_grad if bool((g != 0).any())]
    norms = [float(g.norm()) for _, g in with_grad]
    return {
        "lora_tensors": len(grads),
        "lora_tensors_with_grad": len(with_grad),
        "lora_tensors_nonzero": len(nonzero),
        "lora_max_grad_norm": max(norms) if norms else 0.0,
    }


def grad_norm(params: list[tuple[str, Any]]) -> float:
    """L2 norm of a parameter group's gradients, accumulated in a tensor (empty group -> 0.0)."""
    total = torch.zeros((), dtype=torch.float32)
    for _, p in params:
        if p.grad is not None:
            total = total + (p.grad.detach().float() ** 2).sum()
    return float(total.sqrt().item())


def load_teacher(path: Path, wanted: set[int]) -> dict[int, list[float]]:
    """Read the teacher file for the rows we will use. Format: {"i": <global row index>, "probs": [...]}.

    `i` indexes the 123,339-row CONCATENATION of the six files in `CORPUS_FILES` order, which is why
    the global index travels with each row instead of being re-derived from position in a split.
    Parsing goes through `corpus.parse_row`, so a malformed line names its file and line.
    """
    out: dict[int, list[float]] = {}
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = corpus.parse_row(line, path, lineno)
            try:
                i = int(rec["i"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: teacher record has no usable index: {rec.get('i')!r}") from exc
            if i in wanted:
                # The teacher file is THIRD-PARTY input: a malformed entry must name the record it
                # came from, not surface as a bare ValueError somewhere in the training loop.
                try:
                    out[i] = [float(x) for x in rec["probs"]]
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: teacher record {i} has unusable probs: {exc}") from exc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train LoRA + the pointer head jointly on the corpus (not run 7's arm).")
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--base-model", default=str(BASE))
    ap.add_argument("--limit", type=int, default=1200, help="global rows to read, in file order")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA lr")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="randomly-initialised head: bigger step")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--teacher", default=None, help="teacher jsonl (rows keyed by global index)")
    ap.add_argument("--teacher-weight", type=float, default=1.0)
    ap.add_argument("--no-permute", action="store_true", help="never permute options (no augmentation)")
    ap.add_argument("--warmup", type=int, default=-1, help="warmup steps (-1 = max(5, 3 percent of steps))")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument(
        "--eval-batch",
        type=int,
        default=4,
        help="held-out eval batch; the eval phase is the memory peak (one 200-step run was killed there)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help=(
            "default cuda; the cpu path is mechanism-verified (autocast derives its device type) but has "
            "not been run end-to-end -- a 2.4B bf16 forward is minutes per batch on CPU"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=None,
        help="save adapter + head + report here (a run that saves nothing cannot be re-evaluated)",
    )
    args = ap.parse_args()

    # SEED FIRST — before anything that draws random numbers. `build_joint_model` initialises the LoRA
    # adapters and `PointerHead(...)` below is randomly initialised too, so a seed placed after those
    # calls cannot make either reproducible; it would only fix the sampler and dropout from then on.
    # Peer review measured the principle directly: with the seed after construction the two head
    # fingerprints differed (`afa3010079de` vs `ce259825b5db`); with it before, they matched. Our
    # `trainer.train_head_only` already seeds first, which is why those controls reproduce exactly.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    mode = os.environ.get("BITSONE_DETERMINISTIC", "").strip()
    # `0`, `false` and `off` must mean OFF. As a truthiness test this ENABLED the switch for all three and
    # then recorded `determinism=0` in `report.json` — a report contradicting its own run, the same class
    # of defect as a mis-recorded `weight_decay` (peer review). Only `1` and `strict` switch it on; an
    # unknown value warns and stays off rather than silently guessing.
    enabled = mode in ("1", "strict")
    if mode and not enabled:
        print(f"warning: BITSONE_DETERMINISTIC={mode!r} not understood: determinism stays OFF (use 1 or strict)")
    if enabled:
        # `1` warns, `strict` raises. warn_only=True does NOT enforce determinism: ops with no
        # deterministic implementation still run and merely warn — measured, a pair with warn_only=True
        # still produced different LoRA fingerprints (`ceda299142db` vs `40e0910afe5a`). `strict` is the form
        # that guarantees the property by refusing to run instead of hoping.
        # CUBLAS_WORKSPACE_CONFIG has to be set before cuBLAS initialises; this is the first statement of
        # `main`, before any model work, so it is effective here. Do not move it below a CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=(mode != "strict"))
        # Measured: with THESE switches (math SDPA) two runs produced identical LoRA and head fingerprints
        # (`6d8c65692718` / `3c445a2922a3`), while a default-kernel pair taken under GPU contention
        # differed in the LoRA fingerprint only. So the bit-exactness rests on the SDPA switches, not on the
        # call above.
        # CAUSE NOT ATTRIBUTED, and the evidence is now specific. This configuration (200 rows / 2
        # steps) differs run to run with DEFAULT kernel selection — four instances, on an idle card as
        # well as under contention (`8295a2f84390`/`c1ac36059896`, then `46b81583d018`/`888a38e1db29`),
        # always in the LoRA fingerprint only, with start fingerprints, head fingerprint and held-out CE identical.
        # At 2000 rows / 20 steps the same code was bit-identical across four env-unset runs — TWO with
        # default kernel selection and TWO with flash/mem-efficient FORCED (peer measurement,
        # `a651b8f422dd`). So the difference
        # tracks the CONFIGURATION — shapes and batch composition choose the backend — and not the load
        # on the card, and 'atomics in the flash backward' is not established by it. What IS established:
        # forcing the math kernel makes this configuration bit-exact. Hence a conservative measure for
        # comparisons, not a demonstrated fix.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"deterministic mode: {mode} (math SDPA forced; warn_only={mode != 'strict'})")

    data = corpus_dir(args.corpus_dir)
    rows: list[dict[str, Any]] = []
    for name in corpus.CORPUS_FILES:
        for row in corpus.open_jsonl(data / name):
            row["_global_i"] = len(rows)
            rows.append(row)
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    if len(rows) < args.limit:
        raise SystemExit(f"only {len(rows)} rows available, wanted {args.limit}")
    # Hold out every 10th row in file order: teacher indices stay valid because the global index
    # travels with the row, not with its position in a shuffled split.
    held = rows[::10]
    train_rows = [r for i, r in enumerate(rows) if i % 10 != 0]
    print(f"corpus: {data}\nrows: {len(train_rows)} train / {len(held)} held out (global indices preserved)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(Path(args.base_model)), fix_mistral_regex=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    teacher_rows: dict[int, list[float]] = {}
    if args.teacher:
        want = {int(r["_global_i"]) for r in rows}
        teacher_rows = load_teacher(Path(args.teacher), want)
        print(f"teacher: {len(teacher_rows)} of {len(rows)} rows carry a distribution")

    model = build_joint_model(Path(args.base_model), args.device, grad_ckpt=not args.no_grad_ckpt)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters (LoRA only, head added separately): {n_train:,}")
    lora_before = snapshot([(n, p) for n, p in model.named_parameters() if "lora_" in n])

    # The head's own shape is recorded in the report below: `head.pt` is a bare state_dict, so a
    # re-measure needs hidden/d/mid, and `mid` is the H4e capacity control (mid=0 here). Peer review.
    head_d, head_mid = 256, 0
    head = PointerHead(hidden=model.config.hidden_size, d=head_d, mid=head_mid).to(args.device)
    head_before = snapshot(head.named_parameters())
    torso_before = snapshot([(n, p) for n, p in model.named_parameters() if "lora_" not in n and p.dim() >= 2])
    n_head = sum(p.numel() for p in head.parameters())
    print(f"head parameters: {n_head:,}")
    # Printed because a reproducibility claim needs an observable: two runs of the same command must
    # show these two lines identical, or the two runs' later numbers are not comparable.
    print("start fingerprints taken: head and LoRA (for the moved checks below)")

    head_ce = mean_ce(model, head, held, tok, args.device, batch=args.eval_batch)
    print(f"held-out CE before training: {head_ce:.4f}")

    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad], "lr": args.lr},
            {"params": list(head.parameters()), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    rng = random.Random(args.seed)
    base_lrs = [g["lr"] for g in opt.param_groups]
    det_label = mode if enabled else "off"
    warmup = args.warmup if args.warmup >= 0 else max(5, int(0.03 * args.steps))
    print(f"warmup: {warmup} steps, then constant lr (LoRA {args.lr}, head {args.head_lr})")
    # PROVENANCE INCLUDING THE EVAL BATCH. A pair differing only in this number is not comparable:
    # the same weights gave held-out CE 1.3917 at eval batch 8 and 1.3908 at eval batch 4 (peer
    # measurement, identical start fingerprints, before any optimizer step). Print it, so a reader never
    # has to infer it from the command line.
    print(
        f"provenance: rows={len(rows)} train={len(train_rows)} held={len(held)} seed={args.seed} "
        f"steps={args.steps} micro={args.micro_batch} accum={args.grad_accum} warmup={warmup} "
        f"eval_batch={args.eval_batch} grad_ckpt={not args.no_grad_ckpt} determinism={det_label} "
        f"teacher={Path(args.teacher).name if args.teacher else 'none'} w={args.teacher_weight}"
    )
    model.train()
    t0 = time.time()
    history: list[float] = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        parts: dict[str, float] = {}
        for _ in range(args.grad_accum):
            batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.micro_batch)]
            batch = collate(
                batch_rows, tok, args.device, rng=rng, permute=not args.no_permute, max_length=args.max_length
            )
            teacher = None
            if teacher_rows:
                K = batch["gold"].shape[1]
                dist = []
                for j, row in enumerate(batch_rows):
                    p = teacher_rows.get(int(row["_global_i"]))
                    order = batch["orders"][j]
                    if p is None:
                        dist.append([0.0] * K)  # uncovered: skipped by the loss, not flattened
                    else:
                        perm = [p[i] for i in order]
                        s = sum(perm) or 1.0
                        # PAD TO THE BATCH WIDTH. A batch can mix rows with different option counts —
                        # the corpus opens with a K=4 block and the first K=15 row is index 6000 — and a
                        # ragged list of lists is not a tensor: `torch.tensor(dist)` raises "expected
                        # sequence of length 4 at dim 1 (got 15)" instead of training, which reads as a
                        # torch error rather than a data-shape one. Zero padding is safe:
                        # `pointer_soft_ce` slices each row to its own leading k, and the covered-mask
                        # uses the row SUM, which padding at the end does not change. Found by peer
                        # review, reproduced without a GPU on rows 0 and 6000.
                        # Pad AFTER normalising, never before: dividing a padded row by its own sum would
                        # dilute the distribution and break the covered-mask, which is the only place
                        # trailing zeros could matter (peer review, measured inert as written).
                        dist.append([x / s for x in perm] + [0.0] * (K - len(perm)))
                teacher = torch.tensor(dist, dtype=torch.float32, device=args.device)
            loss, parts = batch_loss(model, head, batch, teacher, args.teacher_weight)
            (loss / args.grad_accum).backward()
            total += float(loss.item()) / args.grad_accum
        # PRE-clip norms, computed BEFORE the clip calls below — otherwise both numbers are capped at
        # 1.0 and the instrument cannot tell 'the clip saved it' from 'the clip did nothing'.
        norms = {
            "gn_lora": grad_norm([(n, p) for n, p in model.named_parameters() if "lora_" in n]),
            "gn_head": grad_norm(list(head.named_parameters())),
        }
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.0)
        # A randomly initialised head on top of a pretrained torso is the one place where the full
        # step is too large from the first batch, so a 3 percent warmup is used. Without it the first
        # measured joint run diverged — 2.69, 12.7, 26.4, 41.3 — and
        # never came back, so this is a defect fix with a number behind it, not a precaution.
        scale = min(1.0, (step + 1) / max(1, warmup))
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * scale
        opt.step()
        history.append(total)
        if step % 5 == 0 or step == args.steps - 1:
            extra = "  " + " ".join(f"{k}={v:.4f}" for k, v in {**parts, **norms}.items())
            print(
                f"  step {step:3d}  loss {total:.4f}  lr={opt.param_groups[0]['lr']:.2e}{extra}  ({time.time() - t0:.1f}s)",
                flush=True,
            )

    head_moved = moved(head_before, head.named_parameters())
    lora_moved = moved(lora_before, [(n, p) for n, p in model.named_parameters() if "lora_" in n])
    torso_moved = moved(
        torso_before, [(n, p) for n, p in model.named_parameters() if "lora_" not in n and p.dim() >= 2]
    )
    head_ce_after = mean_ce(model, head, held, tok, args.device, batch=args.eval_batch)
    grads = lora_grad_report(model)
    print(f"loss curve: {' '.join(f'{x:.4f}' for x in history[:: max(1, len(history) // 8)])}")
    print(f"held-out CE: {head_ce:.4f} -> {head_ce_after:.4f}")
    print(f"head moved: {head_moved}   lora moved: {lora_moved}   torso unchanged: {not torso_moved}")
    print(f"LoRA grads: {grads}")
    # THE REFUSALS COME FIRST (peer review). Artifacts written before these checks would certify a run
    # that failed its own integrity tests, and the failure would be visible only in the exit code while
    # `lora/`, `head.pt` and `report.json` sat there looking re-measurable.
    if not head_moved:
        raise SystemExit("the head did not move: nothing was trained")
    if not lora_moved:
        raise SystemExit("the adapter did not move: this was not joint training")
    if torso_moved:
        raise SystemExit("a non-adapter torso tensor moved: this must be adapter-only training")
    if int(grads["lora_tensors_nonzero"]) == 0:
        raise SystemExit("no LoRA tensor received non-zero gradient: this was not joint training")
    if args.out:
        # SAVE THE ARTIFACTS. Without this a completed joint run leaves nothing but prints, so its
        # held-out number can never be re-measured under a different eval batch, and a comparison
        # confounded by that batch cannot be repaired afterwards without retraining. That is exactly
        # how the 200-step pair ended up with one arm at eval batch 8 and the other at 4.
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # PeftModel: writes adapter_config.json + adapter_model.safetensors, which is all a re-measure needs.
        model.save_pretrained(str(out / "lora"))
        torch.save(head.state_dict(), out / "head.pt")
        (out / "report.json").write_text(
            json.dumps(
                {
                    "corpus": str(data),
                    "base": args.base_model,
                    "rows": {"train": len(train_rows), "held_out": len(held)},
                    "optimisation": {
                        "steps": args.steps,
                        "micro_batch": args.micro_batch,
                        "grad_accum": args.grad_accum,
                        "lr": args.lr,
                        "head_lr": args.head_lr,
                        "warmup": warmup,
                        "seed": args.seed,
                        "weight_decay": opt.param_groups[0]["weight_decay"],
                    },
                    "evaluation": {
                        "eval_batch": args.eval_batch,
                        "held_out_ce_before": head_ce,
                        "held_out_ce_after": head_ce_after,
                    },
                    "determinism": det_label,
                    "grad_ckpt": not args.no_grad_ckpt,
                    "teacher": {"file": args.teacher, "weight": args.teacher_weight, "covered_rows": len(teacher_rows)},
                    "loss_curve": history,
                    "moved": {"head": head_moved, "lora": lora_moved, "torso_unchanged": not torso_moved},
                    "head_config": {"hidden": model.config.hidden_size, "d": head_d, "mid": head_mid},
                    "lora_grads": grads,
                    "not_claimed": (
                        "our loop on this corpus and head design; not run 7's arm, and no number here "
                        "is comparable to 0.60475. A run with a different eval_batch is not comparable "
                        "in held-out CE either — re-measure from lora/ + head.pt instead."
                    ),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}/lora, {out}/head.pt, {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
