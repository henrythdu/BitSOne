---
license: apache-2.0
library_name: peft
base_model: microsoft/bitnet-b1.58-2B-4T
tags:
- system-one
- decision-model
- ternary
- bitnet
- lora
- pointer-head
---

# BitSOne-s1 — a ternary System One decider (run 7)

A **decision model**, not a chat model. It answers typed questions about a state — a `noul` (yes/no,
returned as P(true)), a `choice` (one of N options) or a `score` (a level on an ordered scale) — in
**one forward pass, with no autoregressive decode**.

This is the **run-7** checkpoint: the ternary BitNet b1.58 2B-4T base plus a rank-16 LoRA adapter,
read out by a **pointer head** over the final hidden states — one score per option slot, softmaxed.
The head is trained, not zero-shot.

## Measured

On the 4,000-row `norule` holdout, verified disjoint from the training corpus (0 of 21,000 rows
appear in it):

| | uncalibrated | after per-kind temperature |
|---|---|---|
| accuracy | **0.60475** | 0.60475 |
| ECE (10 bins) | 0.0769 | **0.0439** |
| NLL | **0.9224** | 0.9515 |
| mean confidence | 0.5278 | 0.5825 |

Accuracy is temperature-invariant; the per-kind fit cuts ECE 43% and raises NLL. The `test` and
`ood` splits were never read, and no benchmark suite has been run against this checkpoint.

### Our own evaluation

Nine evaluation splits, scored through `bitsone.serve` — the same renderer, key-token rule and
pointer head that serve a request — on the committed checkpoint. These are **our splits**, not the
`norule` holdout above: the numbers are a different measurement, not a replacement, and they are
not comparable to `0.60475`. None of the nine is a `test` or `ood` split; those remain unread.

| split | rows | accuracy | NLL | ECE |
|---|---|---|---|---|
| `adequacy_gen` | 302 | 0.6424 | 0.5913 | 0.0439 |
| `adequacy_hs2` | 234 | 0.6068 | 0.6707 | 0.0407 |
| `catchall_v20` | 800 | 0.6675 | 1.0190 | 0.0580 |
| `flips_v20` | 120 | 0.5333 | 0.7737 | 0.1405 |
| `generated_v16` | 350 | 0.6829 | 0.7768 | 0.0519 |
| `generated_v18` | 247 | 0.6113 | 0.9296 | 0.0677 |
| `multistep_v14` | 4,084 | 0.7272 | 0.6209 | 0.0599 |
| `para_pairs_v16` | 678 | 0.6844 | 0.7730 | 0.0363 |
| `para_pairs_v18` | 484 | 0.6116 | 0.9286 | 0.0732 |
| **total** | **7,299** | **0.6924** | | 0.0363–0.1405 |

Each split was read once through `bitsone.eval`, which writes a receipt under `runs/eval/`; the card
quotes the committed reports. An earlier pass measured the bare base — the adapter had not bound —
and those numbers are void, recorded in `runs/AMENDMENTS.md` with their corrected counterparts.

## Files

| file | where | what |
|---|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | Hub | the LoRA, r=16, alpha=32, 7 targets (`q,k,v,o,gate,up,down`), `task_type: FEATURE_EXTRACTION` |
| `slot_head.pt` | Hub | the pointer head, 1,316,352 params fp32 |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Hub | tokenizer (the chat template is not read at inference) |
| `bitsone-s1_config.json` (repo) / the previous release's config name (Hub) | both, under two names | the model config, with the calibrated temperatures. Renamed in this repo on 2026-10-19, and its `base_model` changed from a machine-local path to the Hub id; the published copy differs in **name and in `base_model`** until the next release, so the two are not byte-identical by design |
| `calibration.json` | Hub + repo record (`runs/run/`) | the calibration record those temperatures came from |
| `slot_head.npz` | Hub only | the same head as numpy, 5,266,918 B — for runtimes with no torch |
| `adapter-run7.gguf` | Hub only | the LoRA as GGUF, 86,536,320 B, 420 tensors |

Weights are Hub-only: the adapter and the head live beside the run on disk under `runs/run/`, which
is not tracked (they are 90 MB and 5 MB of bytes), so a clone gets the record without the weights.
`runs/run/calibration.json` is the one file that is in both, byte-identical to the published copy
(752 B, checked 2026-10-19). The config is the one file that differs, for the reason in its row.

The base weights are **not** in this repository: they are
[`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), and they do
not change. The config's `base_model` names that id, so a load resolves it from the Hub without
edits; pass a local path instead if you already have the weights.

**The calibrated temperatures are not applied by `bitsone.serve`.** It returns raw softmax over the
head's scores. The temperatures in `bitsone-s1_config.json` were fitted for reporting (the ECE column
above), so a served probability is the uncalibrated one; dividing the logits by the per-kind
temperature reproduces the calibrated column.

## Pairing rule — this is the one way to get garbage

A run-7 head with any other run's adapter produces garbage, and another run's head with the run-7
adapter produces garbage. The heads are **key- and shape-identical**, so no shape check can catch the
mix-up. Only a digest can:

- run-7 head, tensor digest over the 6 tensors in sorted-key order:
  `90252773fadd7ce9dc7114cc341c72e8c16e132d0d54704247802e9bf7760500`
- run-6 head: `8b589362c12c0617070b66354b8479bba2c958144a26f8bd674b9f2f7f7bbc0c`
- file digests: `adapter_model.safetensors` `cf4826aa80505056e6c8ed64178150d9fd5cb4d9c960a4633570ffff2eb0b43e`;
  `slot_head.pt` `59debd4613949edbad35ee5a82b7663bad82480cbe278366743a3bd50a2fa01b`

## Calibration

Fitted on the run-7 calibration split (`calibration.json` records the objective and the half-B
verification). Accuracy is temperature-invariant; ECE and NLL are not.

| scope | T |
|---|---|
| global | 1.0156505703923981 |
| `noul` | 0.5035636236428354 |
| `choice` | 0.8629556148274228 |
| `score` | 1.1296801903935303 |

## Running it with llama.cpp

The head is computed **host-side** — llama.cpp needs no change, because the bitnet graph already
emits exactly the tensor the head consumes (`res->t_embd`, assigned after the final norm, the same
tensor the LM head reads).

```bash
llama-server -m <bitnet base>.gguf --lora adapter-run7.gguf \
    --port 8085 -ngl 0 -c 4096 -b 2048 -ub 2048 \
    --embeddings --pooling none --embd-normalize -1
```

`--pooling none` and `--embd-normalize -1` are load-bearing: a pooled vector has nothing for a
pointer head to address, and the default normalisation destroys the magnitudes the head reads.
`TQ2_0` has no CUDA `vec_dot`, so `-ngl 0` is honest rather than a loss. The base GGUF is not hosted
here; the conversion recipe and the conversion patches live in the companion `BitSOne_GGUF` repo,
where the `TQ2_0` base measured max|dProb| **0.0090** with 25/25 argmax agreement against the torch
arm (F16 arm 0.0048).

**The BOS trap is silent.** `POST /embeddings` prepends BOS; `POST /tokenize` does not. Tokenise with
`add_special_tokens=True` or every index is off by one and the query lands a token early, behind a
plausible answer. Guard: the server's row count must equal the token count.

## Provenance

Trained 2026-10-14/15 on a rented A6000 by an **external training harness at a recorded revision**,
outside this tree, on the six-file corpus (123,339 rows) with a distilled-teacher term at weight 1.0,
a constant learning-rate schedule, seed 0, 3,738 steps. The recipe is `experiments/run7.yaml`
The run's evidence is under
`runs/run/` in the training repository, whose `RESULT.md` states the result and the one
thing that went wrong in the launch gate (two digest checks never ran; they were run afterwards).

**What the training repository owns.** The run itself was not executed by that repository's code —
the record is kept as evidence, and no number printed by the trainers there is comparable to this
one. What its code owns is the serving path, and every part of it is checked against first
principles, not against anyone else's implementation, because none is imported: the **pointer head**
(its forward recomputed from its own weights against the formula its docstring specifies, including
the single shared LayerNorm), the **renderer** (deterministic on all 123,339 corpus rows — 662,935
option spans and key tokens, self-consistent, with a recorded digest that fails the gate on any
change), and **serving** (one request through two of that repository's own paths gives identical
log-probs on these exact artifacts). Adopted by ruling: the **corpus** and the **pointer-head
design**, as facts to reproduce rather than code to import. See *Attribution* in the training
repository's `README.md` for the corpus terms.

**One numerical change since release.** `PointerHead.forward` now casts its inputs to fp32 at entry.
Before that, a bf16 torso fed bf16 vectors straight into the projections: outside autocast it raised
a dtype error, and inside autocast the projection silently ran in bf16, contradicting the head's
documented contract of an fp32 head over a bf16 torso. The published weights are unchanged; the
served head now does what it always claimed, and the difference is smaller than the fp32 rounding
of the logits themselves.

**History.** `main` previously carried the run-2 **letter** head (4 targets, one letter per option
read from the LM logits). That record has been removed from this Hub repo and is preserved in the
training repository at the tag `research-archive-2026-10-15`; it is not this model, and the two are
not interchangeable.
