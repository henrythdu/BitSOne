# BitSOne-s1 — a ternary System One decider

**State in, typed probabilities out.** One forward pass, no autoregressive decode: read a state and
return a distribution over the options, which the host turns into a `noul` (yes/no), a `choice` (one
of N) or a `score` (a level on an ordered scale). It does not generate prose, and that is the point.

This repository is the model's home: the recipe, run 7's measured numbers, the pipeline code that
loads and reads the model, and the record of how it was produced. It is not an experiment log — run
7 is the last run, and the research record behind it is frozen at the git tag
`research-archive-2026-10-15`.

**Published:** <https://huggingface.co/hdu/BitSOne-s1> (private; the run-7 LoRA adapter, pointer
head, tokenizer, calibrated config and card).

## The model

| | |
|---|---|
| base | [`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) — ternary BitNet b1.58, 2B, revision `04c3b9ad…`, unmodified |
| trained | LoRA `r=16`, `alpha=32` on 7 projections + a **pointer head** over the last hidden state (`num_slots` 24, `pointer_dim` 256) |
| trainable | 22,943,232 of 2,435,763,712 parameters (0.94%) |
| recipe | [`experiments/run7.yaml`](experiments/run7.yaml) |
| run by | the arm's own harness, outside this tree; nothing from it is vendored or imported here — the record is [`runs/run/`](runs/run) |
| weights | `adapter_model.safetensors` `cf4826aa…`, `slot_head.pt` `59debd46…` |

## What is ours and what is adopted

Ours: the ternary base swap, the 12 → 7 LoRA target reduction, the constant schedule, the two
generated corpus files, the teacher distributions, the calibration, the holdout discipline, the
runs — and every line of the pipeline code in [`bitsone/`](bitsone) and
[`checks/`](checks).

Adopted, by ruling: the **corpus** (six files, 123,339 rows) and the **pointer-head design** — one
attention score per option, read from that option's own last token. Both are adopted as facts to
reproduce, not as code to import: this tree imports nothing of anyone else's, and
[`checks/imports_check.py`](checks/imports_check.py) proves it mechanically on every push. Run 7 was
executed outside this tree; the record of it is [`runs/run/`](runs/run), and no number printed by
the trainers here is comparable to its result.

Each claim has a gate that reads the code it is about — no gate compares us to anyone else's
implementation, because none is imported:

| claim | gate | measured |
|---|---|---|
| our pointer head is the function its own docstring specifies | `checks/pointer_head_check.py` | the shipped head loads strict, and its forward matches the documented formula recomputed independently with `F.layer_norm`/`F.linear`; mask → −inf incl. an all-masked row; K-permutation equivariant; fp32 out |
| our renderer is deterministic and unchanged | `checks/corpus_render_check.py` | 123,339 rows / 662,935 spans / 662,935 key tokens, self-consistent; rendered digest `f67859b0…` recorded, so a rendering change fails the gate |
| our serving path is self-consistent | `checks/serving_replay_check.py` | one request through two of our paths gives identical log-probs; option permutation permutes the scores; log-probs finite and summing to 1; key indices strictly increasing |

## Results

Measured on the 4,000-row `norule` holdout, which is **verified disjoint from the training corpus**
(0 of 21,000 rows appear in it). Calibration was fitted on half of the split and verified on the
other half.

| | uncalibrated | after per-kind temperature |
|---|---|---|
| accuracy | **0.60475** | 0.60475 |
| ECE (10 bins) | 0.0769 | **0.0439** |
| NLL | **0.9224** | 0.9515 |
| mean confidence | 0.5278 | 0.5825 |

Accuracy is temperature-invariant, so calibration buys no accuracy: it cuts ECE by 43% and, because
the per-kind fit optimises ECE while the global fit optimises NLL, it raises NLL. Temperatures:
global 1.0157, `noul` 0.5036, `choice` 0.8630, `score` 1.1297.

Run 7 over run 6 is **+0.30 pp against a 2-SE resolution of 1.55 pp** — inside resolution, and stated
that way.

**What is not claimed.** The `test` and `ood` splits were never read (read-once discipline), and no
evaluation suite has been run against this checkpoint. No benchmark number is stated here, because
none has been measured.

## Quick start

```bash
pip install -r requirements.txt
sh patches/verify.sh          # every import accounted for: no weights, no GPU, no torch
```

Load and answer with the published artifacts. One import, our code the whole way — renderer,
key-token rule and head:

```python
from bitsone.serve import load

engine = load()                       # base + adapter + head, from the published artifacts
response = engine.answer(state, kind, instructions, options)
response.best                         # index of the chosen option
response.probs                        # per-option probabilities, summing to 1
```

`load()` takes `checkpoint=<dir>` for a local copy; without it, the published artifacts are resolved
from the Hub.
`checks/serving_replay_check.py` is that path end to end: run it with a GPU, the base weights and
the published checkpoint.

## What run 7 is

The arm is two changes from run 6, and the second is a consequence of the first:

| field | run 6 | run 7 |
|---|---|---|
| `teacher_weight` | `0.0` | `1.0` — distributions on 72,434 of 123,339 rows |
| `train_files` | one file, 52,393 rows | **six files, 123,339 rows** |

The teacher's index is positional into the full concatenation, so the weight is only usable with all
the data — the corpus is not a second lever. Everything else is held: `lr_schedule: constant`,
`gradient_checkpointing: true`, 7 LoRA targets, `seed: 0`.

3,738 optimizer steps at an effective batch of 32 (warmup 112), 0.14 step/s, peak 15.1 GiB, ~$4.4 of
pod time. Run 7's own record — the teacher, the smokes, the result — is in
[`runs/run/`](runs/run).

**One thing was learned by failure:** both pod smokes reported `SETUP EXIT=1` (an unbound variable
under `set -u`), so two digest checks in the launch gate never ran. They were run afterwards. The
episode is recorded rather than tidied away.

## Limitations

- **One seed, one instrument.** 0.60475 comes from a single 4,000-row held-out split; there is no
  seed variance and no second measurement to compare it against.
- **Training, precisely.** Our code trains a pointer head on cached features over the adopted corpus
  (`bitsone/train_head.py`, the run-4 method) and, separately, LoRA + head **jointly** on the packed base
  (`bitsone/train_joint.py`, our loop and our values, the seven LoRA targets and the pointer-head
  design). Neither
  reproduces run 7: that arm ran outside this tree, and no number either trainer prints
  is comparable to 0.60475.
- **Only the data and the head were adopted.** The recipe's *values* came with the harness, not from
  a ruling: the five deviations run 5 recorded, and run 6/7's deltas, are the whole difference.
- **Option count ≤ 24.** `num_slots` is 24 and the measured corpus has K in 2..24. Wider option sets
  are outside the measured range.
- **Calibration trades NLL for ECE**, as the table shows. Choose the objective for your use.
- **The pointer head is architecture-dependent.** A run-7 head with any other run's adapter (or vice
  versa) produces garbage, silently: they are shape- and key-identical, so no shape check catches it.
  model card.

## Layout

```text
bitsone/           our pipeline code: pointer.py (the head), corpus.py (the six corpus files, our
                   renderer + the key-token rule), serve.py (render -> key tokens -> head),
                   train_head.py (head on cached features), train_joint.py (LoRA + head, one loop),
                   eval.py (read-once evaluator)
checks/            the three gates above -- each one reads the code it claims about
patches/           verify.sh -- accounts for every import; nothing fetched
experiments/       the arm: run7.yaml
runs/run/          run 7's record and local checkpoints
docs/model-cards/  the model card, also published as the Hub README
```

The research record that produced this model — earlier runs, the registers, the wiki, the checks —
is not in the tree. It is at the tag: `git show research-archive-2026-10-15:openwiki/`.

## Authors

- Henry Du ([@hdu](https://huggingface.co/hdu))
- Base model: [Microsoft BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (MIT).
- The `System One` architecture and the name `Jev` are TypeSafe's; this project is an independent
  implementation and is **not affiliated with TypeSafe**.

## Attribution

One thing is adopted: the **corpus** — six JSONL files, 123,339 rows, which run 7 trained on. It is
third-party data, used under its own terms, and it is **not redistributed here**: the gates read it
where it lives and refuse to run without it.

Nothing else is taken. The pointer-head design is reproduced from its description and verified
against first principles by [`checks/pointer_head_check.py`](checks/pointer_head_check.py); no code
from anywhere else is vendored, imported or patched into this tree.

## License

[Apache-2.0](LICENSE).
