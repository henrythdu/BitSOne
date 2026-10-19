# run-7 corpus manifest — five objects, one name collision

Written 2026-10-14. Every number below is a measurement or an arithmetic derivation
stated as such. Nothing here is copied from a comment.

## Why this file exists

During run-7's design, **five different row counts were each called "the pool"** —
by two agents, in the same session. They differ by up to 2.3x, every one of them is
correct, and none substitutes for another. Two of them floor to the same step count
and a third does not, so the collision is not cosmetic.

| object | value | derivation |
|---|---|---|
| **index space** | **123,339** | `wc -l` of the six files in v21b's order, summed |
| **idealised train rows** | **119,638** | `123,339 × 0.97`, `val_fraction: 0.03` |
| **realised train rows** | **119,623** | the same cut applied per task: `max(1, int(len(items) × 0.97))`, so the total is task-dependent |
| **post-dedup unique** | **119,040** | unique rows of the same six files under the full-row key (state, kind, task, options, label) |
| **run-5/6's corpus** | **54,210** | the six files as actually subsampled for those runs |

`119,638` and `119,623` differ by 15 rows and **floor to the same step count, 3,738**.
`119,040` floors to **3,720**. So the step count does not depend on which of the
first two you name, and does depend on whether you dedup.

**The dedup is ours, not the recipe's.** `pin_split.py` dedups and pins; nothing in
`train.py`, `format.py` or `recipe.sh` does. Run-7 runs v21b verbatim, so it trains on
the raw 119,62x rows and the trainer's `steps_per_epoch = len(loader) // grad_accum`
counts those. **Run-7 is therefore 3,738 steps.**

## The six inputs, in v21b's order — the order IS the teacher's index space

| file | rows | sha256 (16) |
|---|---|---|
| `train_v5.jsonl` | 100,449 | `3d17d1484116d7f1` |
| `multistep_v14.jsonl` | 12,909 | `19e769cf7da6ff71` |
| `generated_v16p.jsonl` | 2,148 | `4b7f9ac6ad610923` |
| `generated_v18p.jsonl` | 1,667 | `c2ac8f7191911940` |
| `adequacy_hs2.jsonl` | 4,866 | `2a84d4f555ebd4cb` |
| `adequacy_gen.jsonl` | 1,300 | `afbf3dc350456320` |
| **sum** | **123,339** | |

Read from `<corpus-dir>/data/`. The first row of each begins an
absolute offset: train_v5 `[0 .. 100,448]`, multistep `[100,449 .. 113,357]`,
v16p `[113,358 .. 115,505]`, v18p `[115,506 .. 117,172]`, hs2 `[117,173 .. 122,038]`,
gen `[122,039 .. 123,338]`. **v21b:29-31 states this same semantics** ("indices into
train_v5, and v14's own on the multi-step rows (offset past train_v5)"), so the space
is fixed by two independent records.

**Any change to this order or to a file's rows moves every teacher index.** That is
why the corpus is restored rather than repinned: against run-5/6's 54,210-row subsample,
50.1% of the teacher's rows would `IndexError` (`i >= 44,149`) and 49.9% would resolve
to the wrong example, with only the option-width guard at `train.py:477` in between —
and that guard cannot fire when two rows share an option count.

## The teacher

| artifact | rows | sha256 |
|---|---|---|
| `runs/run7-full-corpus/teacher/teacher_v20.jsonl` | 72,434 | `c6048ddc21c4cc347980b38d0561f8dd849daca42c970f9103ea482681ee0c5b` |

Built 2026-10-10 by `<corpus>.data.distill` with no GPU, as a merge of two
committed files. Schema `{"i": int, "probs": [float, ...]}` — an offset plus a
distribution over option slots, **not a row count**. 59,525 of its rows come from the
teacher's agreement with gold on `train_v5` and 12,909 are appended `v14` replay, which
is why it is 72,434 rather than 100,449. 0 width mismatches against the index space
above. It is gitignored (`runs/run7-full-corpus/.gitignore` holds `teacher/`), so this
digest is its only binding.

## The base — the one field that is the experiment

`BitNet Models/bitnet-b1.58-2b-4t/`, `quantization_mode: offline`,
`linear_class: autobitlinear`, `model_type: bitnet`, hidden 2560, 30 layers,
`model.safetensors` 1,178,623,988 B, sha256 `8143ae115ed6babe...`. All seven files
verified byte-identical against the `run5/base` prefix the pod synced.

**`lora_targets` is SEVEN, not v21b's twelve.** Measured on the base's own safetensors
header: 542 tensors, layers 0..29, `q/k/v/o/gate/up/down_proj` at 60 each,
`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` at zero — absent.

## Launch gates, with their expected values

Not measurements to be re-taken once and filed — assertions that must hold on the pool
that ships.

1. **Length.** Expected max **2,741** tokens over all 123,339 rows through the real
   collator; **zero** rows at or over 3,584; hard bar `max_length` 4096. v21b:15
   declares "no training row is longer than 3,072" and 2,741 sits under it.
   **This gate has teeth because `collate.py:99-118` RAISES** on an option span with
   no tokens left — an over-length row would kill the run at whichever step it landed
   on, after paying for every step before it.
2. **Disjointness.** `holdout_v5_norule.jsonl` against `train_v5.jsonl`: **0 shared
   rows** at row level, 24 shared states under `" ".join(str(state).lower().split())`.
   This is the property the 21,000-row OOD comparison rests on.
3. **First-400 task mix.** `evaluate_loss(..., max_batches: int = 50)` at
   `train.py:256` with a val batch of 8 scores 400 rows, and 400 of ~3,600 drawn across
   ~46 tasks can be skewed by chance. Assert the first 400's task mix against the whole
   val set, so "the 400-row number is representative" is measured rather than believed.
4. **Config diff.** Parse `experiments/run7.yaml` (was `runs/run7/config-run7.yaml` until the
   2026-10-15 cleanup; same bytes) and v21b.yaml and report the
   field-by-field diff. It must be exactly four: `base_model`, `lora_targets`,
   `teacher_file`, `output_dir`. Asserting the parsed diff rather than trusting the
   config's own header comment is the point — the header made two claims about its own
   diff today and both were wrong before review.
