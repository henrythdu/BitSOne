# The teacher works during training — measured, not argued

Two smoke arms, three optimizer steps each, on the laptop's RTX 5000 Ada.
The question: is the teacher term LIVE, or merely wired?

## The result

| step | teacher_weight 1.0 | teacher_weight 0.0 |
|---|---|---|
| 1/3 | loss 1.7397 · teacher kl 0.0000 | loss 1.7397 |
| 2/3 | loss 1.6485 · teacher kl 0.0000 | loss 1.6485 |
| 3/3 | loss 2.6616 · teacher kl 1.3300 | loss 1.3316 |

`2.6616 - 1.3316 = 1.3300` = the printed `teacher kl`. That is
`train.py:662  step_loss = step_loss + cfg.teacher_weight * tkl` confirmed
numerically -- but only to about 0.7%, not to four decimals. The section
after next is why.

## The identity holds in matched pairs; one third-party run landed low

This caveat came from an independent reproduction, not from my own runs.
Step-3 losses for the OFF arm, same config, same seed:

```text
  run                                     OFF step 3/3
  mine (first)                                  1.3316
  mine (re-run, paired with an ON re-run)       1.3316
  Agent_B (independent)                         1.3230
```

The ON arm gave `2.6616` in both of my runs. So both of my matched pairs give
`2.6616 - 1.3316 = 1.3300`, which IS the printed `teacher kl` exactly, and that
is not retracted. But one independent OFF run landed 0.0086 lower and we have no
matched ON run of theirs to pair it with, so the identity is NOT established to
four decimals across every observation. It holds exactly in the two pairs where
both arms ran under the same conditions, with a measured run-to-run spread of
**0.0086 (0.65%)** on step 3.

What this does and does not refute: the coefficient is a SOURCE fact at
`train.py:662` and needs no numerical support. What the numbers show is the
MAGNITUDE of the term, and they show it to about 0.7%.

**The mechanism is not asserted.** Steps 1 and 2 match exactly in every run,
which rules out a different row, a different batch order and a different pool --
any of those would move step 1. The divergence is confined to step 3, the last
step, where the parameters have absorbed two updates and the LR has reached
zero. Floating-point non-associativity in the backward pass is a plausible
mechanism and was NOT measured, so it stays named and unexplained. This is the
second time this session a plausible mechanism was nearly presented as a
finding; the first was the SIGTERM, and it was caught.

**The finding that outlasts the argument:** the same config and the same seed
can produce a step-3 loss differing by 0.86% across runs. Every cross-run
comparison in this project -- run-5 vs run-6, run-7 vs either -- carries at least
that much. The run-5/run-6 same-rows diagnostic quoted at +3.1 pp should be read
against that floor. Three steps is the least stable place to measure a noise
floor, so treat ~0.9% as a LOW estimate.

## Why the mutation is single-variable

`config-teacher-on.yaml` and `config-teacher-off.yaml` differ in exactly two
fields, measured by parsing both with `yaml.safe_load` and comparing every key:

```text
fields differing: 2
  output_dir:     ON='out/teacher-on'   OFF='out/teacher-off'    (path only)
  teacher_weight: ON=1.0                OFF=0.0
```

`teacher_file` is deliberately LEFT SET in both arms. `train.py:470` gates the
LOAD on the file, not on the weight, so both arms read the same 72,434 rows and
attach them to the same corpus rows; only the multiplier at `:662` changes.

```text
[strands-decider] teacher distributions on 72,434 of 123,339 rows (weight 1.0)   <- arm A
[strands-decider] teacher distributions on 72,434 of 123,339 rows (weight 0.0)   <- arm B
```

72,434 is the teacher file's own row count, so ALL of its rows resolved. The
teacher file is `teacher/teacher_v20.jsonl`, sha256
`c6048ddc21c4cc347980b38d0561f8dd849daca42c970f9103ea482681ee0c5b`.

## The alignment is correct, and the check is shown to fire

The teacher's rows carry `{i, probs}`; the corpus row at index `i` carries
`label`. If the alignment is right, `argmax(probs)` must be that row's label.
Measured over all 72,434 teacher rows against the six-file concatenation:

```text
    offset   width-guard passes   argmax==label   verdict
        +0              1.0000          0.9790   CORRECT
        +1              0.7857          0.4678   MISALIGNED
        -1              0.7856          0.4687   MISALIGNED
        +2              0.7866          0.4652   MISALIGNED
        -2              0.7863          0.4661   MISALIGNED
      +137              0.7778          0.4656   MISALIGNED
```

- `+0` is the file as written: **97.90%** agreement, **zero** width mismatches and
  **zero** indices out of range, over all 72,434 rows.
- Every deliberate misalignment collapses it to ~46.6%. The test therefore FIRES.
- Chance, computed as the pool's `sum p_K * (1/K)`, is **0.3220**. The misaligned
  value sits above chance because the corpus is task-grouped, so a neighbouring
  row often carries the same label anyway.

**The width guard alone would not have been enough, and this is the part worth
keeping.** `train.py:477-479` fires only when the mis-resolved row has a
different option count. Measured over the pool, adjacent rows share an option
count **80.91%** of the time pooled — `train_v5` 86.7%, and `adequacy_hs2` and
`adequacy_gen` both **100.0%**, where an off-by-one is undetectable by it at all.
Read the other way, that same figure is the guard's power: it fires on **21.43%**
of offset-+1 rows, against a **19.09%** prediction from the adjacent-K rate. Two
independent routes, one answer.

So the guard's silence over 72,434 rows is WEAK evidence that the alignment is
right; the argmax test is strong evidence. They agree, and the strong one is the
one to cite.

PEER VERIFICATION, as it actually stands. Agent_B reproduced the alignment test
EXACTLY: 0.9790 at offset 0, all five injected offsets under the 0.60 bar, chance
0.3220, exit 0, from a fresh process. They also re-ran the OFF arm and reproduced
every line except step 3, which is the one divergence recorded above. On the
source: `train.py:470` gating the load on the file, `:657` on the weight, and the
attach-before-split ordering across `:471-487` were all established by them
independently -- and the last is why the log line reads 123,339 rather than a
post-split count.

## A memory finding, and where it stops being a measurement

Run-7 sets `gradient_checkpointing: false` at `micro_batch_size: 8`. Measured
peaks for the same model and corpus:

| configuration | peak |
| --- | --- |
| run-6, mb 8, checkpointing ON | 14.6-15.1 GiB |
| smoke, mb 1, checkpointing ON | 5.6 GiB (four runs, all identical) |
| smoke, mb 1, checkpointing OFF | **died** above 13.04 GiB allocated |

The mb-1 / OFF attempt failed with `OutOfMemoryError: tried to allocate 22.00 MiB`
on a 15.65 GiB card holding 2.13 GiB externally and 13.04 GiB already allocated.
By a hair, but it failed.

**What IS measured:** one row at 4096 tokens with checkpointing off needs more
than 13.04 GiB of a 15.65 GiB card, where the same row with it on peaks at 5.6.

**What is NOT measured:** that run-7 at mb 8 exceeds 48 GB. That is an inference
from a single mb-1 row on a card under pressure, and the failing allocation was
only 22 MiB, so it establishes "just over 13.04 GiB", not "eight times that". A
pod smoke at run-7's exact settings tests it directly. Do not upgrade the
inference by repeating it.

`v21b.yaml:15` classifies checkpointing itself as a speed-only setting --
"gradient_checkpointing off, precompute_frozen_kl on" -- so if a smoke OOMs,
turning it ON is result-neutral by the reference's own description, and runs 5
and 6 already ran that way. Run-5's `DEVIATIONS.md` does not record it, which is
the C27 shape: a fact present in the artifacts and absent from the record.

## The step rate is a range, not a number

Identical config, same seed, same machine, four 3-step arms:

```text
              step 1   step 2   step 3
  on (first)    0.28     0.37     0.51
  off (first)   0.21     0.34     0.49
  on (re-run)   0.60     0.97     1.26
  off (re-run)  0.64     1.04     1.38
```

A **2.5x spread across identical configurations**. So a 3-step arm cannot rank
anything -- including the checkpointing question it was going to be used for --
and a single pod smoke yields a rate with a 2.5x error bar. It would replace the
cost floor with a RANGE, not with a number. Reporting one smoke's rate as "the
run-7 step rate" would put a rate error in a new place.

## What this does NOT show

- Steps 1 and 2 are identical across arms because `micro_batch_size: 1` puts ONE
  row in each batch and only a teacher-covered row carries the term. In run-7
  the effective batch is 32 and 72,434/123,339 = 58.7% of rows carry a teacher,
  so P(a batch of 32 containing no teacher row) = (1 - 0.587)^32 ~ 1e-12: the
  term is on essentially every run-7 step. The sparse coverage here is a
  micro-batch artifact, NOT a property of run-7.
- `1.3300` is ONE row's KL, not a per-batch mean. Do not extrapolate a loss
  magnitude from it. `_teacher_kl(...).mean() * part.teacher` averages over the
  batch's teacher rows and divides by the batch share, so the value run-7 sees
  will differ.
- The `final eval` lines (ON `val_loss 2.28202` / OFF `2.2960530566237867`) come
  AFTER THREE STEPS with the LR already decayed to zero by the cosine schedule.
  They are not evidence about quality and must not be read as any.

## What the same two runs do confirm

```text
[strands-decider] trainable 22,943,232 / 2,435,763,712 (0.94%)    both arms
[strands-decider] train=119,623 val=3,716                         both arms
```

- `trainable` matches run-5's and run-6's recorded line exactly, so the
  seven-target LoRA resolved identically on all three.
- `119,623 + 3,716 = 123,339`, the six-file index space. This is the REALIZED
  split, measured rather than derived, and it settles run-7's step count:
  `119,623 // 8 = 14,952` then `// 4 = 3,738`.

## Invocation, verbatim

```bash
cd runs/run7-full-corpus/smoke
PYTHONPATH=/home/hdu/sdbuild/strands-decider/src \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hdu/Projects/BitSOne/.venv-s1/bin/python -m strands_decider.cli train \
  --config config-teacher-on.yaml \
  --base-model "/home/hdu/Projects/BitSOne/BitNet Models/bitnet-b1.58-2b-4t"
```

Arm B is the same line with `config-teacher-off.yaml`. Logs kept here as
`smoke-on.log` and `smoke-off.log`.

Both arms exit 0.

## Deviations from run-7 — all listed in the configs' own headers

`max_steps 3`, `micro_batch_size 1`, `grad_accum 1`, `gradient_checkpointing true`,
`kl_frozen_weight 0`, `precompute_frozen_kl false`, `log_every 1`,
`eval_every 100000`.

`train_files` is NOT a deviation: it carries run-7's six files in run-7's order,
because the teacher's indices address that space and a subset would move every
index while the option-count guard at `train.py:477` stayed silent — the exact
misalignment the six-file requirement prevents.

`gradient_checkpointing true` is a forced deviation: the first attempt at
`micro_batch_size: 2` with it false died with `OutOfMemoryError: CUDA out of
memory` at 12.81 GiB allocated, because a peer session's `llama-server` holds
2.1 GiB (`nvidia-smi --query-compute-apps`). Checkpointing changes recomputation,
not arithmetic.

## Stack identity

The smoke ran on `.venv-s1`: torch 2.11.0+cu128, transformers 5.18.0, peft
0.21.2, safetensors 0.8.0, onebitllms 0.0.4, numpy 2.5.3, datasets 5.0.1,
accelerate 1.15.0. These are the versions the pod's `requirements-pod.txt` pinned (that file was
removed from the tree in the 2026-10-15 cleanup; it is at the tag
`research-archive-2026-10-15`), so what was measured here is on the stack the pod ran.
