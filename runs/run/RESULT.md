# Run-7 result — the teacher term plus the six-file corpus

**One line: the two changes bought less than the instrument can resolve.**
+0.30 pp over run-6 on the 4,000-row holdout, against a 2-SE resolution of 1.55 pp.

## What was run

Run-7 is run-6's config with **two** changes and nothing else:

| field | run-6 | run-7 |
| --- | --- | --- |
| `teacher_weight` | `0.0` | **`1.0`** |
| `train_files` | one file, 52,393 rows | **six files, 123,339 rows** |

These are one decision, not two: the teacher's `i` is a positional index into the
full 123,339-row concatenation, so the weight is only usable with all the data.
Everything else was held: `lr_schedule: constant`, `gradient_checkpointing: true`,
`lora_targets` 7, `kl_frozen_weight: 0.0`, `precompute_frozen_kl` absent, `seed: 0`.

Config `experiments/run7.yaml`, 9,631 bytes, 35 keys. Digests are no longer kept in this
repository; the recipe file is the record.

The upload script that pushed these artifacts to R2 was retired with the digest machinery: it
archived an older tree layout and every step after that was a SHA comparison. The artifacts in
this directory are the record now.

## How it ran

| | |
| --- | --- |
| pod | `hb08jc2k0dscz8` on machine `3nvo9fqxxych`, A6000, $0.59/h |
| steps | 3,738 optimizer steps (warmup 112) |
| rate | 0.14 step/s, held; `peak_mem` 15.1 GiB |
| data | `train=119,623 val=3,716` |
| trainable | `22,943,232 / 2,435,763,712 (0.94%)` |
| teacher | `distributions on 72,434 of 123,339 rows (weight 1.0)` |
| exit | `SETUP EXIT=0`, `TRAIN EXIT=0`, `DONE (full).` |
| stopped | `EXITED`, `Exited by user: Thu Oct 15 2026 04:11:42 GMT+0000` |
| cost | ~$4.4 for 7.5 h |

`SETUP EXIT=0` is the launch gate: both pod smokes read `SETUP EXIT=1` because
`pod_setup_run7.sh:58` was an unbound variable under `set -u` at line 210 and 219,
which meant the patched-trainer digest and the base-model digest **never ran**.

## The result

Same 4,000-row `norule` holdout, same instrument (`runs/calibrate.sh`,
`--split calib --limit 4000`), `holdout_v5_norule.jsonl` sha256 `f5796ad2301903ff`:

| run | accuracy |
| --- | --- |
| run-5, cosine, no teacher, 1 file | 0.57075 |
| run-6, constant, no teacher, 1 file | 0.60175 |
| **run-7, constant, teacher, 6 files** | **0.60475** |

```text
run-7 - run-6 = +0.00300 = +0.30 pp
SE at acc=0.60475, n=4000 = 0.77 pp
that delta = 0.39 SE  ->  NOT significant
run-7 - run-5 = +0.03400 = +3.40 pp = 4.40 SE
this instrument resolves 1.55 pp at 2 SE
```

**The honest sentence: the teacher term plus the 2.35x corpus bought less than
1.55 pp, measured at +0.30 pp.** Not zero — below resolution. The instrument is
sharp enough to have seen an effect that size, and did not.

The fit, verbatim:

```text
global temperature = 1.0157
per-primitive choice=0.8630, noul=0.5036, score=1.1297  (written to checkpoint)

before:        n=4000  accuracy=0.60475  ece=0.07693445760663974  nll=0.9224174273682376  mean_confidence=0.5278155423933603
after global:  n=4000  accuracy=0.60475  ece=0.08092329899738833  nll=0.9223586791646411  mean_confidence=0.5238267010026116
after:         n=4000  accuracy=0.60475  ece=0.043936183964703064 nll=0.9514904524288937  mean_confidence=0.5824696598233541
```

- **accuracy is identical in all three rows**, so the headline is temperature-invariant.
- **global `T = 1.0157`: the model was already globally calibrated**, and fitting it
  made ECE slightly *worse* (0.0769 -> 0.0809). That is consistent with the script's
  own note that the global comes from `objective="nll"` while `by_kind` comes from
  `objective="ece"`.
- **the per-kind temperatures do the work**: ECE 0.0769 -> 0.0439, a 43% cut.
  `noul=0.5036` is 2x from 1, so that head was substantially underconfident.

## The in-run prints, and why they are not the result

The trainer prints `evaluate_loss`, which is capped at 50 batches = **400 rows**
(`train.py:256` default, `:261` break, `:708` and `:727` pass no `max_batches`), and
the val loader does not shuffle (`:549-552`), so every in-run eval scores the SAME
400 rows:

```text
step  500   0.6937 / 0.6875
step 1000   0.5952 / 0.7750
step 1500   0.5584 / 0.7525
step 2000   0.5507 / 0.7550
step 2500   0.4932 / 0.7925
step 3000   0.5133 / 0.7925
step 3500   0.4756 / 0.78
final       0.4668 / 0.82
```

**These are not comparable to any other run.** The prefix is biased: run-6's final
checkpoint scored `0.94` on the first 400 rows of its val file and `0.7717917675544794`
on all 1,652, a **+16.8 point** bias. And the bias amplifies differences enough to
flip their sign: run-5 and run-6 are **3 rows apart** on the full 1,652 and **70 rows
apart** on their 400-row prefixes — 23.3x. Neither `val_loss` nor `val_acc` is
monotone within the run either; `val_loss` fell to 0.4932 at step 2500 and rose to
0.5133 at step 3000.

## The pinned-1,652 number, and why it cannot be compared

Agent_B produced it on identical rows at T=1.0 by argmax
(`checks/run6_clean_eval.py`, output `out/clean_eval_1652.json`):

| run | correct | accuracy |
| --- | --- | --- |
| run-5 cosine | 1278/1652 | 0.7736077481840193 |
| run-6 constant | 1275/1652 | 0.7717917675544794 |
| **run-7 teacher** | **1352/1652** | **0.8184019370460048** |

run-7 - run-6 = **+4.66 pp at 3.32 SE**, which reads as a significant win. **It is not.**

**Every one of the 1,652 val rows is inside run-7's training data.** Measured by keying
every row by content `(kind, state, instructions, options, label)` and looking for the
val keys inside the six-file corpus on R2:

```text
val rows                                      1652
distinct val keys found in the six-file union 1652
tasks fully in the union 46, partly 0, absent 0
```

and against run-6's own training file:

```text
train_pinned 52,393 rows   val_pinned 1,652 rows   INTERSECTION = 0
```

**So the leakage is ASYMMETRIC, which is not how either of us had it framed.** run-6's
`train_files` was one file, `train_pinned.jsonl`, 52,393 rows — a subset of `train_v5`
with the val rows held out, intersection exactly 0. run-7's `train_files` is the full
six files, which include all 100,449 rows of `train_v5`. **run-6 held these rows out;
run-7 trained on them.** The +4.66 pp is the size of that, and it is memorisation.

The 4,000-row `norule` holdout, by the same test, is clean:

```text
holdout rows inside run-7's training corpus: 0 of 21000
```

**So the two instruments do not disagree about the model. One of them is not held out.**
The held-out number is +0.30 pp. The headline above stands unchanged.

### The `gen:*` concentration is real, and contamination does not explain it

The top eight per-task gains are all `gen:*` (`tie_break` +0.500, `surface_trap` +0.500,
`precedence` +0.400, `exception` +0.400, `running_total` +0.333, `cross_artefact`
+0.333, `version_in_force` +0.250, `rubric` +0.250), and each individual cell is n=3 to
5, so only the pattern carries information.

**But exposure cannot be the mechanism, because the exposure is uniform**: every task's
val rows are in the corpus, `gen:*` and real-source alike, so "the gain sits on the
families run-7 trained on" cannot distinguish anything.

The decomposition, weighted by val rows:

```text
gen:*  n=80     run-5 0.5375  run-6 0.4625  run-7 0.7125
                run6-run5  -7.50 pp     run7-run6 +25.00 pp = 4.94 SE
real   n=1572   run-5 0.7856  run-6 0.7875  run-7 0.8238
                run6-run5  +0.19 pp     run7-run6  +3.63 pp = 3.77 SE
```

The real-task gain is significant too (+3.63 pp at 3.77 SE), and it is also entirely on
contaminated rows. 20 of the net 77 rows come from the 80 `gen:*` rows — 4.8% of the set
supplying 26% of the gain — and 57 from the 1,572 real rows.

**Three candidate explanations, each tested, each refuted:**

1. **"run-7 trained on generated data and run-6 did not."** REFUTED. run-6's own corpus,
   `train_pinned.jsonl`, carries **21 `gen:*` tasks and 2,168 rows**. Both runs trained on
   generated data; run-7 simply had more of it (5,115 rows).
2. **"`gen:*` is over-represented in the val split."** REFUTED. `gen:*` val rows are
   **1.56%** of the corpus's `gen:*` rows; real val rows are **1.33%** of theirs. A factor
   of 1.17, not the 4.6x the hypothesis needs.
3. **"Headroom — the lower the baseline, the bigger the gain."** WEAK. The weighted
   Pearson correlation between run-6's baseline and the delta is **-0.306**; tasks below a
   0.60 baseline gain +6.48 pp and tasks at or above gain +4.22 pp. Real, but small, and
   nowhere near +25 pp.

**What IS established:** run-7 trained on those exact 80 rows and run-6 trained on none of
them, so the entire effect is on trained rows and is memorisation. **What remains open is
why memorising a synthetic template moves 25 pp where memorising real-source rows moves
3.6 pp.** The current instrument reports per-task, so a per-row correctness dump is what
would answer it; that does not exist yet.

The aggregate is also not a clean sweep: **26 of 46 tasks improved and 20 worsened**,
and the largest real-source task, `yelp_stars` (n=79), **lost 10.1 pp** (0.684 -> 0.582).
The net +77 rows come from small `gen:*` cells against losses elsewhere.

### An earlier version of this analysis was wrong, and how

My first pass grouped the val rows by whether they appeared in `train_v5` and reported
"25 tasks fully CLEAN, +13.61 pp on them versus +2.67 pp on the contaminated ones",
which would have meant the gain was on the *clean* rows and there was no contamination
problem at all. **That grouping tested one file out of six.** The 316 rows it called
clean are in `multistep_v14`, `generated_v16p`, `generated_v18p`, `adequacy_hs2` or
`adequacy_gen` instead. "Clean of `train_v5`" is not clean. The union test above is the
one that means something, and it returns 1652 of 1652.

## What this does not answer

- **`test` and `ood` splits remain UNREAD.** Read-once discipline holds.
- **Whether the teacher term helped at all is unresolved, not refuted.** It was on for
  58.7% of rows every step, and it cost measurable throughput, but this holdout cannot
  see an effect below 1.55 pp. A larger holdout or a different split could.
- **Why the gain concentrates on `gen:*`.** Contamination is uniform, so it is not
  exposure. This is now the interesting open question.
- **Whether any existing run's val split is leak-free.** run-6's is (intersection 0
  with its own training file). run-7's is not, and the cause is the corpus choice
  rather than anything the trainer did. **Any future run that trains on the full six
  files cannot use the pinned split as a held-out set**, and should use
  `holdout_v5_norule.jsonl`, which is verified clean against all six.

## Artifacts

- `out/run7-artifacts-20261015T0412Z.tar.gz`, sha256
  `c61e5dfa055b244e920969f8641b559c894f8214994abbaa257c303894d7d722`
- Built by `build_artifact.sh`, because run-7's pod retrieve writes **no archive and
  no digest**, unlike `pod_train_run6.sh:106-118`. **The sidecar is locally built and
  says so**: field 1 is the digest of the file the script wrote, field 2 names a local
  path, which is itself the signal the pod did not write it.
- Checkpoint nested under `checkpoints/hobson-2b-run7-full`; `calibrate.sh`'s search
  finds exactly one. **The run's own output confirms the flat layout it was nested
  out of**: it printed `saved to <run-root>/out/checkpoints`, against run-6's nested
  `checkpoints/hobson-2b-run6-constant`.
