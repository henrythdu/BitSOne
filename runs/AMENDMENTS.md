# Amendments to the record

The run record is evidence, so it is not edited. This file is the ledger of every deliberate
exception: what changed, when, why, and what was checked to show nothing else moved. An entry here
is part of the change, not a note added afterwards — a commit that touches `runs/` without an entry
is the defect this file exists to make visible.

Everything below is a change to how the record is *named, located or formatted*. No measured number
in this repository has been altered after the fact. Where a change could have touched numbers, the
check that says it did not is written down.

## 2026-10-19 — the naming ruling

One ruling drove most of these: no third-party code and no third-party name anywhere but the
corpus-attribution line in `README.md`.

| path | change | why | checked |
|---|---|---|---|
| `runs/run7/`, `runs/run7-full-corpus/` | consolidated into `runs/run/`; `run7_teacher_align.py` → `scripts/teacher_align.py`; `CLAIMS.md`, `R2-SPEC.md`, `README.md`, `upload_to_r2_run7.sh` deleted | two run trees for one run; the deleted files described paths that no longer exist, and the uploader archived a different repository's layout | results, manifest and calibration moved byte-for-byte; `RESULT.md`, `CORPUS-MANIFEST.md`, `calibration.json` unchanged in content |
| `runs/run/DIGESTS.md` | deleted | the digests were provenance for third-party code this tree no longer contains or imports | no number lived only there: the result, the calibration record and the manifest stayed |
| `runs/run/out/run7-artifacts-*.tar.gz.sha256` | deleted | it named a tarball that is not in the tree and never was | nothing references it |
| `runs/run/out/calibrated/<previous name>.json` | renamed `bitsone-s1_config.json`; `base_model` changed from a machine-local path to `microsoft/bitnet-b1.58-2B-4T` | the old name was the upstream's; `AGENTS.md` requires a published config to name a usable base rather than a directory that exists on one machine | temperatures unchanged (`temperature` 1.0157, `temperature_by_kind` noul 0.5036 / choice 0.8630 / score 1.1297); every other field identical |
| `runs/run/calib-work/` | untracked (`git rm --cached`) | nine files duplicating `runs/run/out/`; both copies were on disk | byte-identical: `start.log`, `train.log`, `train_config.json` all compare equal |
| `runs/run/out/{start,train}.log`, `runs/run/calib-work/{start,train}.log`, `runs/run/smoke/{smoke-on,smoke-off}.log` | upstream path and name replaced by `<corpus-dir>` / `<external>` / `<corpus>`, `/home/hdu` → `<home>`; three-line header added to each | the ruling, on a repository that is now public | redacted on BYTES from the `352e7b1` blobs, then verified by a single-pass alignment: every step is either an identical byte or a substitution site, so no other byte moved. CR counts preserved (start.log 71 → 71, smoke logs 26 and 21), LF counts moved by exactly the three header lines (820 → 823, 393 → 396). An earlier attempt read the files in text mode, which normalised the `\r`-separated progress lines to `\n`; that version was never pushed as final and was redone from the blob. Its header line and this entry both claimed the counts were unchanged, which was false of that artifact |
| `runs/run/CORPUS-MANIFEST.md`, `runs/run/smoke/TEACHER-SMOKE.md` | same substitution | same | row counts and every figure unchanged |

## 2026-10-19 — how the redactions were verified

The rule for touching evidence: write bytes, never text. A text-mode read/write normalises line
endings, and the run logs carry `\r`-separated progress updates that are not lines at all.

The predicate used, and the one to use again: rebuild the file from its blob at the recorded commit,
then walk both byte strings in one pass. At every offset either the bytes are equal, or a
substitution pattern matches and its replacement matches in turn. If the walk ends at both ends,
nothing but the substitution sites changed — length cannot drift, a dropped byte cannot pass, and a
newline conversion fails immediately because `\r` has no pattern. Counts are asserted as a second,
independent witness: CR equal, LF equal plus the header.

One more rule, learned by a checker passing for the wrong reason: derive the substitution set from
the artifact or assert it — never restate it from memory. A verification that rebuilds a file with a
shorter pair list reconstructs something the tool never produced, and its failure looks like a
defect in the file. The set in `bitsone` terms is the six pairs applied in the longest-prefix-first
order recorded above; a checker that omits `/home/hdu/sdbuild/strands-decider` will report a
spurious 18-byte shortfall on every log.

## 2026-10-19 — receipt format

| path | change | why | checked |
|---|---|---|---|
| `runs/eval/*.read.json`, `runs/eval/*.report.json` | `"split"` changed from an absolute path to the file's basename; `"split_source": "external corpus, not redistributed"` added | receipts are published, and the path named both the upstream source and this machine's home directory | every report parsed before and after and compared field by field with `split`/`split_source` excluded: identical |

The change also lives in `bitsone/eval.py`, so receipts written from now on have this shape. A
receipt still answers what read-once needs: which split, that it came from outside the tree, when it
was read, and why it was read again.

## 2026-10-19 — the eval pass that measured the bare base

Not an amendment, a correction, recorded here because the void numbers are still on disk:

| split | void number | reason |
|---|---|---|
| `catchall_v20` | accuracy 0.1725 | the served adapter never bound; the model scored every row with no fine-tune |
| `adequacy_hs2` | accuracy 0.5000 | as above |
| `adequacy_gen` | accuracy 0.5993 | as above |
| `flips_v20` | accuracy 0.5167 | as above |

Their receipts say `first read`; the re-reads say `adapter-weights-did-not-bind: the first read
scored the bare base model`. The reports on disk are the corrected ones; the void numbers survive
only in this table and in the receipts.
