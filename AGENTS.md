# AGENTS.md — working in this repository

This repository presents **one model**: BitSOne-s1, trained by run 7. It is not an experiment log.
The research record that produced it (runs 1–6, the registers, the wiki, the verification harness) is
frozen at the git tag `research-archive-2026-10-15` and is deliberately absent from the tree:
`git show research-archive-2026-10-15:openwiki/quickstart.md` still resolves.

## Commands

- **Install**: `pip install -r requirements.txt`.
- **Provenance**: `sh patches/verify.sh` — asserts that every import in the executable tree is
  stdlib, declared in `requirements.txt`, or ours. Nothing is fetched and nothing is patched. No
  weights, no GPU, no torch; this is what CI runs.
- **Lint**: `ruff check .` (config in `ruff.toml`; E/F/W/I, line length not enforced).
- **Type check**: `mypy .` / `pyright` — our code is in scope; the read-only input trees are
  excluded by name in `mypy.ini` and `pyrightconfig.json`. `mypy.ini` tolerates a missing
  `torch`/`transformers`/`peft` stub because CI installs none of them; locally they are present and
  their real annotations are used.
- **The three gates** (each reads the code it claims about; the first needs torch, the second needs
  the corpus, the third needs a GPU and the base weights):
  - `.venv-s1/bin/python checks/pointer_head_check.py` — the shipped head loads strict, and its
    forward matches the formula `pointer.py` documents, recomputed here independently.
  - `.venv-s1/bin/python checks/corpus_render_check.py` — every corpus row renders deterministically,
    with spans, key tokens and the option→label binding self-consistent, and the rendered digest
    unchanged from the recorded one.
  - `.venv-s1/bin/python checks/serving_replay_check.py` — one real corpus row through the served
    path: the published artifacts load, the same request twice is bit-identical, every option is
    scored, the scores form a distribution, and permuting the options permutes the labels and
    re-derives exactly the same key tokens. (`bitsone.serve` has no batched entry point; an earlier
    version of this line claimed one.)
- **CI** (`.github/workflows/ci.yml`): `sh patches/verify.sh`, then `ruff check .`, then `mypy .`.
  Nothing else — no weights, no torch, no GPU, nothing fetched. The three gates stay out of CI
  because each needs an input CI does not have (torch, the corpus, the weights).

## Layout

- `bitsone/` — **our code**, eight modules and nothing else: `paths.py` (filesystem facts),
  `corpus.py` (the six corpus files, our renderer and the key-token rule), `pointer.py` (the head,
  its loss and the batch builders), `model.py` (load the base, assert its quantisation mode),
  `serve.py` (the owned entry point: render → key tokens → head), `train_head.py` (head on cached
  features; `dump_corpus` is the cache), `train_joint.py` (LoRA + head, one loop), and `eval.py`
  (the sealed read-once evaluator, receipts under `runs/eval/`).
- `bitsone/` — filesystem facts and artifact resolution (`paths.py`): where the base, the corpus
  and the published checkpoints live, resolved in one place instead of in every caller.
- `checks/` — the gates and the shared harness. A gate must read the code under test; if it cannot,
  it says so and exits non-zero rather than passing. Exit `1` = a claim failed; exit `2` = could not
  check (missing tokenizer, corpus, weights, GPU). Both are failures to CI. A printed `skip` that
  still returns `0` is the bug this rule exists to prevent.
- `patches/` — `verify.sh`.
- `experiments/run7.yaml` — the arm.
- `runs/run/` — run 7's evidence (result, corpus manifest, calibration, history).
- `runs/AMENDMENTS.md` — the ledger of every deliberate change to that evidence: what changed,
  why, and the check that shows nothing else moved.
- `docs/model-cards/` — the model card, also published as the Hub README.
- Attribution (the corpus, adopted under its own terms and not redistributed) — `README.md`,
  *Attribution*.

## Notes

- **No third-party code and no third-party naming.** Two things are adopted: the **corpus** and the
  **pointer-head design**. Both are adopted as facts to reproduce, not as code to import: nothing in
  this tree imports, vendors or clones anyone else's implementation, and `checks/imports_check.py`
  enforces that mechanically. The design debt is paid by verification — `checks/pointer_head_check.py`
  recomputes the head's formula from its own weights, and `checks/corpus_render_check.py` pins our
  renderer with a recorded digest — never by copying. The **corpus attribution line in `README.md`
  stays**: the data is third-party and its terms require it. Do not add a new mention of where the
  data came from beyond that line.
- **Our code trains the head, and by ruling LoRA + head jointly.** `bitsone/train_head.py` trains a
  pointer head on cached features (the run-4 method; the torso is frozen there by construction).
  `bitsone/train_joint.py` trains LoRA and the head in ONE loop on the packed offline base, over the
  adopted corpus: our loop and our values, the adopted corpus, the adopted head design, the seven
  LoRA targets, teacher KL optional. Neither reproduces run 7 — that arm ran outside this tree, and
  no number printed by our trainers is comparable to 0.60475.
- **The corpus is third-party and lives outside the tree.** `checks/corpus_render_check.py` needs it;
  point it with `--corpus-dir` or `BITSONE_CORPUS_DIR`. It is read-only: never write into it.
- **Never rerun an earlier experiment.** The numbers are the record; the record is frozen at the tag.
- **The record is immutable, and amendments are ledgered.** Files under `runs/` are evidence: do
  not edit a number, a log or a receipt in place. When a record genuinely has to change — a
  naming ruling, a format change, a rename — the change lands in the same commit as an entry in
  `runs/AMENDMENTS.md` naming the file, the reason, and what was checked to show nothing else
  moved. A commit that touches `runs/` without an entry is the defect the ledger exists to expose.
- **The published config must name a usable base.** The checkpoint config carries the calibrated
  temperatures and a `base_model` field; if you regenerate it, point `base_model` at
  `microsoft/bitnet-b1.58-2B-4T` rather than at a machine-local directory. Its filename is due to
  change under the 2026-10-19 ruling (no third-party naming). A published artifact's name is not a
  local edit: renaming it means the released checkpoints, the loader and the model card change in
  one release, and the divergence from the released bytes is recorded afterwards in `README.md`
  (the recipe row) and `runs/run/RESULT.md`. Until
  that release happens the current name is the one on the Hub, so treat a repo-only rename as a
  regression, not as progress.
- **No evaluation runs are scheduled here yet.** The benchmark is still to be run; `test`/`ood`
  remain unread by design (read-once discipline).

Until the third-party strip is finished, this file is also where three working rules live; they move
into `docs/adr/` when the strip lands, and the pointers replace these lines.

- **A decision lives in the repo, not in a session.** The 2026-10-19 ruling (no third-party code and
  no third-party naming; one corpus-attribution line in `README.md`) is recorded in the Notes above
  because a session inherited it second-hand and the file said the opposite. Record rulings here, or
  in `docs/adr/` once it exists — never let one live only in chat.
- **Claim directories before a multi-file change.** Parallel sessions coordinate by directory: say
  which of `bitsone/`, `bitsone/`, `checks/`, `patches/`, `runs/`, docs you are editing before you start,
  and leave a directory you have not claimed alone. Two sessions rewriting `AGENTS.md` and `checks/`
  at once cost more than the change was worth.
- **The lens' `unchecked-numeric-parse-python` hits are declared noise.** They sit outside the `ruff`
  rule set (E/F/W/I), gate nothing, and return every run. Either wire the rule into `patches/verify.sh`
  with a declared exemption list, or switch it off in the lens config; until one of those happens, do
  not re-classify them per session.

## Agent skills

### Issue tracker

Issues and specs live as markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
