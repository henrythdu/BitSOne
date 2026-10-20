"""references_check — every repository path that the prose names must exist.

A path in a docstring or a comment is an instruction to the reader. When the file is absent, the
reader finds nothing, and this check reports it.

What it scans: `bitsone/*.py`, `checks/*.py`, the root `*.md` and `docs/**/*.md`. It does not scan
`runs/`: that tree is frozen evidence, and its prose names paths that exist only at the archive tag.

The allow-list is explicit, one entry per path with the reason it is still named. Prefixes are
deliberately not allowed: a prefix that permits a whole historical tree also permits a genuinely
dead file inside it.

Run:  python3 checks/references_check.py
Exit: 0 every named path is tracked, 1 a named path is not, 2 this check could not run.

When testing it, redirect and read `$?`: `python3 checks/references_check.py | head` reports HEAD's
status, so a failing probe reads as a pass. That trap made two negative tests look green before the
exit code was measured directly.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths our prose may name although they are not in the tree, each with the reason it is named.
ALLOWED: dict[str, str] = {
    "experiments/ternary-s1/PREREG-run-4.md": (
        "run 4's pre-registration, frozen at the research-archive tag; the head's axes are defined there"
    ),
}

# Only extensions this repository uses or ships as artifacts. Weight formats are included so a
# prefixed reference to an ignored checkpoint fails; a bare filename (`slot_head.npz`) does not match,
# because whether a Hub-only artifact exists needs the network, not the tree.
PATTERN = re.compile(
    r"\b((?:bitsone|checks|patches|docs|experiments|runs)/[\w./-]+"
    r"\.(?:py|md|sh|json|yaml|jsonl|jinja|txt|pt|npz|gguf|safetensors))"
)

TRAILING = ".,;:)`'\""


def scanned() -> list[Path]:
    files = [*sorted(ROOT.glob("bitsone/*.py")), *sorted(ROOT.glob("checks/*.py")), *sorted(ROOT.glob("*.md"))]
    files += sorted((ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.is_file()]


def present() -> set[str]:
    """Tracked files, plus untracked ones, minus ignored ones.

    Existence on disk is NOT the test: a gitignored artifact (a checkpoint under `runs/`, say) is
    present on exactly one machine, and prose that names it resolves for its author and for nobody
    else. `--others --exclude-standard` keeps the dev loop honest -- a file you just created passes
    before you stage it -- while ignored artifacts stop counting as real.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: cannot list the tree's files, so no reference can be confirmed: {exc}", file=sys.stderr)
        sys.exit(2)
    names = {line for line in out.splitlines() if line}
    if not names:
        print("FAIL: git lists no files, so no reference can be confirmed", file=sys.stderr)
        sys.exit(2)
    return names


def main() -> int:
    names = present()
    files = scanned()
    if not files:
        print("FAIL: no files matched the scan set, so this check would pass on nothing", file=sys.stderr)
        sys.exit(2)

    dangling: list[str] = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in PATTERN.finditer(line):
                ref = match.group(1).rstrip(TRAILING)
                # Tracked or untracked-but-not-ignored counts; a gitignored artifact does not, and
                # neither does a path that is tracked but deleted -- that is the case this exists for.
                if ref in names or ref in ALLOWED:
                    continue
                dangling.append(f"{path.relative_to(ROOT)}:{lineno}: {ref}")

    if dangling:
        print(f"FAIL: {len(dangling)} named path(s) are not in the tree:")
        for item in dangling:
            print(f"  {item}")
        print("\nEither fix the reference or add the path to ALLOWED with the reason it is named.")
        return 1

    print(f"PASS: {len(files)} files name only paths that exist ({len(ALLOWED)} allowed by reason)")
    for allowed, reason in ALLOWED.items():
        print(f"      allowed: {allowed} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
