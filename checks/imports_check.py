#!/usr/bin/env python3
"""checks/imports_check.py — every import in the executable tree is one we can account for.

The rule this enforces: our code is ours. An import that is neither stdlib, nor declared in
`requirements.txt`, nor one of our own packages means one of two things, and both are silent
until they break — a dependency nobody declared, or somebody else's code that arrived through
`sys.path`. So this reads the source and names the offender instead of trusting that none
exists. Stdlib-only itself: it must run in CI, where there is no torch and no GPU.

What counts as ours: `bitsone` and `checks`. What counts as declared: the distribution names in
`requirements.txt`, mapped to their import names where they differ, plus `TRANSITIVE` — the ones
those packages bring in themselves. Anything else fails.

Run:  python checks/imports_check.py
Exit: 0 when every absolute import is accounted for, 1 otherwise, printing each offender.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN = {"bitsone", "checks"}
SKIP_DIRS = {"src", "System One Models", "BitNet Models", "Training Data", ".venv-s1", ".git"}

# Distribution name -> import name. Empty today (torch, transformers and peft import under their own
# names); the map exists so a differently-named distribution fails loudly instead of silently.
IMPORT_NAMES: dict[str, str] = {}

# Brought in by the packages above rather than named in requirements.txt; torch imports numpy.
TRANSITIVE = {"numpy"}


def declared() -> set[str]:
    """The import names `requirements.txt` promises."""
    names: set[str] = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        dist = re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0].strip().lower()
        names.add(IMPORT_NAMES.get(dist, dist.replace("-", "_")))
    return names


def files() -> list[Path]:
    """Every Python file we own: the three packages at the root, and root-level scripts."""
    out = [p for p in sorted(ROOT.glob("*.py"))]
    for pkg in sorted(OWN):
        base = ROOT / pkg
        if base.is_dir():
            out += [p for p in sorted(base.rglob("*.py")) if not SKIP_DIRS & set(p.parts)]
    return [p for p in out if "__pycache__" not in p.parts]


def imports_of(path: Path) -> list[tuple[int, str]]:
    """(line, top-level module) for every ABSOLUTE import. Relative imports are ours by
    construction and are skipped.

    Dynamic imports count too: `importlib.import_module("pkg.mod")` and `__import__("pkg")` move the
    module name out of statement syntax and into a string, where a grep for `import` will not see
    it. That is exactly how a path-injected third-party package used to load here, so the check
    reads the call as well as the statement.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a file we cannot parse is a file we cannot clear
        raise SystemExit(f"{path}:{exc.lineno}: syntax error, so its imports are unchecked: {exc.msg}")
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module.split(".")[0]))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name not in ("import_module", "__import__") or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.append((node.lineno, arg.value.split(".")[0]))
    return found


def path_roots(path: Path) -> list[tuple[int, str]]:
    """`sys.path.insert` calls that put a PACKAGE DIRECTORY on the path.

    Why this is a failure and not a style point: adding `bitsone/` itself to `sys.path` gives every
    module in it a second name (`import corpus` alongside `bitsone.corpus`). Two names for one
    module is how the package split reappeared after it was consolidated, and it also lets a file
    in the package shadow anything else on the path. The repository root is the one root; a module
    inside a package reaches its siblings through the package.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:  # already reported by `imports_of`
        return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert"):
            continue
        # insert(index, path): the PATH is the second argument, the index is the first.
        target = node.args[1] if len(node.args) > 1 else node.args[0]
        text = ast.unparse(target)
        if ".parent)" in text and ".parents" not in text:
            out.append((node.lineno, text))
    return out


def main() -> int:
    allowed = set(sys.stdlib_module_names) | declared() | TRANSITIVE | OWN
    offenders: list[str] = []
    checked = 0
    for path in files():
        checked += 1
        rel = path.relative_to(ROOT)
        for lineno, expr in path_roots(path):
            offenders.append(
                f"{rel}:{lineno}: puts a package directory on sys.path ({expr}) — "
                "that is a second import root for the same modules"
            )
        for lineno, mod in imports_of(path):
            if mod not in allowed:
                offenders.append(f"{rel}:{lineno}: imports {mod!r}, which is neither stdlib nor declared")

    if offenders:
        print("FAIL: unaccounted imports in the executable tree\n")
        print("\n".join(offenders))
        print("\nEither declare the dependency in requirements.txt, or delete the code that needs it.")
        return 1
    print(f"PASS: {checked} files imported only stdlib, declared packages and our own ({', '.join(sorted(OWN))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
