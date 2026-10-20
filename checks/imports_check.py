#!/usr/bin/env python3
"""checks/imports_check.py — every import in the executable tree is stdlib, declared, or ours.

Each absolute import must be in the Python standard library, or declared in `requirements.txt`, or
be `bitsone` / `checks`. Any other import fails this check. The check reads the source; it does not
import the modules. It uses the standard library only, so it runs in CI without torch.

The scan covers every Python file in the executable tree: the root scripts, `bitsone/`, `checks/`
and `runs/` (run scripts). Read-only input trees and the virtualenv are skipped.

Run:  python checks/imports_check.py
Exit: 0 when every absolute import is accounted for, 1 when one is not.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN = {"bitsone", "checks"}
SCANNED = ("bitsone", "checks", "runs")
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
    """Every Python file in the scanned trees, plus the root scripts."""
    out = [p for p in sorted(ROOT.glob("*.py"))]
    for pkg in SCANNED:
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
    """`sys.path.insert` calls that are not `sys.path.insert(0, str(ROOT))`.

    One import root is allowed: the repository root. Adding `bitsone/` to `sys.path` gives each
    module in it a second name (`import corpus` and `bitsone.corpus`), and a file in the package can
    then shadow other modules on the path.
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
        if text != "str(ROOT)":
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
                f"{rel}:{lineno}: adds {expr} to sys.path — the only permitted form is "
                "sys.path.insert(0, str(ROOT))"
            )
        for lineno, mod in imports_of(path):
            if mod not in allowed:
                offenders.append(f"{rel}:{lineno}: imports {mod!r}, which is neither stdlib nor declared")

    if offenders:
        print("FAIL: unaccounted imports in the executable tree\n")
        print("\n".join(offenders))
        print("\nDeclare the dependency in requirements.txt, or remove the code that needs it.")
        return 1
    print(f"PASS: {checked} files import only stdlib, declared packages and {'/'.join(sorted(OWN))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
