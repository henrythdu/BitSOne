#!/bin/sh
# Verify this repository's own code is its own.
#
# No network, no weights, no GPU, no torch. CI runs this on every push: it asserts that every
# import in the executable tree is stdlib, declared in requirements.txt, or ours. That is the
# whole claim this repository makes about provenance — our code, accounted for, fetching nothing.
set -eu
cd "$(dirname "$0")/.."

# Our code is ours: stdlib + what requirements.txt declares + bitsone/checks. A dependency nobody
# declared, or a package that arrived through sys.path, fails here by name and line.
python3 checks/imports_check.py

echo "OK: every import in the executable tree is accounted for"

# What our prose points at is real: every repository path a docstring or a doc names must exist.
# A named path that was deleted is an instruction to a reader that leads nowhere.
python3 checks/references_check.py

echo "OK: every path our prose names exists"
