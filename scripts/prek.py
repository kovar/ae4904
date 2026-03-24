"""
prek — pre-commit checks for the ae4904 workspace.

Requires ruff and ty in the workspace dev group:
    uv sync --group dev

Run directly:
    uv run scripts/prek.py

Install as git pre-commit hook (run once):
    cp scripts/install-hooks.sh .git/hooks/  # or run it
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CHECKS = [
    (["uv", "run", "ruff", "check", "packages/"], "ruff lint"),
    (["uv", "run", "ruff", "format", "--check", "packages/"], "ruff format"),
    (["uv", "run", "ty", "check", "packages/"], "ty type check"),
    (["uv", "run", "pytest", "-m", "not slow", "-q"], "pytest (fast)"),
]


def run_check(cmd: list[str], label: str) -> bool:
    """Run one check command. Returns True if it passed."""
    print(f"  {label}...", end=" ", flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        print("ok")
        return True
    print("FAILED")
    return False


def main() -> None:
    print("prek: running pre-commit checks")
    passed = [run_check(cmd, label) for cmd, label in CHECKS]
    if all(passed):
        print("All checks passed.")
        sys.exit(0)
    else:
        n_failed = passed.count(False)
        print(f"{n_failed} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
