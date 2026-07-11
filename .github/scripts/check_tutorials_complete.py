#!/usr/bin/env python3
"""Fail if any HowTo tutorial script looks truncated.

Several tutorial scripts were once cut off mid-generation (a long file was
re-emitted and the output was severed), losing all content below the cut and
leaving a script that ends part-way through — typically on a docstring that
promises a plot or code block which never appears. This check guards against
that regression recurring.

A tutorial is considered *complete* when it contains a recognised terminal
section marker (``__Wrap Up__`` or ``__Summary__``). A truncated script never
reaches its wrap-up, so the absence of the marker is a reliable signal that the
script lost content. Deliberate "not written yet" stub tutorials still carry a
``__Wrap Up__`` section, so they pass.

A second, cheaper guard flags any script whose final docstring block ends on a
colon (``:``) — the classic "a plot/code block follows" promise left dangling
by a mid-docstring cutoff.

Run from the repo root::

    python scripts/check_tutorials_complete.py

Exit status is non-zero if any tutorial fails, listing each offender.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TERMINAL_MARKERS = ("__wrap up__", "__summary__")


def final_docstring(text: str) -> str | None:
    """Return the last triple-quoted block if it sits at the end of the file."""
    blocks = list(re.finditer(r'"""(.*?)"""', text, re.DOTALL))
    if not blocks:
        return None
    last = blocks[-1]
    if text[last.end():].strip() == "":
        return last.group(1)
    return None


def check(path: Path) -> str | None:
    """Return a failure reason for a truncated-looking tutorial, else None."""
    text = path.read_text(encoding="utf-8")
    if not any(marker in text.lower() for marker in TERMINAL_MARKERS):
        return "no __Wrap Up__/__Summary__ terminal section (looks truncated)"
    trailing = final_docstring(text)
    if trailing is not None and trailing.strip().endswith(":"):
        return "final docstring ends on ':' (dangling promise of a following block)"
    return None


def main(root: str) -> int:
    scripts_dir = Path(root) / "scripts"
    files = sorted(scripts_dir.rglob("tutorial_*.py"))

    failures = [(f, reason) for f in files if (reason := check(f)) is not None]

    print(f"Checked {len(files)} tutorial scripts under {scripts_dir}.")
    if failures:
        print(f"\n{len(failures)} tutorial(s) look incomplete / truncated:\n")
        for f, reason in failures:
            print(f"  [FAIL] {f.relative_to(root)} — {reason}")
        print(
            "\nEach tutorial must end with a `__Wrap Up__` (or `__Summary__`) "
            "section. If a script is genuinely truncated, restore its lost "
            "content; if it is complete, add the terminal section."
        )
        return 1

    print("All tutorial scripts have a terminal section — none look truncated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
