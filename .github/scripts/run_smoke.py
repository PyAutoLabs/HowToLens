"""
Run the workspace smoke test suite: every script under `scripts/`, minus the
exclusions in `config/build/no_run.yaml`.

Coverage is **opt-out**. A new tutorial is smoke-tested the moment it is added;
excluding one is a deliberate, documented entry in `config/build/no_run.yaml` —
the same file the notebook runner already honours, so scripts and notebooks can
no longer disagree about what is skipped.

This replaces the former `smoke_tests.txt` allowlist, under which a script was
tested only if someone remembered to add it. That design left HowToGalaxy
testing 4 of its 26 scripts, and a public teaching notebook stayed broken in
three places because no job had ever executed it (HowToGalaxy #58).

Nothing about discovery, exclusion or environment resolution is implemented
here. This is a thin shim over PyAutoHands' `autohands/run_python.py` — the same
entry point PyAutoHeart's workspace-validation uses for its `run_scripts` job —
so the PR gate and the validation runner cannot drift apart. That runner
provides:

  * recursive discovery, ordering `simulator*` first and then `start_here.py`,
    which is what tutorials depending on simulated datasets need
  * `should_skip()` against `config/build/no_run.yaml`
  * per-script env from `config/build/profile_smoke.yaml`

Mirrors the `/smoke-test` skill so CI and local runs stay in sync.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = "howtolens"

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    import build_util
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    import build_util

AUTOHANDS = Path(build_util.__file__).resolve().parent


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(AUTOHANDS), env.get("PYTHONPATH", "")) if p
    )

    # --report-dir is REQUIRED, not cosmetic. run_python.py only propagates
    # failures (`sys.exit(1)`) when a report was built; without it the suite
    # runs to completion and always exits 0 — a vacuously green gate. It also
    # switches execute_script from "abort on the first failure" to "record and
    # continue", which is the behaviour the old runner had.
    cmd = [
        sys.executable,
        str(AUTOHANDS / "run_python.py"),
        PROJECT,
        "scripts",
        "--report-dir",
        str(WORKSPACE / "test-results"),
    ]
    # run_python.py resolves config/build/ relative to the cwd.
    return subprocess.run(cmd, cwd=str(WORKSPACE), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
