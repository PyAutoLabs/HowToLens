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

# pyauto:smoke-bootstrap:begin
# Generated from PyAutoMind/policy/smoke_bootstrap.py; edit the source, not copies.
# CI supplies build_util on PYTHONPATH. Discovery is only a local fallback.
try:
    import build_util
except ModuleNotFoundError as _smoke_import_error:
    if _smoke_import_error.name != "build_util":
        raise  # An installed Hands dependency is broken, not missing Hands.

    def _smoke_hands_path():
        import importlib.util

        explicit = os.environ.get("PYAUTO_ROOT")
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            root = next(
                (p for p in WORKSPACE.parents if (p / ".pyauto-root").is_file()),
                WORKSPACE.parent,
            )
            # Prefer the shared resolver, but never require Brain. A bundle's
            # Brain may be a symlink to canonical; reject that foreign answer.
            resolvers = [p for p in (
                root / "PyAutoBrain" / "agents" / "_pyauto_root.py",
                root / "organs" / "PyAutoBrain" / "agents" / "_pyauto_root.py",
            ) if p.is_file()]
            if len({p.resolve() for p in resolvers}) > 1:
                raise RuntimeError("PyAutoBrain has distinct flat and grouped checkouts")
            resolver_file = resolvers[0] if resolvers else None
            if resolver_file is not None:
                try:
                    spec = importlib.util.spec_from_file_location(
                        "_smoke_root_resolver", resolver_file
                    )
                    resolver = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(resolver)
                    resolved = Path(resolver.workspace_root_reason()[0]).resolve()
                    if resolved in WORKSPACE.parents:
                        root = resolved
                except (Exception, SystemExit):
                    pass  # Missing/broken optional resolver cannot break discovery.
        flat = root / "PyAutoHands" / "autohands"
        grouped = root / "organs" / "PyAutoHands" / "autohands"
        if (flat / "build_util.py").is_file() and (grouped / "build_util.py").is_file() and flat.resolve() != grouped.resolve():
            raise RuntimeError("PyAutoHands has distinct flat and grouped checkouts")
        hands = grouped if (grouped / "build_util.py").is_file() else flat
        if not (hands / "build_util.py").is_file():
            raise ModuleNotFoundError(
                f"PyAutoHands not found for {WORKSPACE}; looked for "
                f"{hands / 'build_util.py'}. Supply PyAutoHands/autohands on "
                "PYTHONPATH or set PYAUTO_ROOT to the intended workspace root.",
                name="build_util",
            )
        return hands

    sys.path.insert(0, str(_smoke_hands_path()))
    import build_util

AUTOHANDS = Path(build_util.__file__).resolve().parent
# pyauto:smoke-bootstrap:end


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
