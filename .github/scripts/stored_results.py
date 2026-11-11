#!/usr/bin/env python3
"""Maintain the stored Chapter 2 fit results that let readers skip sampling.

The repository ships the completed output of every Chapter 2 search after
Tutorial 1, under ``output/howtolens/chapter_2``. When a reader runs a
tutorial, PyAutoFit finds the ``.completed`` marker for that search's
identifier and loads the stored result in seconds ("Fit Already Completed:
skipping non-linear search") instead of running a search lasting tens of
minutes. Tutorial 1 is not stored: its search has no ``name`` or
``path_prefix``, so it writes no output and is each reader's one live fit.

Each stored fit keeps only the files PyAutoFit needs to rebuild the result:

* ``.completed`` and ``.identifier`` (the fit is finished, and for which setup);
* ``files/samples_summary.json`` (the maximum-likelihood and median-PDF models);
* ``files/samples_info.json`` (evidence and the recorded run time);
* ``files/model.json`` and ``files/search.json`` (model and search settings);
* ``files/samples.csv``, thinned to a few equal-weight draws so that
  ``result.samples`` exists for the cells that use it (``CORNER_PLOT_SAMPLES``
  keeps more for the fits whose corner plots the tutorials interpret);
* ``search.summary`` (the human-readable timing summary, when the search wrote one).

Tutorial 2 walks the reader through a real output folder, so its fit keeps
the full folder instead, minus the large search corner-plot images.

The output identifier is a hash of the model and the identifying search
settings. A library change to a profile's module path, a default prior or a
search default moves it, after which a tutorial silently samples again.
``check`` catches that.

Run from the repo root::

    python .github/scripts/stored_results.py harvest <full-output-root>
    python .github/scripts/stored_results.py check

``harvest`` copies from a directory holding complete, unthinned runs (the
``output`` folder of a regeneration run with test mode off) into
``output/howtolens/chapter_2``. New files are gitignored with the rest of
``output/``; add them with ``git add -f output/howtolens/chapter_2``.

``check`` runs Tutorials 2-6 with test mode off and a search that refuses to
sample. Exit status is non-zero if any tutorial would run a search.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STORED_ROOT = Path("output") / "howtolens" / "chapter_2"

MINIMAL_FILES = (
    ".completed",
    ".identifier",
    "search.summary",
    "files/model.json",
    "files/search.json",
    "files/samples_info.json",
    "files/samples_summary.json",
)

# Not every completed search writes its summary; the run time is also in
# samples_info.json, which is always kept.
OPTIONAL_FILES = ("search.summary",)

ILLUSTRATED_FIT = "tutorial_2_practicalities"
ILLUSTRATION_EXCLUDES = ("image/search", "files/search_internal")

THIN_SAMPLES = 10
THIN_SEED = 1

# Fits whose corner plots the tutorials ask readers to interpret keep enough
# draws for the posterior's shape to be visible.
CORNER_PLOT_SAMPLES = {
    "tutorial_2_practicalities": 1000,
    "tutorial_6_search_2": 1000,
}

CHECKED_TUTORIALS = (
    "tutorial_2_practicalities.py",
    "tutorial_3_why_modeling_is_hard.py",
    "tutorial_4_linear_profiles.py",
    "tutorial_5_data_and_likelihoods.py",
    "tutorial_6_search_chaining.py",
)

REFUSE_TO_SAMPLE = """
import runpy, sys
import autofit as af

def refuse(self, *args, **kwargs):
    raise SystemExit(f"SAMPLED: {self.paths.output_path} has no stored result")

af.NonLinearSearch.start_resume_fit = refuse
runpy.run_path(sys.argv[1], run_name="__main__")
"""


def thin_samples(source: Path, destination: Path, size: int) -> None:
    """
    Write ``size`` draws of ``source`` as equal-weight samples.

    Posterior samples are drawn in proportion to their weights. An optimiser's
    samples carry no usable weights, so its highest-likelihood rows are kept.
    """
    import numpy as np
    import pandas as pd

    samples = pd.read_csv(source, skipinitialspace=True)
    samples.columns = [column.strip() for column in samples.columns]

    # A failed search can put most of its weight on elliptical components of
    # magnitude >= 1, which the model refuses to rebuild. PyAutoFit then falls
    # back through the stored samples, so keep only rows it can rebuild, and
    # always the best of them.
    valid = np.ones(len(samples), dtype=bool)
    for column in samples.columns:
        if column.endswith("ell_comps_0"):
            other = column[: -len("0")] + "1"
            valid &= np.hypot(samples[column], samples[other]).to_numpy() < 1.0
    if valid.any():
        samples = samples[valid].reset_index(drop=True)
    best = int(samples["log_likelihood"].to_numpy().argmax())

    weights = samples["weight"].to_numpy() if "weight" in samples else np.zeros(1)
    if np.isfinite(weights).all() and weights.sum() > 0:
        weights = weights / weights.sum()
        size = min(size, int(np.count_nonzero(weights)))
        rows = np.random.default_rng(THIN_SEED).choice(
            len(samples), size=size, replace=False, p=weights
        )
    else:
        rows = samples["log_likelihood"].to_numpy().argsort()[-size:]

    rows = np.union1d(rows, [best])
    thinned = samples.iloc[np.sort(rows)].copy()
    thinned["weight"] = 1.0 / len(thinned)
    thinned.to_csv(destination, index=False)


def harvest_fit(fit: Path, destination: Path) -> None:
    """Copy one completed fit's stored files into ``destination``."""
    if destination.exists():
        shutil.rmtree(destination)

    if fit.parent.name == ILLUSTRATED_FIT:
        shutil.copytree(
            fit,
            destination,
            ignore=lambda folder, names: [
                name
                for name in names
                if str((Path(folder) / name).relative_to(fit)) in ILLUSTRATION_EXCLUDES
            ],
        )
    else:
        for name in MINIMAL_FILES:
            if name in OPTIONAL_FILES and not (fit / name).exists():
                continue
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fit / name, destination / name)

    thin_samples(
        fit / "files" / "samples.csv",
        destination / "files" / "samples.csv",
        size=CORNER_PLOT_SAMPLES.get(fit.parent.name, THIN_SAMPLES),
    )


def harvest(source_root: Path) -> int:
    source = source_root / "howtolens" / "chapter_2"
    fits = sorted(marker.parent for marker in source.rglob(".completed"))
    if not fits:
        print(f"No completed fits under {source}.")
        return 1

    for fit in fits:
        destination = REPO / STORED_ROOT / fit.relative_to(source)
        harvest_fit(fit, destination)
        size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
        print(f"{fit.relative_to(source)}  {size / 1024:.0f} KB")

    print(f"\nHarvested {len(fits)} fits. Track them with:")
    print(f"  git add -f {STORED_ROOT}")
    return 0


def check() -> int:
    env = {key: value for key, value in os.environ.items() if key != "PYAUTO_TEST_MODE"}
    env.setdefault("MPLBACKEND", "Agg")

    # A refused search still writes its setup files first; remove those folders
    # afterwards so a failed check leaves no half-made results behind.
    stored = REPO / STORED_ROOT
    before = {path for path in stored.glob("*/*/*")} if stored.exists() else set()

    failures = []
    for name in CHECKED_TUTORIALS:
        script = Path("scripts") / "chapter_2_lens_modeling" / name
        completed = subprocess.run(
            [sys.executable, "-c", REFUSE_TO_SAMPLE, str(script)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print(f"OK      {name}")
        else:
            tail = (completed.stdout + completed.stderr).strip().splitlines()
            print(f"FAILED  {name}: {tail[-1] if tail else completed.returncode}")
            print("\n".join(f"        {line}" for line in tail[-20:]))
            failures.append(name)

    for path in (set(stored.glob("*/*/*")) - before) if stored.exists() else ():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        if not any(path.parent.iterdir()):
            path.parent.rmdir()

    if failures:
        print(
            f"\n{len(failures)} tutorial(s) would run a search. Regenerate the stored "
            "results (see this script's docstring)."
        )
        return 1
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["harvest"] and len(argv) == 2:
        return harvest(Path(argv[1]))
    if argv == ["check"]:
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
