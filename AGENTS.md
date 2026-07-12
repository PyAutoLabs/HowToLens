# HowToLens — Agent Instructions

This is the **HowToLens** tutorial lecture series for **PyAutoLens**, a Python library for strong
gravitational lens modeling. Tutorials teach new users what strong lensing is and how to model it
from first principles. It is the teaching companion to `../autolens_workspace`. These are the
canonical, agent-agnostic instructions for this repo.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts:
  - `chapter_1_introduction/` — Grids, profiles, galaxies, ray-tracing, data, fitting
  - `chapter_2_lens_modeling/` — Non-linear searches, Bayesian inference, lens modeling
  - `chapter_3_search_chaining/` — Search chaining, prior passing, automated pipelines
  - `chapter_4_pixelizations/` — Pixelized source reconstruction, inversions, regularization
  - `chapter_optional/` — Alternative searches and advanced topics
  - `simulator/` — Simulator scripts that generate the tutorial datasets at runtime
- `notebooks/` — Jupyter versions, generated from `scripts/` (do not edit directly)
- `config/` — PyAutoLens configuration YAML
- `dataset/` — Empty in the repo; written at runtime by the simulator scripts
- `output/` — Model-fit results (generated at runtime, not committed)

## Running Scripts

Scripts are run **from the repo root** so relative paths to `dataset/` and `output/` resolve. A
tutorial that needs a dataset invokes the matching `scripts/simulator/` script via `subprocess` if
the dataset folder is absent — no manual simulate-then-run step.

```bash
python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

Fast mode for integration: `PYAUTO_TEST_MODE=1` skips sampling (`=2` also bypasses; combine with
`PYAUTO_SKIP_FIT_OUTPUT=1 PYAUTO_SKIP_VISUALIZATION=1 PYAUTO_SKIP_CHECKS=1 PYAUTO_FAST_PLOTS=1` for a
fast smoke run). **Dataset realism:** `PYAUTO_SMALL_DATASETS` is deliberately **not** used in
HowToLens — tutorials assume the full-resolution simulated datasets (unlike the workspaces, which
cap grids/masks).

## Testing

On CI, every PR is gated on Python **3.12 and 3.13** by `smoke_tests.yml` (runs
`python .github/scripts/run_smoke.py`, driven by `smoke_tests.txt` + `config/build/env_vars.yaml` —
the definition of green), `navigator_check.yml` (PyAutoBuild's reusable navigator-catalogue check;
see *Notebooks vs Scripts*), and `url_check.yml` (link checking). The smoke and navigator jobs check
out **PyAutoBuild** as a sibling and run the PyAuto* libraries from the **same-named branch** of each
source repo, so a HowToLens PR is validated against matching library branches.

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are **generated** from the `.py` scripts via PyAutoBuild. **Always edit the
`.py` scripts, never the `.ipynb` directly.** The `# %%` marker alternates code and markdown cells.
Regenerate from the repo root:

```bash
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py howtolens
```

The `howtolens` project target is registered in PyAutoBuild (`run_all.py`, `navigator.py`,
`config/`). The navigator catalogue — `llms-full.txt` + `workspace_index.json` — is what
`navigator_check.yml` gates; it is rebuilt by the same PyAutoBuild generate/merge flow that builds
the notebooks. Commit regenerated notebooks and catalogue alongside the script changes.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Scientific Context

When a tutorial benefits from broader framing — landmark surveys, named lenses, degeneracies, or the
literature behind a mass-model or source-reconstruction choice — pull from the `autolens_assistant`
literature wiki at https://github.com/PyAutoLabs/autolens_assistant (`wiki/literature/` — concepts,
entities, sources). If cloned as a sibling, read it locally at `../autolens_assistant/wiki/literature/`.

## Related Repos

- `../PyAutoLens`, `../PyAutoGalaxy` — source libraries.
- `../autolens_workspace` — the user-facing workspace (tutorials point here as the next destination).
- `../PyAutoBuild` — notebook generation + CI tooling.
- `../autolens_assistant` — science-assistant workspace (literature wiki; see *Scientific Context*).

## Task Workflows

**`[API Update]` issues:** find every renamed/moved/removed/changed public API, update each tutorial
script (preserving the teaching prose), run `python .github/scripts/run_smoke.py`, and fix `[FAIL]`
entries until the summary passes; regenerate notebooks + catalogue after. **General issues:** edit
only files in `scripts/` (never `notebooks/`), preserve docstrings and explanations, test, then
regenerate. Flag any change that affects `autolens_workspace` or the source libraries in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
<!-- repos_sync:history:end -->
