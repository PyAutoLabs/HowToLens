# HowToLens

This is the **HowToLens** tutorial lecture series for `PyAutoLens`, a Python library for strong gravitational lens modeling. Tutorials teach new users what strong lensing is and how to model it from first principles.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts
  - `chapter_1_introduction/` — Grids, profiles, galaxies, ray-tracing, data, fitting
  - `chapter_2_lens_modeling/` — Non-linear searches, Bayesian inference, lens modeling
  - `chapter_3_search_chaining/` — Search chaining, prior passing, automated pipelines
  - `chapter_4_pixelizations/` — Pixelized source reconstruction, inversions, regularization
  - `chapter_optional/` — Alternative non-linear searches and advanced topics
  - `simulator/` — Simulator scripts that generate the tutorial datasets at runtime
- `notebooks/` — Jupyter notebook versions of scripts (generated from `scripts/`, do not edit directly)
- `config/` — `PyAutoLens` configuration YAML files
- `dataset/` — Empty in the repo; tutorial datasets are written here at runtime by the simulator scripts
- `output/` — Model-fit results (generated at runtime, not committed)

## Running Scripts

Scripts are run from the repository root so relative paths to `dataset/` and `output/` resolve correctly:

```bash
python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

Tutorials in chapters 1 and 2 that need a dataset invoke the relevant script in `scripts/simulator/` via `subprocess` if the dataset folder does not already exist — there is no manual simulate-then-run step.

**Integration testing / fast mode**: set `PYAUTO_TEST_MODE=1` to skip non-linear search sampling:

```bash
PYAUTO_TEST_MODE=1 python scripts/chapter_2_lens_modeling/tutorial_1_non_linear_search.py
```

**Fast smoke tests**: combine test mode with the skip flags:

```bash
PYAUTO_TEST_MODE=2 PYAUTO_SKIP_FIT_OUTPUT=1 PYAUTO_SKIP_VISUALIZATION=1 PYAUTO_SKIP_CHECKS=1 PYAUTO_FAST_PLOTS=1 python scripts/chapter_1_introduction/tutorial_7_fitting.py
```

Note: `PYAUTO_SMALL_DATASETS` is deliberately **not** used in HowToLens. Tutorials assume the full-resolution simulated datasets that the simulator scripts produce.

**Codex / sandboxed runs**: set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py
```

## Core API Patterns

Imports used throughout the tutorials:

```python
import autofit as af
import autolens as al
import autolens.plot as aplt
```

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the `.py` files in `scripts/` using `generate.py` from the `PyAutoBuild` repo. **Always edit the `.py` scripts**, never the notebooks directly. The `# %%` marker alternates between code and markdown cells.

### Building Notebooks

Run from the workspace root:

```bash
PYTHONPATH=../PyAutoBuild/autobuild python3 ../PyAutoBuild/autobuild/generate.py howtolens
```

The `howtolens` project target in `PyAutoBuild/autobuild/config.yaml` is what drives this.

## Relationship to autolens_workspace

HowToLens is the teaching companion to `autolens_workspace`. Many tutorials (particularly in chapters 2–4) point users to `autolens_workspace` scripts (e.g. `scripts/imaging/modeling.py`, `scripts/guides/...`) as the next destination after the relevant concept has been introduced. Those cross-references use absolute paths like `autolens_workspace/scripts/...` and refer to the separate `autolens_workspace` repository — not to anything inside HowToLens.

## Related Repos

- **PyAutoLens** source: `../PyAutoLens`
- **PyAutoGalaxy** source: `../PyAutoGalaxy`
- **autolens_workspace**: `../autolens_workspace` — main user-facing workspace
- **PyAutoBuild**: `../PyAutoBuild` — notebook generation and CI/CD tooling
