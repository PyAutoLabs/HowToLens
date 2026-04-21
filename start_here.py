"""
Start Here: HowToLens
=====================

Welcome to **HowToLens** — the `PyAutoLens` tutorial lecture series on strong gravitational lensing.

This script gives you a one-page overview of the series and points you to the first tutorial.

__HowToLens__

**HowToLens** is a four-chapter guide which takes you, step by step, from knowing nothing about strong lensing
to being able to model real strong lens data with **PyAutoLens** for scientific research.

- Chapter 1: Introduction to strong lensing — grids, light profiles, mass profiles, galaxies, ray-tracing,
  simulated data, and how fitting works.
- Chapter 2: Bayesian inference and non-linear searches — how to fit a lens model to CCD imaging data.
- Chapter 3: Search chaining — automated pipelines that fit complex systems by chaining multiple searches.
- Chapter 4: Pixelized source reconstruction — reconstructing sources with irregular morphology using
  inversions and regularization.

Each chapter is organised into numbered tutorial files: `chapter_<N>_<name>/tutorial_<M>_<topic>.py` (Python script)
or the matching `.ipynb` in `notebooks/`. Tutorials build on each other within a chapter and assume you have
completed the earlier ones.

__Recommended Path__

We recommend working through the tutorials in order:

1. First run `scripts/chapter_1_introduction/tutorial_0_visualization.py` — this checks your installation
   and sets up matplotlib for the rest of the series.
2. Work through `chapter_1_introduction` tutorial-by-tutorial.
3. Work through `chapter_2_lens_modeling`.
4. At this point you can start modeling real lenses using scripts in the separate
   `autolens_workspace` repository — you know enough to be productive.
5. Come back for `chapter_3_search_chaining` and `chapter_4_pixelizations` when you want to tackle more
   complex systems.

__Notebooks vs Scripts__

Every tutorial exists as both a Python script (under `scripts/`) and a Jupyter notebook (under `notebooks/`).
The notebooks are ideal for reading because plots render inline between small blocks of code. The Python
scripts are more convenient for actual **PyAutoLens** use — which chapter 3 onwards starts to assume.

The notebooks are auto-generated from the Python scripts, so if you want to make changes, **edit the
Python scripts, not the notebooks.**

__Next Step__

Open `scripts/chapter_1_introduction/tutorial_0_visualization.py` (or the `.ipynb` equivalent) and start there.
"""

print(__doc__)
