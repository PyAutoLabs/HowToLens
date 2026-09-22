# HowToLens

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/start_here.ipynb)

[Start Here on Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/start_here.ipynb) |
[Installation Guide](https://pyautolens.readthedocs.io/en/latest/installation/overview.html) |
[PyAutoLens readthedocs](https://pyautolens.readthedocs.io/en/latest/index.html) |
[Browse Chapter 1 With Images](markdown/README.md) |
[autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace)

<img src="https://github.com/Jammy2211/PyAutoLogo/blob/main/gifs/pyautolens.gif?raw=true" width="900" />

Welcome to **HowToLens** — the tutorial lecture series for [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens),
an open-source library for strong gravitational lens modeling.

**PyAutoLens** can be used with an AI coding agent to compose lens models, fit data and explore results
**using natural language**. **HowToLens** teaches the core principles behind this workflow, so you
understand the science and inference being performed rather than treating them as a black box.

The tutorials assume minimal prior knowledge of astronomy or statistics. They start from first principles:
grids, light and mass profiles, and ray-tracing, then progress to Bayesian lens modeling, pixelized
source reconstructions and group- and cluster-scale lenses.

For experienced scientists who already know the fundamentals of strong lensing and Bayesian modeling, the
[autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace) examples will be more appropriate —
they are concise and assume the concepts taught in **HowToLens** as background.

## Chapters

- `chapter_1_introduction` — An introduction to strong gravitational lensing and **PyAutoLens**: grids, light
  and mass profiles, galaxies, ray-tracing, point sources, the lensing formalism, simulated imaging data,
  and fitting.
- `chapter_2_lens_modeling` — Bayesian inference, non-linear searches, and how to fit a lens model to CCD
  imaging data with **PyAutoLens**, ending with search chaining and automated pipelines.
- `chapter_3_pixelizations` — Pixelized source reconstructions (inversions) for sources with irregular
  morphologies, including the Bayesian formalism underpinning them.
- `chapter_4_scaling_up_lensing` — Scaling lens modeling up beyond a single lens galaxy: extra galaxies,
  multi-galaxy lenses, scaling relations, group and cluster scales, and weak lensing.
- `chapter_optional` — Optional tutorials on alternative non-linear searches and other advanced topics.

**HowToLens** currently sits at four chapters. Each chapter will take around a day to work through.
We recommend completing chapters 1 and 2, then applying what you've learned to real lens modeling in the
`autolens_workspace` before returning for the more advanced material in chapters 3 and 4.

## Getting Started

### Study with the assistant

Use the [Jupyter notebooks](notebooks/) to run the code (recommended), or read the
available [Markdown lectures](markdown/README.md) directly on GitHub.

For help alongside the lectures, open the [autolens_assistant](https://github.com/PyAutoLabs/autolens_assistant)
repository in your AI coding agent, following its setup instructions, and paste:

```text
Enter teacher mode.

I want to work through the HowToLens lectures. Show me where to find them
and how to use Jupyter Notebook or Markdown, then help me with questions
as I go.
```

The assistant can answer questions about concepts, equations, code and results as you study, and help with
notebook errors. Share the lecture link and section or the cell you are working on; you choose when to move on.

### Run in Google Colab (nothing to install)

Every tutorial opens in Google Colab in one click. There is nothing to install on your own machine and
no local Python environment to set up — **PyAutoLens** installs itself in the notebook's first cell.
In Colab you *run* the tutorial: edit the code, change the model, and see the output for yourself.

Whilst in Colab, you can use **Gemini** as a study assistant alongside the lecture: ask it to explain an
equation, unpack what a cell is doing, or help interpret the output of a fit.

The `markdown` links are the same tutorial already executed and rendered on GitHub, with its real
output figures inline. Nothing runs and nothing installs — you just read it. They are good for
skimming a tutorial before running it, or for reading on a phone. Markdown pages currently exist only
for the chapter 1 tutorials listed with a `markdown` link below; every other tutorial is Colab-only.

**[Start Here](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/start_here.ipynb)** — a one-page overview of the whole series.

- **[Chapter 1: Introduction](scripts/chapter_1_introduction/README.md)** — Grids, light and mass profiles, galaxies, ray-tracing, point sources, the lensing formalism, data, and fitting.
  - Tutorial 0: Visualization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_0_visualization.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_0_visualization.md))
  - Tutorial 1: Grids and Galaxies — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_1_grids_and_galaxies.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_1_grids_and_galaxies.md))
  - Tutorial 2: Ray Tracing — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_2_ray_tracing.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_2_ray_tracing.md))
  - Tutorial 3: More Ray Tracing — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_3_more_ray_tracing.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_3_more_ray_tracing.md))
  - Tutorial 4: Point Sources — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_4_point_sources.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_4_point_sources.md))
  - Tutorial 5: Lensing Formalism — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_5_lensing_formalism.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_5_lensing_formalism.md))
  - Tutorial 6: Data — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_6_data.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_6_data.md))
  - Tutorial 7: Fitting — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_7_fitting.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_7_fitting.md))
  - Tutorial 8: Summary — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_1_introduction/tutorial_8_summary.ipynb) / [markdown](markdown/chapter_1_introduction/tutorial_8_summary.md))
- **[Chapter 2: Lens Modeling](scripts/chapter_2_lens_modeling/README.md)** — Bayesian inference, non-linear searches, lens modeling, search chaining, prior passing, and SLaM pipelines.
  - Tutorial 1: Non-linear Search — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_1_non_linear_search.ipynb))
  - Tutorial 2: Practicalities — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_2_practicalities.ipynb))
  - Tutorial 3: Realism and Complexity — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_3_realism_and_complexity.ipynb))
  - Tutorial 4: Dealing with Failure — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_4_dealing_with_failure.ipynb))
  - Tutorial 5: Linear Profiles — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_5_linear_profiles.ipynb))
  - Tutorial 6: Masking and Positions — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_6_masking_and_positions.ipynb))
  - Tutorial 7: Results — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_7_results.ipynb))
  - Tutorial 8: Need for Speed — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_8_need_for_speed.ipynb))
  - Tutorial 9: Search Chaining — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_9_search_chaining.ipynb))
  - Tutorial 10: Prior Passing — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_10_prior_passing.ipynb))
  - Tutorial 11: SLaM — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_2_lens_modeling/tutorial_11_slam.ipynb))
- **[Chapter 3: Pixelizations](scripts/chapter_3_pixelizations/README.md)** — Pixelized source reconstructions, inversions, the Bayesian formalism, and adaptive pixelizations and regularization.
  - Tutorial 1: Pixelizations — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb))
  - Tutorial 2: Mappers — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_2_mappers.ipynb))
  - Tutorial 3: Inversions — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_3_inversions.ipynb))
  - Tutorial 4: Bayesian Regularization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_4_bayesian_regularization.ipynb))
  - Tutorial 5: Bayesian Formalism — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_5_bayesian_formalism.ipynb))
  - Tutorial 6: Borders — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_6_borders.ipynb))
  - Tutorial 7: Lens Modeling — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_7_lens_modeling.ipynb))
  - Tutorial 8: Adaptive Pixelization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_8_adaptive_pixelization.ipynb))
  - Tutorial 9: Model Fit — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_9_model_fit.ipynb))
  - Tutorial 10: Fit Problems — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_10_fit_problems.ipynb))
  - Tutorial 11: Brightness Adaption — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_11_brightness_adaption.ipynb))
  - Tutorial 12: Adaptive Regularization — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_3_pixelizations/tutorial_12_adaptive_regularization.ipynb))
- **[Chapter 4: Scaling Up Lensing](scripts/chapter_4_scaling_up_lensing/README.md)** — Extra galaxies, multi-galaxy lenses, scaling relations, group and cluster scales, and weak lensing.
  - Tutorial 1: Extra Galaxies — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_1_extra_galaxies.ipynb))
  - Tutorial 2: Multi Galaxy — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_2_multi_galaxy.ipynb))
  - Tutorial 3: Scaling Relation — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_3_scaling_relation.ipynb))
  - Tutorial 4: Group Scale — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_4_group_scale.ipynb))
  - Tutorial 5: Cluster Scale — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_5_cluster_scale.ipynb))
  - Tutorial 6: Weak Lensing — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_4_scaling_up_lensing/tutorial_6_weak_lensing.ipynb))
- **[Optional Tutorials](scripts/chapter_optional/)** — Alternative non-linear searches and advanced topics.
  - Tutorial Optional: Searches — ([Colab](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/2026.9.19.1/notebooks/chapter_optional/tutorial_searches.ipynb))

Model-fits can run faster on a GPU. In Colab, enable one via *Runtime* → *Change runtime type*
→ *Hardware accelerator* before running a notebook.

### Run on your own machine

You can run the tutorials on your own machine by following the
[PyAutoLens installation guide](https://pyautolens.readthedocs.io/en/latest/installation/overview.html),
then cloning this repository:

```bash
git clone https://github.com/PyAutoLabs/HowToLens.git
cd HowToLens
```

The tutorials are distributed as both Jupyter notebooks (`notebooks/`) and Python scripts (`scripts/`).
We recommend the notebooks for reading — images and plots render inline, and you can step through small
code blocks interactively. Use the Python scripts for actual **PyAutoLens** use, which is the workflow
chapter 3 onwards transitions you to.

## Before Chapter 1

Before starting chapter 1, complete `scripts/chapter_1_introduction/tutorial_0_visualization.py`
(or the equivalent notebook). This confirms your **PyAutoLens** installation, walks you through how images
and figures display in Jupyter, and configures matplotlib for the rest of the tutorial series.

## Lensing Theory

**HowToLens** assumes minimal previous knowledge of gravitational lensing. It is helpful to have the following
lecture course on gravitational lensing by Massimo Meneghetti open as you go through the tutorials:

<http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf>

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts, one subfolder per chapter.
- `notebooks/` — Jupyter notebook versions of the scripts (auto-generated; see below).
- `config/` — **PyAutoLens** configuration YAML files used by the tutorials.
- `dataset/` — Tutorial datasets are generated at runtime by scripts in `scripts/simulator/` —
  no `.fits` files are committed.
- `output/` — Model-fit results (generated at runtime, not committed).

## Notebooks vs Scripts

Notebooks in `notebooks/` are generated from the Python files in `scripts/`. **Always edit the \`\`.py\`\`
scripts, never the notebooks directly.** The `# %%` markers in each script alternate between code and
markdown cells, which [PyAutoHands](https://github.com/PyAutoLabs/PyAutoHands) uses to produce the
`.ipynb` files.

## Relationship to autolens_workspace

[autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace) is the main user-facing workspace
for **PyAutoLens** — concise examples, guides, and science templates aimed at users who have a working
understanding of strong lensing. **HowToLens** is the teaching companion. Many tutorials in chapters 2–4
reference `autolens_workspace` scripts as the next place to go after the relevant concept has been
introduced.

## Citations

If you use **HowToLens** or **PyAutoLens** in your research, please cite the references listed in
`CITATIONS.rst`.

## Community & Support

Support for **PyAutoLens** is available via our Slack workspace. Slack is invitation-only; send an email
if you'd like an invite.

For installation issues, bug reports, or feature requests, raise an issue on the
[PyAutoLens GitHub issues page](https://github.com/PyAutoLabs/PyAutoLens/issues) (for library issues)
or the [HowToLens GitHub issues page](https://github.com/PyAutoLabs/HowToLens/issues) (for tutorial
content issues).
