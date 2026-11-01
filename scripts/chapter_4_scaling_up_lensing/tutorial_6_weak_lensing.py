"""
Tutorial 6: Weak Lensing
========================

Our modeling tutorials have concentrated on **strong lensing**: the multiple images, arcs and Einstein rings produced by galaxy, group and cluster lenses.

In the previous tutorial, we used the positions of multiply imaged sources to constrain a cluster’s mass distribution. Those images provide particularly valuable information in the central region where they form. To measure the mass distribution over a much wider region, we can also use **weak lensing**.

The same gravitational field distorts the shapes of background galaxies beyond the region containing prominent arcs. These distortions are usually difficult to distinguish from a galaxy’s unknown intrinsic shape: an elongated galaxy might have been elongated even without lensing.

The signal becomes measurable statistically. Nearby background galaxies experience a coherent lensing distortion, while their intrinsic orientations contribute scatter. Measuring many galaxies allows us to estimate that coherent signal.

This scatter is called **shape noise**. Increasing the number of usable background galaxies reduces its contribution to the uncertainty, although observational biases and correlations between intrinsic galaxy shapes also require attention in real analyses.

You first encountered a weak-lensing dataset in Chapter 1. Here we return to its **shear catalogue**: background-galaxy positions, two components describing the estimated distortion, and their uncertainties.

We simulate a catalogue around a cluster-scale lens, visualize it, inspect a fit using the input mass model, and then infer the model parameters with a non-linear search. The familiar `Tracer`, `Fit`, `Analysis` and search workflow remains central.

This tutorial closes both this chapter and the four core chapters of **HowToLens**.

__Contents__

- **Shear Catalogues:** The measurements used in a weak-lensing analysis.
- **Mass Scales:** Weak lensing around galaxies, groups and clusters.
- **Ray Tracing:** Build a simple cluster-scale mass model.
- **Source Galaxy Positions:** Sample background galaxies across the observed field.
- **Simulate:** Generate a shear catalogue with shape noise.
- **Visualize:** Inspect the shear field and dataset summary.
- **Mass Map:** Reconstruct a convergence map from the catalogue.
- **Fitting:** Inspect the likelihood and residuals for a specified mass model.
- **Model Fit:** Infer mass-model parameters with a non-linear search.
- **Result:** Inspect the inferred model and tangential shear profile.
- **Joint Strong and Weak Lensing:** Combine constraints across different radii.
- **Wrap Up:** Complete the core HowToLens lectures.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

r"""
__Shear Catalogues__

A shear catalogue contains one row per background galaxy, including:

- Its observed sky position, expressed here as `(y, x)` offsets in arc-seconds.
- Two components describing its estimated lensing distortion.
- An uncertainty that accounts for intrinsic shape dispersion and measurement noise.

Real catalogues also require information about source redshifts, selection and calibration. The lensing signal depends on the geometry between the observer, lens and source, just as it did for the multiple source redshifts in the cluster tutorial.

Producing these measurements begins with telescope images. Galaxies must be detected and selected, and their shapes measured while accounting for the PSF, noise and instrumental effects. The PSF can both blur galaxy shapes and introduce coherent distortions that could be mistaken for lensing.

We start after this shape-measurement stage. The dataset supplied to this tutorial contains catalogue measurements rather than image pixels, so our likelihood does not convolve a model image with a PSF. This does not make observational systematics disappear: their calibration and uncertainty remain part of a real weak-lensing analysis.

There is also an important distinction between **shear** and **reduced shear**. Galaxy-shape measurements generally estimate the reduced shear:

\[
g = \frac{\gamma}{1-\kappa}.
\]

When the convergence is sufficiently small, reduced shear is approximately equal to shear. This tutorial uses an idealized catalogue of shear measurements generated directly from the model. When working with observed catalogues, check which quantity the measurements estimate and ensure that the model predicts the same quantity.

__Mass Scales__

Weak lensing is useful across the mass scales explored in this chapter.

For individual galaxy lenses, the available background galaxies usually provide too little information for a precise halo measurement. **Galaxy–galaxy lensing** therefore commonly combines measurements around many foreground galaxies to constrain the average mass distribution of a population.

Groups produce larger signals, but the precision obtainable for an individual system depends on its mass, redshift and the depth and quality of the observations. Combining groups can also improve the measurement.

Massive clusters can produce measurable coherent distortions across hundreds or thousands of background galaxies. These measurements extend well beyond the central arcs and help constrain the cluster’s extended dark matter halo.

There is no sharp Einstein-radius boundary separating these applications. Detectability depends on the lensing geometry, the radii sampled, the number of usable sources and their shape uncertainties.

We use a cluster-scale lens here because it provides a useful demonstration of an individual-system weak-lensing analysis.

__Ray Tracing__

We begin with a `Tracer`, using an elliptical `Isothermal` mass profile with an Einstein radius of 25.0 arc-seconds.

This is a deliberately simple cluster-scale mass distribution. Real cluster analyses often use NFW dark matter halos and additional mass components, as discussed in the previous tutorial.

The lens has no light profile because this fit uses catalogue measurements rather than an image of the lens.

The source galaxy also has no light profile. It supplies the source-plane redshift needed to calculate the lensing geometry; it does not represent the detailed appearance of an individual background galaxy.

For simplicity, all background galaxies in this simulation lie at redshift 1.0, behind a lens at redshift 0.5.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=25.0,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Source Galaxy Positions__

We draw 1500 observed background-galaxy positions in an annulus extending from 50 to 200 arc-seconds from the lens centre, distributed uniformly in area.

This excludes the central region containing the strongest lensing features. In real observations, the usable region also depends on blending, cluster-member contamination and the reliability of the shape measurements.

The annulus covers approximately 33 square arc-minutes, giving a source density of about 46 galaxies per square arc-minute. This represents a deep illustrative catalogue; the usable density in an observation depends on its depth, resolution and selection criteria.

For comparison, a spherical isothermal lens with an Einstein radius of 25 arc-seconds has shear and convergence of 0.25 at 50 arc-seconds, falling to about 0.06 at 200 arc-seconds. Our elliptical model varies with position angle as well.

The inner boundary is therefore outside the Einstein radius, but it does not guarantee that reduced shear is approximately equal to shear. For the spherical example, the reduced shear at 50 arc-seconds is about 0.33. Remember that our simulation generates shear directly rather than simulating observed galaxy ellipticities.

The positions are stored in an `al.Grid2DIrregular`, using the same `(y, x)` coordinate convention as the point-source tutorials. These are observed image-plane positions at which we evaluate the lensing field, not reconstructed source-plane coordinates.
"""
rng = np.random.default_rng(1)

n_galaxies = 1500
radius_inner = 50.0  # arc-seconds — excludes the central region of the simulated lens.
radius_outer = 200.0  # arc-seconds — the edge of our simulated weak-lensing field.

radii = np.sqrt(rng.uniform(radius_inner**2.0, radius_outer**2.0, n_galaxies))
phi = rng.uniform(0.0, 2.0 * np.pi, n_galaxies)

positions = al.Grid2DIrregular(
    values=np.stack([radii * np.sin(phi), radii * np.cos(phi)], axis=1)
)

"""
__Simulate__

`SimulatorShearYX` evaluates the tracer’s shear field at each catalogue position and adds independent Gaussian noise to its two components.

We use `noise_sigma=0.25` per component as an illustrative shape-noise dispersion. The intrinsic shapes of real galaxies produce substantial scatter, and measurement errors add further uncertainty.

This is a simplified statistical simulation. We do not generate images of individual background galaxies or run a shape-measurement pipeline. Instead, we directly generate noisy estimates of the shear field that our likelihood will fit.

The simulator’s default configuration generates shear rather than reduced shear, matching the idealization described above.

We keep the catalogue in memory and pass it directly to the fitting tools. The script `scripts/simulator/weak_lensing.py`, used for the weak-lensing example in Chapter 1, demonstrates saving a catalogue to disk.
"""
dataset_name = "weak_lensing"

simulator = al.SimulatorShearYX(noise_sigma=0.25, seed=1)

dataset = simulator.via_tracer_from(
    tracer=tracer,
    grid=positions,
    name=dataset_name,
)

print(dataset.info)

"""
__Visualize__

We display the catalogue as a **whisker map**. Each galaxy position has a line segment whose length represents the magnitude of its measured shear estimate and whose orientation represents the corresponding position angle.

The segments have no arrowheads. A shear orientation is unchanged by a rotation of 180 degrees, so a line represents it more appropriately than a directed arrow.

Around a centrally concentrated lens, the coherent pattern is predominantly tangential: the whiskers tend to follow circles around the lens centre. Ellipticity and additional mass structure can modify that pattern.

Shape noise introduces substantial scatter in both the lengths and orientations of the plotted whiskers. Individual measurements need not align with the underlying field, even when the ensemble constrains it well.

The plot provides an initial view of the measurements. Later, we will compare them with a model and summarize their tangential component in radial bins.
"""
aplt.plot_shear_yx_2d(shear_yx=dataset.shear_yx)

"""
`aplt.subplot_weak_dataset` summarises the full dataset as a 2x2 mosaic: the whisker map, the per-galaxy noise-map,
the shear magnitude |gamma| and the position angle phi at every galaxy.
"""
aplt.subplot_weak_dataset(dataset=dataset)

"""
__Mass Map__

A remarkable property of weak lensing is that the shear catalogue can be inverted directly into a map of the
convergence `kappa` — the dimensionless projected mass density we have used throughout the series — *without
assuming any mass model at all*. In Fourier space, shear and convergence are related algebraically, so two FFTs
turn the catalogue into a "dark matter map". This is the Kaiser-Squires technique, and it produced some of the most
famous images in cosmology, such as the mass map of the Bullet Cluster showing dark matter offset from the
colliding gas.

`aplt.plot_convergence_map` bins the catalogue onto a regular grid, applies a small Gaussian smoothing (per-cell
shears are shape-noise dominated) and plots the reconstruction. For our simulated cluster it peaks at the lens
centre at (0.0", 0.0"). Two caveats: the map's mean is unconstrained (the mass-sheet degeneracy, which chapter 3
met in its strong-lensing form) and FFT periodicity produces edge artefacts — for quantitative masses we fit a
mass model, which is what the rest of this tutorial does.
"""
aplt.plot_convergence_map(
    shear_yx=dataset.shear_yx,
    shape_native=(30, 30),
    smoothing_sigma_pixels=1.0,
)

"""
__Fitting__

We now fit the catalogue, following the same pattern as every fit in this series: build a model `Tracer`, pass it
with the dataset to a `Fit` object, and inspect residuals and the log likelihood. For weak lensing the fit object
is `FitWeak`, and because we simulated the data ourselves we can hand it the *true* tracer and see what a perfect
model looks like.

`FitWeak` evaluates the model tracer's shear field at the dataset's galaxy positions and compares it with the
measured shears, assuming each component is independently Gaussian-distributed around the model with the
per-galaxy noise. Each galaxy contributes **two** independent data points (gamma_1 and gamma_2), so the number of
degrees of freedom is 2 * n_galaxies, and for a good fit whose residuals are pure shape noise the chi-squared
should be close to that number.

Compare this to the imaging fits of chapter 1: there, the fit convolved a model image with the PSF and compared
tens of thousands of pixels; here there is no convolution, no mask and only a few thousand numbers. A weak-lensing
likelihood is orders of magnitude cheaper — which is precisely why it is so inexpensive to add to a strong-lensing
analysis, as discussed at the end of this tutorial.
"""
fit = al.FitWeak(dataset=dataset, tracer=tracer)

print()
print("Fit Summary (true model)")
print("------------------------")
print(f"n_galaxies        : {dataset.n_galaxies}")
print(f"degrees_of_freedom: {2 * dataset.n_galaxies}")
print(f"chi_squared       : {fit.chi_squared:.3f}")
print(f"log_likelihood    : {fit.log_likelihood:.3f}")

"""
As always, an incorrect mass model produces a worse fit. Halving the Einstein radius halves the predicted shear
everywhere, leaving coherent tangential residuals across the field, and the log likelihood drops accordingly.
"""
tracer_wrong = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=12.5,
                ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
            ),
        ),
        source_galaxy,
    ]
)

fit_wrong = al.FitWeak(dataset=dataset, tracer=tracer_wrong)

print()
print(f"log_likelihood (true model) : {fit.log_likelihood:.3f}")
print(f"log_likelihood (wrong model): {fit_wrong.log_likelihood:.3f}")

"""
`aplt.subplot_fit_weak` visualises a weak-lensing fit as a 2x2 mosaic: the observed shear field, the model shear
field, the two overlaid on one axes (data in black, model in red), and the per-galaxy chi-squared map. For the true
model the overlay's residuals are short and randomly oriented — the shape-noise floor — and the chi-squared map is
spatially uniform.
"""
aplt.subplot_fit_weak(fit=fit)

"""
__Model Fit__

In a real analysis we do not know the mass model — we infer it. The workflow is the one you have used since
chapter 2: compose a model with `af.Model` and `af.Collection`, choose a non-linear search, create an `Analysis`
object and fit. For weak lensing the analysis object is `AnalysisWeak`, whose `log_likelihood_function` builds a
`Tracer` from each sampled parameter set, evaluates its shear field at the catalogue positions and returns the
Gaussian log likelihood — the same `FitWeak` machinery we just used by hand.

The model is the lens's `Isothermal` mass [5 parameters]; the source galaxy again carries no components. Note the
tiny parameter space — no lens light, no source light, no shear-as-nuisance-parameter (the shear field *is* the
data here) — compared to the N=20+ models of the imaging tutorials.

One adjustment is needed, familiar from the group and cluster tutorials: the default priors are tuned for
galaxy-scale lenses (Einstein radii of 0"-8", centres within a fraction of an arc-second). Our cluster's 25"
Einstein radius lies entirely outside that prior, so we widen the priors to cluster scales — always match your
priors to the mass scale of the system you are fitting.
"""
mass = af.Model(al.mp.Isothermal)
mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=60.0)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=20.0)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=20.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free,
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed
parameter.

In weak lensing the source galaxies are the *data*, a catalogue of measured shapes, so there is no source light
to fit and the source galaxy in this model carries no light profile at all.
"""
af.ModelPlotter(model).figure()

"""
We fit the model with the nested sampling algorithm Nautilus, as throughout the series. A weak-lensing likelihood
evaluation takes milliseconds (one shear-field evaluation at 1500 positions plus a chi-squared sum), so this
5-parameter fit completes in minutes on an ordinary CPU — a refreshing contrast to the imaging fits of the earlier
chapters.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_4"),
    name="tutorial_6_weak_lensing",
    unique_tag=dataset_name,
    n_live=100,
    iterations_per_quick_update=5000,
)

analysis = al.AnalysisWeak(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the output/howtolens/chapter_4/tutorial_6_weak_lensing"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Result__

The result's `info` attribute confirms the search recovers the input mass model — Einstein radius, ellipticity and
centre — from nothing but the statistically-averaged shapes of background galaxies. No arc, no ring, no multiple
images: the ensemble alone constrains the cluster's mass.
"""
print(result.info)

aplt.subplot_fit_weak(fit=result.max_log_likelihood_fit)

"""
Cluster weak-lensing measurements are conventionally presented as the **tangential shear profile**: the mean
tangential shear gamma_t of background galaxies in radial bins about the lens centre, which traces the projected
mass profile. `aplt.plot_shear_profile` bins our catalogue and, given the fit, overlays the model's profile as a
line.

The plot also shows the *cross* component gamma_x — the shear rotated by 45 degrees. Gravitational lensing produces
no cross component at leading order, so cross points scattering around zero is the standard "B-mode" null test of a
weak-lensing measurement: a systematic contaminating the shapes would show up there.
"""
aplt.plot_shear_profile(
    result.max_log_likelihood_fit,
    centre=(0.0, 0.0),
    bins=8,
)

"""
__Joint Strong and Weak Lensing__

We fitted this catalogue on its own, and weak-lensing-only fits of real catalogues are a fully supported PyAutoLens
workflow. But the way PyAutoLens really expects weak lensing to be used is **jointly with strong lensing**, because
the two probes are perfectly complementary:

 - **Strong lensing** (everything before this tutorial) pins the mass in the inner region with exquisite precision —
   the positions of multiple images and arcs nail the Einstein mass and the inner density profile — but says
   nothing beyond the Einstein radius.

 - **Weak lensing** (this tutorial) constrains the mass profile out to arc-minute radii — the outskirts of the dark
   matter halo — but its centre and inner profile are only weakly pinned by the field's geometry.

Fit both datasets simultaneously with a single mass model — one analysis for the imaging or point-source data, one
`AnalysisWeak` for the shear catalogue, combined exactly as the multi-dataset fits earlier in this chapter combined
their analyses — and the model is anchored at the centre by strong lensing while its outer profile is measured by
weak lensing. This hybrid is the state of the art for cluster mass measurements, used across systems like the
Frontier Fields clusters, and PyAutoLens's shared `Tracer` infrastructure makes it natural: the same mass profiles
that ray-traced arcs in the previous tutorial produce the shear field fitted here.

The workspace's `autolens_workspace/*/weak` package has the complete treatment:

 - `weak/start_here.py`: fits a real shear catalogue of the merging cluster Abell 2744 ("Pandora's Cluster") —
   download, quality cuts, mass map, NFW halo fit.
 - `weak/simulator.py`, `weak/fit.py`, `weak/modeling.py`: the disk-based versions of this tutorial's workflow.
 - `weak/likelihood_function.py`: a step-by-step walkthrough of the weak-lensing likelihood.
 - `weak/features/strong_lensing`: the joint strong-plus-weak fits described above.

__Wrap Up__

This tutorial completed the chapter's journey up the mass ladder, and with it the four core chapters of
**HowToLens**. Lets take stock of how far we have come.

Chapter 1 built lensing from first principles: grids, light and mass profiles, ray tracing, the lens equation, and
fitting imaging data with a `Tracer`. Chapter 2 turned fitting into inference — non-linear searches, priors,
Bayesian model comparison — so we could *infer* lens models rather than guess them. Chapter 3 replaced analytic
sources with pixelized reconstructions, letting the data itself reveal the source's morphology.

This chapter then scaled those tools up through every regime nature offers, and in this tutorial:

- **Weak Lensing:** Outside the strong-lensing region every background galaxy is still weakly sheared; the signal
  is invisible per galaxy beneath intrinsic shape noise but emerges statistically over an ensemble.

- **Shear Catalogues:** The weak-lensing dataset is a catalogue of galaxy positions, measured shear estimates and
  uncertainties, produced upstream by a dedicated shape-measurement pipeline — PyAutoLens fits mass models to it
  and leaves the shape measurement to those pipelines.

- **Mass Scales:** The statistical signal is hopeless around a single galaxy, marginal around a group and
  measurable around a cluster — and it probes the dark matter halo to radii far beyond the reach of arcs.

- **The Same Workflow:** `SimulatorShearYX`, `FitWeak` and `AnalysisWeak` slot into the identical
  simulate-fit-model pattern the series has used since chapter 1, and a non-linear search recovered the cluster's
  mass from galaxy shapes alone.

- **Joint Fits:** Strong lensing pins the inner mass, weak lensing measures the outskirts; fitting both with one
  model is the expected PyAutoLens workflow for cluster and group science.

And that is the series: from a single lens galaxy, to lenses with extra galaxies, to multi-galaxy lenses, to
scaling relations that keep many-galaxy models tractable, to group-scale and cluster-scale systems, and finally
beyond the Einstein radius altogether into the weak-lensing regime. You have every conceptual tool the modern
lensing literature uses, and you have used each one on data.

Where next? The `autolens_workspace` is the destination for real science. Its `imaging`, `point_source`, `group`,
`cluster` and `weak` packages hold the production-ready versions of everything taught here — including the
`start_here.py` scripts that are each topic's canonical, always-current reference — plus the features this series
could only gesture at: multi-wavelength fits, interferometry, automated SLaM pipelines and more. Take your own
data there, and good luck with your science!
"""
