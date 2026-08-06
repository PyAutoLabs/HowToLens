"""
Tutorial 6: Weak Lensing
========================

Every tutorial in this series so far — indeed, every fit performed in all four chapters of **HowToLens** — has been
a *strong* lensing analysis. Strong lensing occurs when a background galaxy lies so close (in projection) to a
foreground mass that its light is bent into multiple images, arcs or a complete Einstein ring. These dramatic
features are what we simulated, fitted and modeled, from the single galaxy-scale lenses of chapters 1 and 2 up to
the group-scale and cluster-scale systems earlier in this chapter.

Strong lensing is spectacular, but it is also rare and spatially confined. Multiple images only form inside (or very
near) the Einstein radius, so a strong-lensing analysis constrains the mass distribution only within that region —
a few arc-seconds for a galaxy, tens of arc-seconds for a cluster core. Outside it, the arcs and multiple images
simply do not exist, and everything we have built so far has nothing to fit.

Yet the gravity of the foreground mass does not stop at the Einstein radius. *Every* background galaxy, however far
from the lens centre, has its shape slightly distorted — sheared — by the foreground mass it sits behind. This is
**weak lensing**: a percent-level stretching of each background galaxy's observed ellipticity, tangentially aligned
around the foreground mass.

The catch is that we cannot see this distortion in any single galaxy. Galaxies are not intrinsically round: a
typical galaxy has a random unlensed ellipticity of ~0.25 per component, whereas the weak-lensing shear it receives
is ~0.01-0.1. The signal is buried an order of magnitude below the intrinsic "shape noise" of any one object.

The solution is statistical. The intrinsic ellipticities of different galaxies are randomly oriented and average to
zero, whereas the lensing shear is coherent — every galaxy near a given point on the sky is sheared the same way.
Average the measured shapes of many background galaxies and the random part cancels while the lensing part remains.
This is the fundamental contrast with everything the series has done so far: strong lensing extracts an exquisite
signal from a single special system, weak lensing extracts a faint signal from the ensemble of many ordinary ones.

This changes what the data even *is*. There is no image to fit, no PSF to convolve, no mask, no pixelized source
reconstruction. A weak-lensing dataset is a **shear catalogue**: a table of background-galaxy sky positions, each
with a measured shear estimate (two ellipticity components) and a per-galaxy uncertainty. Fitting one means
comparing a mass model's predicted shear field, evaluated at those positions, against the measured values.

In this tutorial we simulate a weak-lensing shear catalogue around a cluster-scale lens (entirely in memory — no
files are written), visualize its shear field, fit it with a mass model whose parameters we know, and then infer
those parameters with a non-linear search. You will recognise every step: the `Tracer`, `Fit`, `Analysis` and
search workflow of the whole series, applied to a completely different kind of data.

This tutorial closes both this chapter and the four core chapters of **HowToLens**.

__Contents__

- **Shear Catalogues:** What a weak-lensing dataset is and where the catalogue comes from in a real analysis.
- **Mass Scales:** Why weak lensing is a cluster-scale tool, tying to the mass scales of the previous tutorials.
- **Ray Tracing:** Build a `Tracer` for a cluster-scale lens, the mass whose shear field we will measure.
- **Source Galaxy Positions:** Draw background galaxies in an annulus outside the strong-lensing core.
- **Simulate:** Create a `WeakDataset` in memory with `SimulatorShearYX`, adding realistic shape noise.
- **Visualize:** Plot the shear field as a whisker map and the dataset's 2x2 summary mosaic.
- **Mass Map:** Invert the shear catalogue into a model-independent convergence map (Kaiser-Squires).
- **Fitting:** Fit the catalogue with the true mass model via `FitWeak` and inspect its goodness-of-fit.
- **Model Fit:** Infer the mass model from the shear catalogue alone with `AnalysisWeak` and a non-linear search.
- **Result:** The inferred model, the fit mosaic and the tangential shear profile.
- **Joint Strong and Weak Lensing:** How PyAutoLens expects weak lensing to be used — combined with strong lensing.
- **Wrap Up:** The end of the chapter, and of the core HowToLens lectures.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Shear Catalogues__

Before we simulate anything, lets be clear about what a real weak-lensing dataset looks like, because it is unlike
any data this series has fitted before.

A **shear catalogue** is a table with one row per background galaxy, containing:

 - The galaxy's sky position (for us, (y, x) arc-second offsets from a chosen centre).
 - A measured shear estimate (gamma_1, gamma_2) — in practice the galaxy's measured ellipticity components, which
   estimate the lensing shear because the intrinsic shape averages to zero over many galaxies.
 - A per-galaxy uncertainty, combining the intrinsic shape dispersion (~0.25 per component) with the measurement
   error, added in quadrature.

Producing this catalogue from raw telescope images is a substantial data-reduction task in its own right: galaxies
must be detected, stars separated from galaxies, the PSF measured and its blurring of galaxy shapes corrected
(the PSF makes every galaxy look rounder, diluting the shear signal), and calibration biases characterised.
Dedicated shape-measurement pipelines exist for exactly this job, and entire survey collaborations are built
around getting it right.

**All of that is outside PyAutoLens's scope.** PyAutoLens assumes the shear catalogue has already been produced
upstream by such a pipeline, and starts where the catalogue ends: fitting mass models to the measured shears. This
division of labour is why the weak-lensing API is so lean — no PSF, no mask, no image — the hard observational
systematics were dealt with before the data reached us.

__Mass Scales__

This chapter has climbed a ladder of mass scales, and weak lensing is its top rung. The strength of the statistical
shear signal around a lens scales with its mass, so whether weak lensing is even measurable depends on where on the
ladder we stand:

 - **A single galaxy** (Einstein radius ~1-2", the lenses of chapters 1-3): the shear imprinted on background
   galaxies beyond the Einstein radius is far too weak to detect around any one system — there are too few
   background galaxies carrying too little shear. (Surveys recover it only by *stacking* thousands of lens
   galaxies, a technique called galaxy-galaxy lensing that averages over the lens population.)

 - **A group** (Einstein radius ~4-10", the scale of the group tutorial earlier in this chapter): the signal is
   marginal — detectable for well-observed individual groups, but noisy.

 - **A cluster** (Einstein radius ~10-30"+, the previous tutorial's scale): the signal is strong enough to measure
   around a *single* system. Hundreds to thousands of background galaxies within a few arc-minutes each carry a
   |gamma| ~ 0.01-0.2 shear, and their average traces the cluster's mass out to radii of arc-minutes — ten or more
   times further out than the strong-lensing core.

This reach is the scientific point. The dark matter halo of a cluster extends to Mpc scales, far beyond where arcs
form. Weak lensing is the only lensing probe of that outer halo, which is why it is the tool of choice for cluster
mass measurements, and it is why this tutorial simulates a cluster-scale lens.

__Ray Tracing__

We begin exactly as the series always has: with a `Tracer`. The lens is a cluster-scale `Isothermal` mass profile
with an Einstein radius of 25.0" — the same order as the previous tutorial's cluster, and a mass for which the
weak shear signal is genuinely measurable. The source galaxy carries no light profile: weak lensing measures the
lens's shear field at the background galaxies' positions, so their appearance is irrelevant — they are pure probes,
included only to give the `Tracer` its source-plane redshift.

Note also what the lens galaxy lacks: no light profile. A shear catalogue contains no image of the lens, so there
is no lens light to model — another simplification compared to every imaging fit in this series.
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

Real weak-lensing measurements avoid the strong-lensing core: inside it the shear is no longer weak (the linear
shear approximation breaks down), and cluster-member galaxies contaminate the background sample. We therefore draw
our background galaxies in an **annulus**, from an inner radius of 50" (twice the Einstein radius, safely into the
weak regime) to an outer radius of 200" (over 3 arc-minutes), distributed uniformly in area.

At 50" from this lens the shear is |gamma| ~ 0.25, and by 200" it has fallen to |gamma| ~ 0.06 — every galaxy is a
weak probe whose individual distortion is at or below the shape noise. With 1500 galaxies (a deep-survey source
density of ~45 per square arc-minute) the *ensemble* nonetheless carries a decisive signal, exactly as in a real
cluster analysis.

The positions are stored as an `al.Grid2DIrregular` of (y, x) coordinates — the same irregular-grid structure the
point-source tutorials used, because a catalogue of sky positions is not a uniform pixel grid.
"""
rng = np.random.default_rng(1)

n_galaxies = 1500
radius_inner = 50.0  # arc-seconds — inside this we are in the strong-lensing core.
radius_outer = 200.0  # arc-seconds — the edge of our simulated weak-lensing field.

radii = np.sqrt(rng.uniform(radius_inner**2.0, radius_outer**2.0, n_galaxies))
phi = rng.uniform(0.0, 2.0 * np.pi, n_galaxies)

positions = al.Grid2DIrregular(
    values=np.stack([radii * np.sin(phi), radii * np.cos(phi)], axis=1)
)

"""
__Simulate__

The `SimulatorShearYX` object simulates a weak-lensing shear catalogue from a tracer: its `via_tracer_from` method
evaluates the tracer's shear field at every galaxy position (by differentiating the deflection-angle field) and
adds Gaussian shape noise to each measurement.

A `noise_sigma` of 0.25 per shear component is the standard intrinsic shape dispersion of real galaxies — this is
the dominant noise source in weak lensing, and it is what makes the per-galaxy signal invisible: the noise on each
measurement is several times larger than the shear it contains.

Unlike every previous tutorial, we do not write this dataset to disk and reload it. The whole catalogue is a few
thousand numbers, so we keep it in memory and use it directly — which also mirrors how you might resimulate
catalogues on the fly when testing survey sensitivities. (This repo's `scripts/simulator/weak_lensing.py` — the
script chapter 1's data tutorial auto-ran to give you your first glimpse of a shear catalogue — shows the
disk-based version, with the same JSON output pattern as the imaging simulators.)
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

Weak-lensing data has its own visual language. The shear field is drawn as a **whisker map**: at each galaxy
position, a line segment whose length is the shear magnitude |gamma| and whose orientation is the shear's position
angle. The segments are deliberately *headless* (no arrowheads), because shear is a spin-2 quantity — rotating a
shear by 180 degrees maps it back onto itself, so a whisker has an orientation but no direction, and an arrowhead
would suggest information the data does not contain.

Around a massive lens, the whiskers align *tangentially* — each traces a small piece of a circle around the lens
centre, the statistical echo of the rings and arcs of strong lensing. Squint at the plot below and you can see it,
though the shape noise makes it far from obvious galaxy-by-galaxy: this is what a signal an order of magnitude
below the per-object noise looks like.
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
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/tutorial_6_weak_lensing"
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
