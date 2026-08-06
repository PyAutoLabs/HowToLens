"""
Tutorial 2: Multi-Galaxy Lenses
===============================

In the previous tutorial, we learned how to deal with extra galaxies near a strong lens — nuisance objects whose
light contaminates the data but which play no meaningful role in the lensing itself. We removed their emission
from the analysis, or gave them a heavily restricted model, and the single dominant lens galaxy remained the
star of the show.

In this tutorial, we meet systems where that picture breaks down entirely: **multi-galaxy lenses**, where two
(or more) galaxies of comparable mass both contribute significantly to the lensing of a single background source.
Neither galaxy is a minor perturber we can mask away or simplify — they are **co-dominant deflectors**, and every
one of them needs its own free light and mass model.

How do such systems arise physically? There are two main channels:

- **Merging galaxies**: two galaxies at the same redshift caught in the act of merging, or orbiting one another
  in close proximity. Their projected separation is small compared to the Einstein radius of their combined mass,
  so the source's light rays feel both galaxies' gravity at once. Because both deflectors are at one redshift,
  ray tracing is single-plane: their deflection angles simply add.

- **Chance line-of-sight alignments**: two physically unrelated galaxies at *different* redshifts that happen to
  lie along the same line of sight. Light from the source is deflected first by the more distant galaxy, and the
  already-deflected rays are deflected again by the nearer one. This is compound, multi-plane ray tracing — the
  `Tracer` handles it natively by simply assigning each galaxy its redshift, and we will defer its details to
  later in this chapter, where multi-plane lensing becomes the default at cluster scale.

In this tutorial, both lens galaxies are at the same redshift, so we stay in the simpler single-plane regime and
focus on the modeling challenge that defines multi-galaxy lensing: the growth of the model's parameter space.

__Contents__

- **Initial Setup:** Load the double lens galaxy dataset, simulating it first if it is not on disk.
- **Mask:** Define a mask which encloses the combined Einstein ring of both lens galaxies.
- **Over Sampling:** Centre the adaptive over sampling grid on every deflector, not just one.
- **Model:** Compose a lens model with a free light and mass model per deflector, and count its parameters.
- **Fixing the Mass Centres:** Fix each galaxy's mass centre to its observed light centre, and why this is standard.
- **Model Fit:** Fit the two-deflector model to the data with a non-linear search.
- **Result:** Inspect the combined critical curve and the fit to the data.
- **Mass Degeneracies:** The total mass is well constrained, but its split between the galaxies is not.
- **No Shared Halo:** What we are deliberately not yet assuming, and why that changes at group scale.
- **Three Lens Galaxies:** A triple-deflector system, and how the parameter count keeps growing.
- **Wrap Up:** Summary and the road to scaling relations.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We begin with `Imaging` of a strong lens where **two** lens galaxies, separated by 2.0", both deflect the light
of a single background source:

 - The two lens galaxies' light are `Sersic` profiles, centred at (0.0", -1.0") and (0.0", 1.0").
 - The two lens galaxies' mass distributions are `Isothermal` profiles with comparable Einstein radii (1.0" and
   0.8") — this comparability is what makes them co-dominant.
 - The source galaxy's light is an `ExponentialCoreSph`.

__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
dataset_name = "x2_lens_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/lens_x2.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
When we plot the data, the difference from every lens we have modeled so far is obvious: there are two bright
lens galaxies, and the lensed source's arcs wrap around the *pair as a whole* rather than around either galaxy
individually. The mass distribution the source's light responds to is the sum of both galaxies' mass.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We define a 3.0" circular mask. For a multi-galaxy lens, sizing the mask needs more care than at galaxy scale:
the Einstein radius that matters is that of the *combined* mass distribution, not either galaxy's individually.
A mask sized by eye from one galaxy's light would clip the arcs, which extend beyond both galaxies' centres.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

We use the adaptive over sampling scheme introduced in earlier chapters, which evaluates the steep central
regions of the lens galaxies' light at high resolution. The one multi-galaxy specific point is that the adaptive
grid is centred on **every** deflector, not just one — each galaxy has its own steep central light profile
needing accurate evaluation, and `centre_list` takes as many centres as we give it.

The two centres below are the observed centres of the two lens galaxies, which for this simulated dataset we
know exactly. For real data you would measure them from the image itself (the `autolens_workspace` provides a
GUI which writes them from mouse clicks).
"""
lens_centres = [(0.0, -1.0), (0.0, 1.0)]

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=lens_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

We now compose the lens model, and here the multi-galaxy regime shows its teeth. Every co-dominant deflector
needs its own free light and mass model, so the model has one `Galaxy` entry per deflector.

We build each lens galaxy in a loop over the observed centres and store them as `lens_0`, `lens_1`, etc. This
list-based composition scales to any number of deflectors, and it is the same API the `autolens_workspace`'s
`multi_galaxy` package (and the group-scale examples above it) use — so nothing needs re-learning later.

For each lens galaxy's light we use a Multi Gaussian Expansion (MGE) made of linear light profiles, built by the
utility function `al.model_util.mge_model_from` (this hides the longer `Basis` composition API we stepped through
in the linear profiles tutorial of chapter 2). For each lens galaxy's mass we use an `Isothermal` (SIE) profile,
initially with every parameter free. The source is also an MGE.

The MGE choice matters more here than anywhere we have used it before. Each lens galaxy's 20 Gaussians add just
**4 non-linear parameters** (a shared centre and shared elliptical components — every `sigma` is fixed and every
`intensity` is solved for by the linear inversion). A single ordinary `Sersic` bulge would add 7 non-linear
parameters per galaxy, and would fit the blended, overlapping light of a close pair far less well. Because the
multi-galaxy regime multiplies whatever light model we choose by the number of deflectors, a light model that is
both cheap and flexible is close to essential — this is exactly where the MGE shines.

Each Isothermal mass model adds 5 free parameters: its centre (2), elliptical components (2) and Einstein
radius (1).

(The `autolens_workspace` multi-galaxy examples additionally include a single external shear for the whole
system, held in its own model entry at the system centre rather than attached to any one deflector — the shear
describes the tidal field of structure *outside* the system, so it belongs to no individual galaxy. We omit it
here to keep the parameter accounting simple.)
"""
lens_dict = {}

for i, centre in enumerate(lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
The model's `info` shows `lens_0` and `lens_1` each carrying their own free mass model — the signature of the
multi-galaxy regime — and its `prior_count` gives the total number of free parameters.
"""
print(model.info)

print(f"Free parameters (free mass centres): {model.prior_count}")

"""
The count is **22**: each deflector contributes 4 (MGE light) + 5 (SIE mass) = 9 free parameters, and the MGE
source contributes the final 4. Compare this to the equivalent single-galaxy model, which has just 13 — one
co-dominant deflector added 9 parameters, and every further deflector will add 9 more. Model complexity compounds
fast in this regime, and with plain Sersic light profiles instead of MGEs it would compound faster still (12 per
deflector rather than 9).

__Fixing the Mass Centres__

Before fitting, we make one change that is standard practice for multi-galaxy lenses: we **fix each galaxy's
mass centre to its observed light centre**, removing 2 free parameters per deflector.

Why? With a single lens galaxy, the data usually constrains the mass centre well: the arcs pin down where the
deflection field is centred, and there is only one plausible culprit. With multiple deflectors this breaks down.
The source's light responds to the *summed* deflection field, so a small shift of one galaxy's mass centre can be
compensated by shifts of the other's centre, ellipticity or Einstein radius — the free centres become extremely
degenerate with one another. The search wanders these degeneracies, converging slowly and often settling in
unphysical corners of parameter space where one galaxy's mass has drifted far from any light.

Fixing each mass centre to the galaxy's light centre resolves this at minimal cost. Light traces stellar mass,
and the stellar body sits at the bottom of the galaxy's potential well, so the light centre is an excellent
estimate of the mass centre. (Measuring a genuine offset between mass and light — which interacting pairs can
show at the kiloparsec level — is real science, but it is a *follow-up* fit performed after a robust model with
fixed centres has been found.)

Assigning a tuple to the mass model's `centre` fixes it, so it is no longer a free parameter with a prior.
"""
lens_dict = {}

for i, centre in enumerate(lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
Printing the model's `info` again, each mass `centre` is now listed as a fixed value with no prior, and the
free parameter count has dropped from 22 to **18** — each SIE now contributes 3 free parameters instead of 5.
"""
print(model.info)

print(f"Free parameters (fixed mass centres): {model.prior_count}")

"""
__Model Fit__

We fit the model with the nested sampling algorithm `Nautilus`, as in previous chapters. Because this model has
more free parameters than the single-galaxy fits of chapter 2 (which used 100 live points), we raise `n_live`
to 200 — a multi-galaxy parameter space is more multi-modal, and too few live points risks converging on a local
maximum where the two galaxies' Einstein radii have been mis-apportioned.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_4"),
    name="tutorial_2_multi_galaxy",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/tutorial_2_multi_galaxy"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result's `info` shows the inferred model, with each deflector's parameters listed under its own `lens_0` /
`lens_1` entry.
"""
print(result.info)

"""
The tracer subplot is worth a close look for a multi-galaxy lens: the critical curve is that of the *combined*
mass distribution, so it wraps around the pair as a whole rather than encircling either galaxy individually —
just as the arcs in the data do.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Mass Degeneracies__

The corner plot of the posterior is where the multi-galaxy regime reveals its most important lesson. Below we
plot the probability density function of every parameter; when you run this tutorial for real (rather than
skimming the generated output), focus on the panel showing `lens_0`'s Einstein radius against `lens_1`'s.
"""
aplt.corner_anesthetic(samples=result.samples)

"""
You will see a strongly elongated, tilted degeneracy: the two Einstein radii are *anti-correlated*. When one
galaxy's Einstein radius goes up, the other's goes down, tracing out a narrow ridge in parameter space.

The physical reason is the summed deflection field. The arcs constrain the **total** mass enclosed by the
combined Einstein ring extremely well — move along the ridge and the sum of the two galaxies' deflections barely
changes, so the fit to the data barely changes either. What the data constrains much more weakly is the **split**
of that mass between the two galaxies. The closer the pair (relative to the combined Einstein radius), the less
information the arcs carry about which galaxy the mass belongs to, and the longer the ridge grows.

This has a real scientific cost. For a single-galaxy lens, the Einstein mass is one of the cleanest measurements
in astrophysics. For a multi-galaxy lens, the total is still clean, but per-galaxy quantities — each galaxy's
mass, its mass-to-light ratio, its dark matter fraction — inherit the full width of the degeneracy, and their
error bars can be many times larger than the total's. Any interpretation built on the split (e.g. comparing the
two galaxies' dark matter content) must honestly propagate this, which is why we fit these systems with a
sampler that maps the full posterior rather than an optimizer that returns a single best-fit point: the best-fit
point sits somewhere on the ridge and tells you nothing about the ridge's length.

__No Shared Halo__

It is worth being explicit about an assumption we did *not* make. We modeled two galaxies, each with its own
untruncated `Isothermal` mass profile, and simply added their deflections. We did not assume the pair inhabits a
single, large, shared dark matter halo — and because there is no host halo in the model, we also made no
assumptions about tidal stripping or the truncation of each galaxy's individual halo.

At this scale, that is the right call: a pair (or triple) of comparable galaxies has no dominant halo for its
members to be stripped by, and the data cannot demand one. But climb the mass ladder and it changes. At the
group scale, a dominant group-sized halo enters the model as an explicit choice, and the member galaxies orbiting
within it are tidally truncated by its potential — assumptions that reshape the entire mass model, as we will
see later in this chapter.

__Three Lens Galaxies__

What happens when a third co-dominant deflector joins the system? Let us load a triple-galaxy lens, simulated
with three `Isothermal` mass profiles of comparable Einstein radii (0.9", 0.8" and 0.7") arranged in a triangle,
all lensing a single source.
"""
dataset_name = "x3_lens_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/lens_x3.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
We compose the three-deflector model with exactly the same loop as before — this is the payoff of the list-based
API, which needs no changes as the deflector count grows. Each mass centre is again fixed to its galaxy's
observed light centre; with three deflectors the centre degeneracies are even more severe than for the pair, so
the standard trick matters even more.
"""
lens_centres = [(0.9, 0.0), (-0.6, -0.9), (-0.6, 0.9)]

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_dict = {}

for i, centre in enumerate(lens_centres):

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=True,
        centre=(centre[0], centre[1]),
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = (centre[0], centre[1])

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
    )

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

print(model.info)

print(f"Free parameters (three deflectors, fixed mass centres): {model.prior_count}")

"""
The count is now **25**: even with fixed mass centres and cheap MGE light models, every deflector adds 7 free
parameters (4 light + 3 mass), and with free centres it would be 9. The trend is linear and unforgiving:

 - 1 deflector: 13 free parameters.
 - 2 deflectors: 18 (this tutorial's fit).
 - 3 deflectors: 25.
 - 10 deflectors: ~74. A group-scale lens can easily have this many members.

And the parameter count is only half the problem. As the deflectors multiply, so do the degeneracies between
their masses — every pair of galaxies gets its own version of the Einstein radius ridge we saw above, and the
posterior becomes progressively harder for any sampler to map.

You are welcome to fit this three-galaxy model yourself — the search above works unchanged, though expect it to
take noticeably longer than the pair. Clearly, freeing every galaxy's mass cannot scale to the tens or hundreds
of member galaxies in groups and clusters. Something has to give.

__Wrap Up__

In this tutorial, we modeled strong lenses with multiple co-dominant deflectors. Let's summarise what we've
learnt:

- **Co-dominant deflectors**: When two or more galaxies of comparable mass both contribute significantly to the
  lensing, every one of them needs its own free light and mass model — unlike the nuisance neighbours of the
  previous tutorial, none can be masked away or simplified.

- **Physical origins**: Multi-galaxy lenses arise from merging galaxies at one redshift (single-plane, deflections
  add) or chance line-of-sight alignments at different redshifts (multi-plane ray tracing, covered later in this
  chapter).

- **Parameter growth**: Each deflector added 9 free parameters (4 for its MGE light, 5 for its SIE mass) — and the
  MGE is what kept it that cheap, since its Gaussian intensities are solved for by linear algebra rather than sampled.

- **Fixed mass centres**: Fixing each galaxy's mass centre to its observed light centre (removing 2 parameters per
  deflector) is the standard trick for taming the severe centre degeneracies of multi-deflector models.

- **Mass degeneracies**: The data constrains the total mass well but its split between the galaxies poorly,
  producing an anti-correlated ridge between the deflectors' Einstein radii — and inflating the uncertainties on
  any per-galaxy science.

- **No shared halo**: At this scale we do not assume the galaxies inhabit one large dark matter halo, so no tidal
  truncation assumptions enter the model — that framing arrives at group scale.

The three-galaxy model made the trajectory clear: freeing every deflector cannot scale. In the next tutorial, we
introduce the tool that stops this growth in its tracks — scaling relations, which tie the masses of many
galaxies to their observed luminosities so that an entire population of deflectors costs almost no extra free
parameters.
"""
