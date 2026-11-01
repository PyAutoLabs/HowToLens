"""
Tutorial 2: Multi-Galaxy Lenses
===============================

In the previous tutorial, we modeled a system with one dominant lens galaxy and a smaller nearby galaxy.
We saw that the extra galaxy's light and gravitational deflections required separate consideration.

We now consider **multi-galaxy lenses**, where two or more galaxies contribute substantially to the lensing of a
background source. Each significant deflector needs to be represented in the mass model, although we can still
fix or link appropriate parameters.

Such configurations can arise when galaxies at similar redshifts lie close together in projection, including
interacting or merging systems. They can also arise through chance alignments of galaxies at different redshifts.

When we approximate the deflectors as lying at the same redshift, their deflection angles add on a single lens
plane. Deflectors at different redshifts require multi-plane ray-tracing, which accounts for how earlier deflections
change a ray's position at subsequent planes.

In this tutorial, both lens galaxies are at the same redshift. We therefore focus on two modeling challenges:
the growing number of parameters and the degeneracies between the galaxies' mass distributions.

We will also reuse the practical workflow introduced in Tutorial 1: measure galaxy light centres first, then
use those positions to constrain the model.

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

We begin with an `Imaging` dataset containing two lens galaxies, separated by 2.0", which both deflect light from
a single background source.

- Their light follows `Sersic` profiles centred at `(0.0, -1.0)` and `(0.0, 1.0)`, using our usual `(y, x)`
  coordinates in arcseconds.
- Their mass follows `Isothermal` profiles with Einstein-radius parameters of 1.0" and 0.8".
- The source light follows an `ExponentialCoreSph` profile.

Both galaxies contribute substantially to the deflection field. Their individual Einstein-radius parameters
describe the two mass profiles; neither alone describes the critical curves of the combined system.

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
Plot the dataset and identify the two lens galaxies and the surrounding lensed source emission.

The source is lensed by their combined deflection field, so we must interpret the image configuration using both
galaxies together. We cannot treat it as two independent single-galaxy lenses.

Notice where the galaxies' light overlaps the source emission: our model will need to separate these contributions
while fitting the data.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use a circular mask with radius 4.0", centred between the two lens galaxies. Inspect the masked dataset to check
that it retains the lensed source emission, including its fainter outskirts. For multi-galaxy lenses, choose the
fitting region from the full image configuration rather than either galaxy's individual Einstein radius.
"""
mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Over Sampling__

We use the adaptive over-sampling method introduced in earlier chapters to evaluate the galaxies' steep central
light profiles accurately.

Here, we supply both galaxy centres through `centre_list`, so the higher-resolution sampling covers the central
region of each galaxy.

This reuses the measured-position workflow from Tutorial 1. For real data, we can obtain the light centres using
a GUI, a source-detection tool or a preliminary light-profile fit. For this simulated dataset, we use the known
input positions.

These positions currently determine where the grid receives finer sampling; supplying them to the over-sampling
function does **not** fix any model parameters. Later, we will explicitly fix the galaxies' mass centres to
these positions.
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

We now build a model containing two lens galaxies and one background source.

We loop over the measured lens-galaxy positions and create a separate `Galaxy` model for each deflector, named
`lens_0`, `lens_1`, and so on. This composition extends naturally to systems containing more galaxies.

Each lens galaxy's light is represented by a Multi-Gaussian Expansion (MGE), using the
`al.model_util.mge_model_from` helper. This constructs the Gaussian basis using the approach introduced in chapter 2.

Each MGE contains 20 Gaussians with fixed widths, a shared centre and shared elliptical components. Their
intensities are solved by the linear inversion, leaving four non-linear light parameters per galaxy: two centre
coordinates and two elliptical components.

An MGE does not necessarily add any non-linear parameters. If its Gaussian widths are fixed, its centre is fixed
to the measured light position, and its Gaussians are spherical, only their intensities remain to be determined.
These are solved by the linear inversion, so the entire light model adds **zero non-linear parameters**,
regardless of how many Gaussians it contains.

If we instead fit shared elliptical components while keeping the centre fixed, the MGE adds **two** non-linear
parameters. Freeing the shared centre as well increases this to **four**, as in the model used here.

An MGE offers flexibility when fitting overlapping galaxy light without sampling every Gaussian intensity.
Whether that flexibility is needed depends on the observed morphology; a simpler light profile can still be
appropriate.

Each galaxy's mass is represented by an `Isothermal` profile, initially with five free parameters: two centre
coordinates, two elliptical components and an Einstein radius. The source light is also represented by an MGE.

The measured positions locate the priors on the lens light centres; those centres remain free in this model.
We will explicitly fix the mass centres in the next step.

We omit external shear in this example. If included, it would enter through a separate `MassField`, representing
an additional tidal contribution to the combined deflection field.
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
The model's `info` shows the light and mass components of `lens_0` and `lens_1`, together with the source model.

Inspect which parameters are free, which are fixed and which are shared between Gaussian profiles. We will use
`model.prior_count` below to count the free non-linear parameters.
"""
print(model.info)

"""
The figure shows the structure of our initial model. Each lens galaxy has its own `Isothermal` mass profile,
with free centre coordinates, elliptical components and Einstein radius.

This is an intermediate model used to illustrate the parameter count. Before fitting, we will fix each mass
centre to its measured light position.

Compare the light and mass components: sharing parameters within an MGE reduces its dimensionality, but the two
galaxies still have separate light and mass models.
"""
af.ModelPlotter(model).figure()

print(f"Free parameters (free mass centres): {model.prior_count}")

"""
The initial model has **22 free non-linear parameters**:

- Each lens galaxy contributes four light parameters and five mass parameters.
- The source contributes four light parameters.

The equivalent single-lens model would have 13. Adding the second deflector therefore adds nine parameters before
we impose further restrictions.

__Fixing the Mass Centres__

We now fix each galaxy's mass centre to its measured light centre, removing two free parameters per deflector.

This follows the common workflow introduced in Tutorial 1: measure galaxy positions from the image using a GUI,
a source-detection tool or a preliminary light-profile fit, then use those positions as fixed inputs to the lens
model.

With multiple deflectors, the lensed images constrain their combined deflection field. Changes in one galaxy's
mass centre can sometimes be compensated by changes in another galaxy's position, ellipticity or mass
normalization. Fixing the centres reduces this freedom and can make the search more manageable.

Here, we fix only the **mass centres**. The MGE light centres retain their existing priors and remain free,
allowing the light fit to refine their positions. The fixed mass positions are the supplied measurements;
they do not move with the fitted light centres.

Fixing the mass centres alone does not reduce the MGE's parameter count: the current code still uses elliptical
MGEs with free light centres. A spherical MGE with fixed widths and a fixed light centre would instead add zero
non-linear light parameters, as described above.

This assumes that each mass distribution is centred on its measured light position. It is a useful starting
point, but does not establish that light and mass must be perfectly aligned. Where positional uncertainty or
possible offsets matter, we can allow appropriate freedom and investigate its effect.

Assigning a coordinate tuple to `mass.centre` fixes both coordinates rather than assigning them priors.
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
Printing the updated model's `info` shows that both mass centres are now fixed.

Each `Isothermal` profile therefore has three free parameters: two elliptical components and one Einstein radius.
The light models are unchanged.

Fixing two coordinates for each of two galaxies removes four parameters, reducing the total from **22 to 18**.
"""
print(model.info)

"""
The figure now shows each galaxy's mass centre fixed to its measured light position.

Two fixed centres remove four non-linear parameters, reducing the model from 22 to 18. The MGE light centres and
elliptical components remain free, so each galaxy's light model still contributes four non-linear parameters.

Recall that a spherical MGE with fixed widths and a fixed light centre would contribute zero non-linear
parameters. We retain the more flexible elliptical light models here.
"""
af.ModelPlotter(model).figure()

print(f"Free parameters (fixed mass centres): {model.prior_count}")

"""
__Model Fit__

We fit the model using `Nautilus`, as in previous chapters.

With two deflectors, the search must explore correlations between their mass profiles as well as the source
parameters. We use 200 live points here, compared with 100 in our earlier examples, to give the search more scope
to explore this parameter space.

Increasing the number of live points does not guarantee a reliable result. Inspect the fitted images and
posterior distributions, and check the stability of important conclusions using repeated searches or more
demanding search settings where needed.
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
    "The non-linear search has begun. Check the output/howtolens/chapter_4/tutorial_2_multi_galaxy"
    " folder for results and visualizations as the search progresses."
    " This cell will finish when the search completes."
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result's `info` summarizes the inferred parameters and their uncertainties. The two deflectors appear
separately under `lens_0` and `lens_1`.

Compare their inferred Einstein radii and elliptical components, remembering that the parameters of one galaxy
may be correlated with those of the other. The fixed mass centres are assumptions supplied to the fit, rather
than positions measured independently by this search.
"""
print(result.info)

"""
Inspect the tracer subplot to see how the two galaxies combine to lens the source.

The critical curves belong to the combined lens mapping. Depending on the galaxies' masses, shapes and separation,
they can form connected or separate structures; they are not obtained by simply drawing an Einstein ring around
each galaxy.

Then inspect the fit subplot. Check whether the model reproduces both galaxies' light and the lensed source
emission, and look for coherent residuals around the arcs or where the light components overlap.

A satisfactory image fit is an essential check, but it does not establish that each galaxy's mass is individually
well constrained. We investigate that next using the posterior.
"""
aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Mass Degeneracies__

Multiple significant deflectors introduce additional mass degeneracies beyond those encountered in single-galaxy
lenses. The lensed images constrain the galaxies' **combined deflection field**, so changes in one galaxy's mass
distribution can be compensated by changes in another's.

Consequently, a model may reproduce the arcs accurately while leaving substantial uncertainty in how the mass is
divided between the galaxies. This is a major challenge of multi-galaxy modeling: a well-constrained combined lens
does not necessarily imply well-constrained individual galaxy masses.

Single-galaxy lenses also have mass-source degeneracies, but multiple deflectors add freedom to redistribute mass
**between galaxies**. How severe this becomes depends on their separation and the positions and quality of the
lensed images.

In the corner plot below, find the panel comparing the `einstein_radius` parameters of `lens_0` and `lens_1`.
Look for a correlation: can a larger mass normalization for one galaxy be compensated by a smaller normalization
for the other?

Also inspect how these parameters correlate with the galaxies' elliptical components and the source parameters.
The degeneracy need not involve only the two Einstein radii.

The shape and extent of these distributions tell us more about the uncertainty in the mass decomposition than a
single best-fit model can.
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

"""
A third deflector, `lens_2`, is an identical copy of the structure above, with its own fixed mass `centre`.

Nothing about the *shape* of the model changed when we added a galaxy, which is exactly the point the
list-based API is making. The model grows by one deflector, and the parameter count by seven.
"""
af.ModelPlotter(model).figure()

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
