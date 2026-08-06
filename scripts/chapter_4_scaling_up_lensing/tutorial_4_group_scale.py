"""
Tutorial 4: Group Scale
=======================

In the previous tutorials we took our first steps beyond the single lens galaxy: we included extra galaxies near
the lens in the model, we modeled systems where two or more galaxies of comparable mass share the lensing, and we
introduced scaling relations, which tie the mass of a galaxy to its light so that adding more galaxies to a model
does not mean adding more free parameters.

In this tutorial we put all of those tools together and take the first big step up the "regime ladder" of strong
lensing: the **group scale**.

In the standard cosmological model (Lambda Cold Dark Matter, or LCDM), galaxies do not live in isolation. They
form and evolve inside "dark matter halos", and these halos are themselves nested: small halos hosting single
galaxies merge and fall into larger halos hosting many. A **galaxy group** is the next rung above a single
galaxy in this hierarchy: a dark matter halo of roughly 10^13 to 10^14 solar masses hosting anywhere from a
handful to a few dozen galaxies. (Our own Milky Way lives in such a structure, the Local Group.) Above groups
sit **galaxy clusters**, halos of 10^14 solar masses and beyond hosting hundreds or thousands of galaxies, which
are the subject of the next tutorial.

When a group acts as a strong lens, several of its galaxies contribute significantly to the deflection of the
source's light. The brightest galaxy of the group -- called the brightest group galaxy (BGG), or the brightest
cluster galaxy (BCG) at cluster scale -- typically sits near the centre of the group's dark matter halo and
dominates the lensing. The other group members perturb the lensed image around it. The group's shared dark
matter halo may also enter the mass model as its own component, often centred on the BGG where one exists --
though as we will see, whether it is needed is an explicit modelling choice, not an automatic ingredient.

__Groups vs Multi-Galaxy Lenses__

You may now be wondering how a "group-scale lens" differs from the multi-galaxy lenses of the previous tutorials,
and the honest answer is that the line between them is **blurry**. Every group is a multi-galaxy lens: it has
multiple galaxies whose mass matters for the lensing. But not every multi-galaxy lens is a group: two galaxies of
comparable mass that happen to sit close on the sky (or are mid-merger) share the lensing without being embedded
in a massive shared halo with a member population.

The physically meaningful distinction is the **environment**: a group has a shared dark matter halo, a dominant
central galaxy, and members orbiting inside the host's potential. That environment leaves signatures in the mass
model, and the most important one is **tidal truncation**, which we introduce below.

__Tidal Truncation and the dPIE Profile__

A galaxy orbiting inside a group does not keep its full dark matter halo. The tidal field of the host group
strips the outer, loosely bound parts of the member's halo, so the member's mass distribution is cut off -- or
"truncated" -- at a finite radius. The isothermal profiles we have used so far are a poor description of this:
their density falls as 1/r^2 forever, so their total mass diverges with radius.

The standard truncated profile of group- and cluster-scale lensing is the **dPIE** (dual Pseudo-Isothermal
Elliptical) profile, available in **PyAutoLens** as `al.mp.dPIEMassSph`. It behaves isothermally at intermediate
radii but its density falls off much more steeply beyond a truncation radius, giving it a finite total mass --
exactly the behaviour we expect for a tidally stripped group member.

__Contents__

- **Initial Setup:** Load the simulated group-scale dataset (auto-simulating it if absent) and plot it.
- **Mask:** Define the 2D mask, which is larger than at galaxy scale because the group spans more sky.
- **Galaxy Centres:** Load the centres of the BGG and member galaxies from .json files.
- **The dPIE Profile:** Introduce the truncated dPIE mass profile and its parameters.
- **Fitting a Group:** Fit the data with a tracer containing the BGG, members and source.
- **A Group Halo?:** Add a group-scale dark matter halo to the tracer and see how it changes the fit.
- **Model Fit:** Compose and fit a group-scale lens model with individually modeled members.
- **Scaling Relation Members:** Tie the members' masses to their light so they cost almost no parameters.
- **The Group Scale Sweet Spot:** Why groups are the sweet spot of the familiar lens modeling toolkit.
- **Wrap Up:** Summary of the script and next steps.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We begin by loading the group-scale strong lens dataset `simple`, which was simulated with:

 - A main lens galaxy (the BGG) at (0.0", 0.0"), with a `SersicSph` light profile and an `IsothermalSph` mass
   profile with a 4.0" Einstein radius.
 - Two member galaxies at (3.5", 2.5") and (-4.4", -5.0"), with `SersicSph` light profiles and tidally
   truncated `dPIEMassSph` mass profiles.
 - A single source galaxy with a `SersicCore` light profile.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "group" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/group.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
When we plot the dataset, the difference from a galaxy-scale lens is immediately clear: the lensed arcs form a
much larger ring (the BGG's Einstein radius is 4.0", compared to the ~1.0" typical of a single galaxy), and two
smaller galaxies are visible away from the centre. Their light -- and, more importantly for the lensing, their
mass -- must be included in our analysis.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We define a 7.5" circular mask, much larger than the 2.5"-3.0" masks of previous chapters. It must enclose the
full lensed arc system and the member galaxies, because all of them contribute light and mass to the region we
fit.
"""
mask_radius = 7.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Galaxy Centres__

Following the convention of the previous tutorials, the centres of the galaxies are loaded from .json files in
the dataset folder. For a group we distinguish two categories:

 - `main_lens_centres`: the main lens galaxies -- here just the BGG at (0.0", 0.0") -- which are modeled with
   full freedom.

 - `member_centres`: the group members, which are modeled with restrictive assumptions (fixed centres, and
   later a scaling relation) to keep the model dimensionality low.

For real data these centres come from the observed light of each galaxy (e.g. via a click-based GUI or a
photometry catalogue), exactly as for the extra galaxies of the earlier tutorials.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

member_centres = al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")

print(f"Main lens centres: {main_lens_centres}")
print(f"Member centres: {member_centres}")

"""
__The dPIE Profile__

Before fitting anything, lets get to know the dPIE profile, since it is the workhorse mass profile of the group
and cluster regimes.

The `dPIEMassSph` profile is parameterized in the convention of Lenstool (a long-established cluster lensing
code), which differs from the profiles we have met so far:

 - `sigma`: the fiducial velocity dispersion of the galaxy in km/s. This sets the overall lensing strength,
   playing the role the `einstein_radius` played for the isothermal profile (for an isothermal sphere the two
   are directly related: a bigger velocity dispersion means a bigger Einstein radius).

 - `r_core`: the core radius in arcseconds, within which the density profile flattens. It is common to fix
   `r_core = 0.0` (a "vanishing core", where the profile has an exact analytic form), which we do throughout
   this tutorial.

 - `r_cut`: the truncation radius in arcseconds -- the key new parameter. Outside `r_cut` the density falls off
   steeply (as 1/r^4 rather than the isothermal 1/r^2), encoding the tidal stripping of the member's outer dark
   matter halo and giving the profile a finite total mass.

 - `redshift_object`, `redshift_source`, `H0`, `Om0`: because `sigma` is a physical velocity, converting it to
   deflection angles requires the lensing geometry and a cosmology. These are fixed inputs describing the lens
   system, not parameters we fit.

Lets make two dPIE profiles with identical `sigma` but different truncation radii -- one truncated at 8.0" and
one truncated so far out (1000.0") that it is effectively untruncated -- and compare their convergence profiles.
"""
dpie_truncated = al.mp.dPIEMassSph(
    centre=(0.0, 0.0),
    sigma=200.0,
    r_core=0.0,
    r_cut=8.0,
    redshift_object=0.5,
    redshift_source=1.0,
)

dpie_untruncated = al.mp.dPIEMassSph(
    centre=(0.0, 0.0),
    sigma=200.0,
    r_core=0.0,
    r_cut=1000.0,
    redshift_object=0.5,
    redshift_source=1.0,
)

"""
We evaluate the convergence of both profiles on a 1D line of radial coordinates, using a `Grid2DIrregular` so we
can choose the radii ourselves.
"""
radii = np.linspace(0.01, 20.0, 200)

radial_grid = al.Grid2DIrregular([(0.0, radius) for radius in radii])

convergence_truncated = dpie_truncated.convergence_2d_from(grid=radial_grid)
convergence_untruncated = dpie_untruncated.convergence_2d_from(grid=radial_grid)

plt.semilogy(radii, np.asarray(convergence_truncated), label="dPIE (r_cut = 8.0)")
plt.semilogy(radii, np.asarray(convergence_untruncated), label="dPIE (untruncated)")
plt.axvline(x=8.0, color="k", linestyle="--", label="r_cut")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Convergence")
plt.legend()
plt.show()
plt.close()

"""
Inside `r_cut` the two profiles are nearly identical -- both behave isothermally. Beyond `r_cut`, the truncated
profile's convergence plummets while the untruncated one keeps its shallow isothermal decline. The truncated
member therefore contributes far less mass at large radii, which is exactly what tidal stripping does.

This is why the dPIE, and not the isothermal profile, is the standard choice for group and cluster members: the
extra galaxies of a galaxy-scale lens have no host environment stripping them, so untruncated isothermal
profiles are fine there, but a member orbiting inside a shared group halo is physically truncated whether or
not the model includes that halo as an explicit component.

For scale: a member with `sigma = 200.0` km/s at these redshifts has an Einstein radius of ~0.7", roughly a
fifth of the BGG's -- a perturber, not a co-dominant lens.

__Fitting a Group__

Now lets fit the data. Following the fitting tutorials of earlier chapters, we first build a tracer from
galaxies whose light and mass profiles match the true values used to simulate the data, and fit it with the
`FitImaging` object.

The tracer contains four galaxies: the BGG, the two members and the source. The fit handles all of them
simultaneously, summing the deflection field of every mass profile to ray-trace the source's light.
"""
bgg = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

member_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(3.5, 2.5),
        sigma=200.0,
        r_core=0.0,
        r_cut=8.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

member_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=1.866, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(-4.4, -5.0),
        sigma=240.0,
        r_core=0.0,
        r_cut=13.3,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[bgg, member_0, member_1, source])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
Because this is the true model, the residuals are consistent with noise and the log likelihood is as high as it
can be for this dataset. Note that nothing about the `FitImaging` object changed: the same fitting machinery we
learnt at galaxy scale simply receives a tracer with more galaxies in it.
"""
print(f"Log likelihood of the true model: {fit.log_likelihood}")

"""
__A Group Halo?__

You may have noticed that the model above contains no group-scale dark matter halo, despite this tutorial
opening with the statement that a group is defined by one. Two things are going on here.

First, the BGG's `IsothermalSph` has a 4.0" Einstein radius -- far larger than the ~1.0" of an isolated galaxy
of its luminosity. A mass profile centred on the BGG cannot tell "stars and dark matter belonging to the BGG"
apart from "group dark matter centred on the BGG": a single isothermal profile at the group centre soaks up
both. So the group's halo is partly hidden inside the BGG's mass profile already.

Second, whether to add a *separate* halo component to the model is an **explicit modelling choice**, and this
dataset was deliberately simulated without one. Lets see what happens if we add one anyway. We add a fifth
galaxy carrying only mass -- an `IsothermalSph` centred on the BGG, the conventional choice for a group halo
where a BGG exists -- and refit.
"""
group_halo = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.5),
)

tracer_with_halo = al.Tracer(galaxies=[bgg, group_halo, member_0, member_1, source])

fit_with_halo = al.FitImaging(dataset=dataset, tracer=tracer_with_halo)

aplt.subplot_fit_imaging(fit=fit_with_halo)

"""
The fit is now visibly worse -- the extra 1.5" Einstein radius of mass at the centre over-deflects the source's
light, displacing the model arcs from the observed ones -- and the log likelihood drops accordingly.
"""
print(f"Log likelihood with a group halo added: {fit_with_halo.log_likelihood}")

"""
On real data the same experiment runs in reverse: fit the group with and without an explicit halo component and
let the data decide (via the residuals and the Bayesian evidence introduced in earlier chapters). Some groups
are well described by their galaxies alone, with the central galaxy's profile absorbing the halo; others --
typically those with the largest Einstein radii or image configurations no galaxy-mass model can reproduce --
require a dominant halo component, often centred on the BGG.

__Model Fit__

We now perform a proper model-fit with a non-linear search, as a scientist would for a group whose true
parameters are unknown. We compose the model as follows:

 - The BGG's light is a linear `SersicSph` (its centre free) and its mass an `IsothermalSph` [7 parameters].

 - Each member's light is a linear `SersicSph` with its centre fixed to the observed centre, and its mass a
   `dPIEMassSph` with a free `sigma`, its centre fixed, `r_core = 0.0` and `r_cut` fixed to a fiducial 10.0"
   [3 parameters per member, 6 total].

 - The source's light is a linear `SersicCore` [6 parameters].

Fixing the member centres to their observed light and fixing `r_cut` follows the reasoning of the extra
galaxies tutorial: the data rarely constrains a perturber's truncation radius, so we spend our parameter budget
on the quantity that matters most -- each member's overall mass, via `sigma`. The redshifts and cosmology of
the dPIE are pinned to their known values, since they are properties of the lens system, not parameters to
sample.
"""
# BGG:

bulge = af.Model(al.lp_linear.SersicSph)

mass = af.Model(al.mp.IsothermalSph)

bgg_model = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Member Galaxies:

member_list = []

for centre in member_centres:

    bulge = af.Model(al.lp_linear.SersicSph)
    bulge.centre = (centre[0], centre[1])

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = (centre[0], centre[1])
    mass.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
    mass.r_core = 0.0  # vanishing core -- fixed; the dPIE is analytic at r_core = 0
    mass.r_cut = 10.0  # truncation fixed at a fiducial radius
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66  # pinned: model constants, not parameters to sample
    mass.Om0 = 0.30966

    member_list.append(af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass))

# Source:

source_model = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp_linear.SersicCore))

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=bgg_model, source=source_model),
    extra_galaxies=af.Collection(member_list),
)

"""
Printing the model's `info` confirms its composition: the BGG and source under `galaxies`, and the two members
under `extra_galaxies` with only their `sigma`, `effective_radius` and `sersic_index` free.

The total number of free parameters is N=19 -- take note of this number, as it is central to the discussion at
the end of this tutorial.
"""
print(model.info)

print(f"Total free parameters: {model.prior_count}")

"""
Before fitting, we set up the adaptive over-sampling scheme at the centre of every galaxy in the group (not
just the BGG), following the standard workspace approach. As in previous tutorials, the details are not
important yet -- just note that at group scale every galaxy centre needs it.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres) + list(member_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
We fit the model with the `Nautilus` non-linear search, using the same `AnalysisImaging` object as every
imaging fit so far -- another sign that the group scale does not require new machinery.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_4",
    name="tutorial_4_group_scale",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/tutorial_4_group_scale"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
The result's `info` and maximum likelihood fit show the model recovers the group: the BGG's Einstein radius,
the members' velocity dispersions and the source's properties are all inferred from the data.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Scaling Relation Members__

The model above gave every member its own free `sigma`. With two members that costs two parameters -- harmless.
But groups can host dozens of members, and one free mass parameter per member quickly bloats the model, slows
the search, and asks the data to constrain masses it barely feels individually.

The previous tutorial's solution applies directly here: tie the members' masses to their light through a
scaling relation. More luminous galaxies are more massive, so we write each member's `sigma` and `r_cut` as
power laws of its luminosity, normalized by a single shared free parameter:

 - `sigma = sigma_ref * (L / L_ref) ** 0.25`
 - `r_cut = r_cut_ref * (L / L_ref) ** 0.7`

These exponents follow the convention of modern cluster lensing analyses, and `L_ref` is a fixed reference
luminosity. The 0.25 is the previous tutorial's Faber-Jackson relation recast for velocity dispersion:
`L ~ sigma^4` inverts to `sigma ~ L^0.25`, and since `theta_E ~ sigma^2` this reproduces the `theta_E ~ L^0.5`
scaling we anchored there. Note also the change of normalization: rather than hanging the relation off an
anchor galaxy the model already fits, we normalise it with a shared free parameter at a fixed reference
luminosity. Now the *entire member population's mass* costs one free parameter (`sigma_ref`), no matter how
many members the group has, and each member's observed luminosity acts as a physically motivated prior on its
mass.

For real data the luminosities come from your photometry catalogue (only luminosity ratios enter the model, so
any consistent units work). Here we compute them from the true member light profiles, using an aperture
luminosity within 3.0".
"""
luminosity_list = [
    member_0.bulge.luminosity_within_circle_from(radius=3.0),
    member_1.bulge.luminosity_within_circle_from(radius=3.0),
]

print(f"Member luminosities: {luminosity_list}")

reference_luminosity = luminosity_list[0]

sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=300.0)
r_cut_ref = 8.0  # fixed, like r_cut in the previous model

member_list = []

for centre, luminosity in zip(member_centres, luminosity_list):

    luminosity_ratio = float(luminosity) / float(reference_luminosity)

    bulge = af.Model(al.lp_linear.SersicSph)
    bulge.centre = (centre[0], centre[1])

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = (centre[0], centre[1])
    mass.sigma = sigma_ref * luminosity_ratio**0.25
    mass.r_core = 0.0
    mass.r_cut = r_cut_ref * luminosity_ratio**0.7
    mass.redshift_object = 0.5
    mass.redshift_source = 1.0
    mass.H0 = 67.66
    mass.Om0 = 0.30966

    member_list.append(af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass))

model = af.Collection(
    galaxies=af.Collection(lens=bgg_model, source=source_model),
    scaling_galaxies=af.Collection(member_list),
)

"""
Note that the tied members now live in the `scaling_galaxies` collection — the home the previous tutorial
introduced for a scaling-relation population — whereas the individually-modelled members above sat in
`extra_galaxies`, like tutorial 1's interlopers. Both collections are included in the ray-tracing; the split
keeps `model.info` readable and tells the results machinery which galaxies are a tied population.

The model's `info` shows that both members' `sigma` values now trace back to the single shared `sigma_ref`
prior, and the parameter count has dropped to N=18.

A drop of one parameter looks unremarkable -- until you scale it up. A group with 20 members modeled
individually would need 20 free mass parameters; with the scaling relation it still needs exactly one. (In
practice the members' light is handled the same way, using the Multi Gaussian Expansion from earlier in the
lectures, whose fixed-centre members add almost no free parameters either -- so entire member populations can
be added to a group model almost for free.)

This dataset was in fact simulated with members that obey this exact scaling relation, so the model remains the
true model.
"""
print(model.info)

print(f"Total free parameters: {model.prior_count}")

"""
We fit this model with an identical search set up.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_4",
    name="tutorial_4_group_scale_scaling_relation",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/"
    "tutorial_4_group_scale_scaling_relation folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_scaling = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

print(result_scaling.info)

aplt.subplot_fit_imaging(fit=result_scaling.max_log_likelihood_fit)

"""
__The Group Scale Sweet Spot__

Step back and consider what this tutorial did *not* require: no new fitting machinery, no new analysis object,
no new data format. The `FitImaging`, `AnalysisImaging`, linear light profiles and `Nautilus` search of the
galaxy-scale chapters handled the group unchanged -- we just put more galaxies in the tracer, reached for the
truncated dPIE profile, and used scaling relations to keep the member population cheap. Even the pixelized
source reconstructions of the previous chapter work at group scale exactly as they do at galaxy scale, and for
real groups with complex arcs they are the recommended source model.

This is why the group scale is the **sweet spot** of strong lens modeling. The full toolkit of galaxy-scale
modeling still applies, and the model dimensionality stays low: our individually-modeled-member fit had N=19
free parameters and the scaling-relation fit N=18. Real group models are of this order -- a few tens of free
parameters at most (N < 30 or so) -- comfortably within reach of the non-linear searches we already know, with
run times of minutes to hours.

Push the system any richer, however, and this toolkit breaks down. A cluster hosts hundreds of members and
lenses many sources at once; its image is too large and complex to fit pixel-by-pixel with a tracer's light
model, and its model would need hundreds of parameters. Cluster-scale modeling therefore switches to a
different tool set -- fitting the *positions* of the multiple images of many point-like sources, with member
catalogues driving the mass model -- which is the subject of the next tutorial.

__Wrap Up__

In this tutorial, we took the tools of the previous tutorials to the group scale. Lets recap what we learnt:

- **Groups in LCDM**: A galaxy group is a dark matter halo of roughly 10^13 to 10^14 solar masses hosting a
  handful to a few dozen galaxies, with the brightest group galaxy (BGG) typically near its centre. The group's
  halo may enter the mass model as its own component, often centred on the BGG -- but whether it is needed is
  an explicit modelling choice, tested by fitting with and without it.

- **Groups vs multi-galaxy lenses**: The line is blurry. All groups are multi-galaxy lenses, but not all
  multi-galaxy lenses are groups -- the group's defining feature is the shared halo environment its members
  orbit within.

- **Tidal truncation and the dPIE**: Members orbiting in the group's potential have their outer dark matter
  tidally stripped, so their profiles are truncated. The `dPIEMassSph` profile encodes this via its `r_cut`
  truncation radius, with `sigma` setting the lensing strength and `r_core` the (usually vanishing) core.

- **Fitting and modeling groups**: The galaxy-scale machinery works unchanged -- a tracer with more galaxies,
  members with fixed centres and free `sigma`, and a `Nautilus` search over N=19 parameters.

- **Scaling relations**: Tying member `sigma` and `r_cut` to luminosity via power laws with one shared free
  normalization means the whole member population costs a single parameter, however many members there are.

- **The sweet spot**: Groups are the largest systems the familiar toolkit -- including pixelized source
  reconstruction -- still handles, with model dimensionality staying below a few tens of parameters.

In the next tutorial we climb the final rung of the ladder to galaxy clusters, where tens to hundreds of member
galaxies lens many sources simultaneously. There the toolkit changes: we will model the positions of multiply
imaged point sources rather than every pixel of the data, and drive the mass model from member catalogues via
a csv interface built on the same scaling relations we used here.
"""
