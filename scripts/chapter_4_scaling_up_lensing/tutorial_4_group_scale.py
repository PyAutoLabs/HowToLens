"""
Tutorial 4: Group Scale
=======================

The previous tutorials introduced nearby galaxies, multiple significant deflectors and scaling relations. We now
combine these ideas to model a **group-scale lens**.

A galaxy group contains galaxies associated with a larger gravitationally bound structure. Its lensing signal
can include contributions from a shared dark matter halo, a central galaxy and other member galaxies.

The brightest group galaxy is often called the **BGG**. In some groups it lies near the centre of the shared halo
and contributes strongly to the lensing. Other groups have several prominent galaxies or a less clearly defined
centre.

Our example contains one dominant central component, two member galaxies and one background source. We will
introduce truncated member-galaxy mass profiles, compare independent member masses with a scaling relation, and
use Bayesian evidence to compare models with and without an explicit NFW dark matter halo.

__Groups vs Multi-Galaxy Lenses__

The distinction between group-scale and multi-galaxy lensing is not a strict boundary in galaxy count or Einstein
radius.

As discussed in Tutorial 2, an interacting or merging pair of field galaxies may be modeled without a separate
group-scale halo. Other systems inhabit a shared halo whose contribution matters for interpreting the lensing.

The important questions are which mass components contribute to the observed images and whether the data can
distinguish them. A mass profile centred on the BGG can describe the combined central mass without separately
identifying the contributions of the galaxy and the group halo.

We therefore distinguish **including the group's gravitational effect** from **representing its halo as a
separate model component**. Our initial model uses a central total-mass profile; later, we examine an additional
halo component.

__Tidal Truncation and the dPIE Profile__

Galaxies orbiting within a group can lose material from their outer dark matter halos through tidal stripping.
The extent of this stripping depends on their orbits, internal structure and interaction history.

The isothermal profiles used so far extend indefinitely and have divergent total mass. They can describe lensing
over a limited radial range, but do not represent a finite outer halo.

The **dPIE** profile provides a useful alternative. It has an approximately isothermal region and steepens smoothly
beyond a characteristic truncation scale, giving it finite total mass.

We use its spherical implementation, `al.mp.dPIEMassSph`, for the member galaxies. Its truncation scale lets us
represent a reduced outer mass distribution, although adopting this profile does not by itself measure a
galaxy's stripping history.

__Contents__

- **Initial Setup:** Load the simulated group-scale dataset (auto-simulating it if absent) and plot it.
- **Mask:** Define the 2D mask, which is larger than at galaxy scale because the group spans more sky.
- **Galaxy Centres:** Load the centres of the BGG and member galaxies from .json files.
- **The dPIE Profile:** Introduce the truncated dPIE mass profile and its parameters.
- **Fitting a Group:** Fit the data with a tracer containing the BGG, members and source.
- **A Group Halo?:** Motivate a shared NFW dark matter halo and define two models to compare.
- **Model Fit:** Compose and fit the galaxy-only model with individually modeled members.
- **Fit With a Shared NFW Dark Matter Halo:** Refit with an additional NFW halo and unchanged galaxy priors.
- **Bayesian Model Comparison:** Compare the evidences of the galaxy-only and galaxy-plus-halo models.
- **Scaling Relation Members:** Tie the members' masses to their light so they cost almost no parameters.
- **The Group Scale Sweet Spot:** Why groups are the sweet spot of the familiar lens modeling toolkit.
- **Wrap Up:** Summary of the script and next steps.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We load the simulated group-scale dataset `simple`, containing:

- A central galaxy at `(0.0, 0.0)`, represented by `SersicSph` light and an `IsothermalSph` total-mass profile
  with Einstein-radius parameter 4.0".
- Two member galaxies at `(3.5, 2.5)` and `(-4.4, -5.0)`, with `SersicSph` light and truncated `dPIEMassSph`
  mass profiles.
- One background source with a `SersicCore` light profile.

Coordinates use the usual `(y, x)` convention in arcseconds.

The simulation contains no separate group-halo profile. Its central mass component provides a simple description
of the dominant deflection field, without decomposing that mass into the BGG and a shared halo.
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
Plot the dataset and identify the central galaxy, the extended lensed arcs and the two member galaxies.

The central mass profile has an Einstein-radius parameter of 4.0", so this example's lensed emission spans a larger
angular region than the earlier examples. The members modify the combined deflection field.

As in Tutorial 1, consider their light and mass separately. We will explicitly model both here, but a galaxy's
mass can affect the arcs even when its light lies outside the fitting region.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use a circular mask with radius 7.5", centred on the central galaxy. This includes the lensed arcs and both
member-galaxy centres.

Inspect the masked subplot to assess the retained source emission and member light. The outer member lies
approximately 6.7" from the origin, so its centre is inside the mask but some of its extended light lies outside it.

The mask selects the image pixels used in the fit; it does not set a boundary on which galaxies can contribute
gravitational deflections. A member outside the mask can still be included in the mass model if its lensing
effect matters.

Here, we retain a broad fitting region so that we can model the central galaxy, member light and lensed source
together.
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

We load the measured galaxy positions from JSON files, separating them into:

- `main_lens_centres`: positions of the main deflectors, here the central galaxy.
- `member_centres`: positions of the additional group members.

This separation helps organize the model; it does not itself fix parameters or determine how much freedom each
galaxy receives.

As in the previous tutorials, measuring light centres before constructing the mass model is a common practical
workflow. For real observations, these positions can come from a GUI, a source-detection catalogue or a
preliminary light-profile fit. Here, the simulator supplies them.

We will fix the member profiles to these positions and use the combined list of galaxy centres when setting up
adaptive over-sampling.
"""
main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

member_centres = al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")

print(f"Main lens centres: {main_lens_centres}")
print(f"Member centres: {member_centres}")

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=list(main_lens_centres) + list(member_centres),
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__The dPIE Profile__

The spherical `dPIEMassSph` profile uses the parameter convention of the lens-modeling software Lenstool:

- `sigma`: a fiducial velocity-dispersion parameter in km/s that sets the mass normalization. It is not
  automatically equal to an observed stellar velocity dispersion; comparisons require attention to the profile
  convention and how the observation was measured.
- `r_core`: the core scale in arcseconds. A nonzero value flattens the central density. We set it to zero here,
  giving a singular central profile.
- `r_cut`: the truncation scale in arcseconds. The density steepens smoothly around this scale, giving a finite
  total mass. It is not a sharp boundary beyond which the mass vanishes.
- `redshift_object`, `redshift_source`, `H0` and `Om0`: the redshifts and cosmological parameters used to
  convert the physical normalization into lensing quantities. We hold them fixed in this tutorial.

We create two profiles with the same `sigma` and zero core radius, but different truncation scales: 8.0" and
1000.0".

The second profile is effectively untruncated over the radial range we plot, although it still has a finite
truncation scale.
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
We evaluate both profiles along a radial line using `Grid2DIrregular`.

The plotted quantity is **convergence**: projected surface mass density divided by the critical surface density
for lensing. It is not the three-dimensional density.

The logarithmic vertical axis helps us compare the profiles over a wide range of convergence. The dashed line
marks the 8.0" truncation scale of the more strongly truncated profile.
"""
radii = np.linspace(0.01, 20.0, 200)

radial_grid = al.Grid2DIrregular([(0.0, radius) for radius in radii])

convergence_truncated = dpie_truncated.convergence_2d_from(grid=radial_grid)
convergence_untruncated = dpie_untruncated.convergence_2d_from(grid=radial_grid)

plt.semilogy(radii, np.asarray(convergence_truncated), label="dPIE (r_cut = 8.0)")
plt.semilogy(radii, np.asarray(convergence_untruncated), label="dPIE (r_cut = 1000.0)")
plt.axvline(x=8.0, color="k", linestyle="--", label="r_cut")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Convergence")
plt.legend()
plt.show()
plt.close()

"""
At radii much smaller than 8.0", the profiles approach similar isothermal behaviour. As we move towards and
beyond the smaller truncation scale, that profile's convergence falls more steeply.

The transition is gradual: `r_cut` does not mark an abrupt edge, and the profiles can differ before reaching it.
The truncated profile also continues to contribute mass and deflections beyond that radius.

This flexibility makes the dPIE useful for representing member galaxies with reduced outer halos. However,
neither group membership nor proximity to another galaxy uniquely determines a truncation radius. Its value
remains a modeling choice or an inferred parameter, depending on the available constraints.

__Fitting a Group__

We first evaluate a model using the parameter values supplied to the simulator. This is a direct fit evaluation,
rather than a search for the best-fitting parameters.

The tracer contains the central galaxy, two members and the source. Because the deflectors share a redshift,
their deflections combine on one lens plane before mapping the source light into the image.

The familiar `FitImaging` object compares this predicted image with the data, including the PSF and noise map.
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
Inspect the residuals of the fit evaluated at the simulation's input parameters.

With consistent simulation and fitting calculations, we expect residuals compatible with the added noise.
Coherent structure could instead indicate a mismatch in the profiles, PSF, numerical sampling or other assumptions.

The fitting machinery is unchanged from earlier chapters. We have supplied a tracer containing more galaxies and
a different member-mass profile.
"""
print(f"Log likelihood of the true model: {fit.log_likelihood}")

"""
__A Group Halo?__

In the standard cosmological model, **Lambda Cold Dark Matter (LCDM)**, galaxies form within dark matter halos.
Galaxy groups are expected to occupy larger **host dark matter halos**, within which member galaxies orbit,
often retaining their own smaller halos.

This gives us a physical reason to include an extended, shared dark matter component when modeling a group-scale
lens.

However, seeing several galaxies close together on the sky does not automatically establish that they inhabit
a group-scale halo. They could be projected neighbours or an interacting pair of field galaxies, for which
separate galaxy-centred mass profiles provide an adequate description. Those galaxies still contain dark matter;
what may be unnecessary is an **additional shared group-scale component**.

The astronomical question is therefore whether the system's environment and lensing signal support a shared
host halo, and whether the data can distinguish its contribution from the masses associated with individual
galaxies.

We investigate this by fitting two models:

- **Galaxy mass profiles only:** a central galaxy and member galaxies, without a separate host halo.
- **Galaxy mass profiles plus a shared dark matter halo:** the same components, with an additional NFW halo
  at the group redshift.

The Navarro-Frenk-White (NFW) profile is a commonly used description of dark matter halos in LCDM. We allow the
galaxy parameters to adjust in both fits and compare the Bayesian evidences. This tests whether the data support
the additional NFW component, given our profile choices and priors.

If the simpler model is sufficient, it does not overturn the LCDM expectation: the shared halo's contribution
may be absorbed by the galaxy-centred profiles, or the available images may not distinguish it separately.

Our simulated dataset contains no separate NFW halo, so it provides a useful starting point for this comparison.

__Model Fit__

We first fit the galaxy-only model:

- The BGG has a linear `SersicSph` light profile and an `IsothermalSph` mass profile, with free centres:
  seven non-linear parameters.
- Each member has a linear `SersicSph` light profile and a `dPIEMassSph` mass profile. Both centres are fixed
  to the measured position. The light's effective radius and Sersic index, together with the mass normalization
  `sigma`, give three parameters per member.
- The source has a linear `SersicCore` light profile: six non-linear parameters.

For each member, we fix `r_core = 0.0` and adopt `r_cut = 10.0"`. We also fix the redshifts and cosmological
parameters.

Fixing the member centres follows the measured-position workflow used throughout this chapter. Fixing their
truncation scales is a further assumption: the arcs may constrain a member's lensing strength more readily
than its outer mass distribution.

The chosen 10.0" truncation scale differs from the simulation's member values of 8.0" and 13.3". This fit therefore
tests a simplified model rather than reproducing every simulation assumption exactly. We use the same member
treatment in both halo-comparison models.
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

source_model = af.Model(
    al.Galaxy, redshift=1.0, bulge=af.Model(al.lp_linear.SersicCore)
)

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

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free,
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed
parameter.
"""
af.ModelPlotter(model).figure()

print(f"Total free parameters: {model.prior_count}")

"""
Adaptive over-sampling was configured around every galaxy centre before the initial fit evaluations. The same
configuration is retained for the non-linear search.
"""


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
    "The non-linear search has begun running - checkout the output/howtolens/chapter_4/tutorial_4_group_scale"
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
__Fit With a Shared NFW Dark Matter Halo__

We now construct Model 2 by copying the galaxy-only model and adding a spherical NFW dark matter halo at the
group redshift. Copying the original model preserves the galaxy priors: we do not replace them with posterior
constraints from the first fit.

For this illustrative comparison, we fix the halo centre to the measured central-galaxy position and fit two
parameters:

- `kappa_s`: the dimensionless NFW density normalization, with a uniform prior from 0.0 to 0.5.
- `scale_radius`: the NFW scale radius in arcseconds, with a uniform prior from 5.0" to 50.0".

These are explicit example priors, not universal choices for galaxy groups. The halo's shape and position are
also assumptions here. For a scientific analysis, choose priors appropriate to the system and assess how they
affect the evidence and inferred mass decomposition.

Both models fit the same dataset, mask, noise map and source description. All galaxy parameters that were free
in Model 1 remain free in Model 2, allowing mass to be redistributed between the central galaxy and host halo.
"""
model_galaxies_only = model
model_with_halo = copy.deepcopy(model_galaxies_only)

halo_mass = af.Model(al.mp.NFWSph)
halo_mass.centre = tuple(main_lens_centres[0])
halo_mass.kappa_s = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)
halo_mass.scale_radius = af.UniformPrior(lower_limit=5.0, upper_limit=50.0)

model_with_halo.galaxies.group_halo = af.Model(
    al.Galaxy, redshift=0.5, mass=halo_mass
)

print(model_with_halo.info)

search_with_halo = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_4",
    name="tutorial_4_group_scale_with_nfw_halo",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_with_halo = search_with_halo.fit(model=model_with_halo, analysis=analysis)

print(result_with_halo.info)
aplt.subplot_fit_imaging(fit=result_with_halo.max_log_likelihood_fit)

"""
__Bayesian Model Comparison__

Compare the log evidences from the two fitted models. We report the difference as the NFW-halo model minus
the galaxy-only model: a positive difference favours including the NFW halo, while a negative difference favours
the galaxy-only description, under the adopted priors.

Interpret this alongside fit diagnostics and the numerical reliability of the searches. The evidence comparison
accounts for the additional parameter space; it is not simply a comparison of the best-fitting images.
For equal prior model probabilities, the difference in log evidences is the log posterior odds between the models.

A preference for the NFW model supports that explicit component within these model choices. A preference for the
simpler model does not demonstrate the absence of a physical dark matter halo.

The next scaling-relation exercise returns to the galaxy-only setup, keeping the halo comparison separate from
the change in member-mass treatment.
"""
log_evidence_galaxies_only = result.samples_summary.log_evidence
log_evidence_with_halo = result_with_halo.samples_summary.log_evidence

if (
    log_evidence_galaxies_only is not None
    and log_evidence_with_halo is not None
    and np.isfinite(log_evidence_galaxies_only)
    and np.isfinite(log_evidence_with_halo)
):
    print(f"Log evidence, galaxy-only model: {log_evidence_galaxies_only:.3f}")
    print(f"Log evidence, galaxy + NFW model: {log_evidence_with_halo:.3f}")
    print(
        "Delta log evidence (NFW minus galaxy-only): "
        f"{log_evidence_with_halo - log_evidence_galaxies_only:.3f}"
    )
else:
    print("Finite evidences are unavailable; complete both nested-sampling fits before comparing the models.")

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

"""
The tied members have moved out of `extra_galaxies` into `scaling_galaxies`, and they no longer share one
description: each member's `sigma` is no longer an independent prior but an expression,
`sigma = scaling_galaxies.1.mass.sigma.sigma_ref * <luminosity ratio>`, with a different ratio for each. One
free prior, two galaxies — the scaling relation.

`r_cut` stays fixed, as it was before. `r_cut_ref * luminosity_ratio ** 0.7` is arithmetic on plain numbers
rather than on a prior, so it yields a fixed value rather than a relation.

Nineteen free parameters fall to eighteen. One parameter is an unremarkable saving for two members, but the
saving keeps growing: however many members you add to `scaling_galaxies`, every one of their `sigma` values
points back at that single `sigma_ref`.
"""
af.ModelPlotter(model).figure()

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
    "The non-linear search has begun running - checkout the output/howtolens/chapter_4/"
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
