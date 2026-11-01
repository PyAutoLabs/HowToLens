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
al.util.dataset.simulate_if_needed(dataset_path, simulator="scripts/simulator/group.py")

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
Inspect `model.info` to check the galaxy-only model's components and priors.

The BGG and source appear under `galaxies`, while the two members appear under `extra_galaxies`. Each member has
three free non-linear parameters: `sigma`, `effective_radius` and `sersic_index`.

The total is **19 non-linear parameters**: seven for the BGG, six across the two members and six for the source.

Light-profile intensities are solved by the linear inversion during each likelihood evaluation, so they do not
add sampling dimensions. This is Model 1 of our dark matter halo comparison.
"""
print(model.info)

"""
The model figure shows how these components are organized.

Notice that the central galaxy's light and mass centres remain free, whereas the member centres are fixed to
their measured positions. Each member's mass profile has only `sigma` free; its core and truncation scales
are fixed.

Compare the figure with `model.info` and check the printed parameter count. These choices keep the model
compact while retaining independent mass normalizations for the members.
"""
af.ModelPlotter(model).figure()

print(f"Total free parameters: {model.prior_count}")

"""
Adaptive over-sampling was configured around every galaxy centre before the initial fit evaluations. The same
configuration is retained for the non-linear search.
"""


"""
We first fit the galaxy-only model using `Nautilus` and the familiar `AnalysisImaging` object.

The likelihood calculation uses the same imaging framework as in earlier chapters, now with the central galaxy
and two members contributing to the predicted image.

This search provides the first fit and Bayesian evidence for our comparison. We will then fit the model
containing an additional NFW dark matter halo using the same dataset and unchanged priors for the
galaxy components.
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
    "The non-linear search has begun. Check the output/howtolens/chapter_4/tutorial_4_group_scale"
    " folder for results and visualizations as the search progresses."
    " This cell will finish when the search completes."
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished — you may now continue the notebook.")

"""
Inspect the inferred parameters in `result.info` and the fit subplot below.

Check how well the model reproduces the central galaxy, member light and lensed arcs. Look for coherent
residuals, particularly near the member galaxies and the source emission they perturb.

The inferred member `sigma` values describe their dPIE mass normalizations under our adopted profile
assumptions. Remember that we fixed both truncation scales to 10.0", whereas the simulation used 8.0" and 13.3".

We next add the shared NFW dark matter halo and refit, allowing the galaxy parameters to adjust alongside the
halo parameters.
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

r"""
__Scaling Relation Members__

We now return to the galaxy-only model and change how its member galaxies are described. This keeps the
scaling-relation exercise separate from the NFW halo comparison.

Previously, each member had an independent mass normalization, `sigma`. We now link the members through
their luminosities:

$$
\sigma_i=\sigma_{\rm ref}
\left(\frac{L_i}{L_{\rm ref}}\right)^{0.25},
\qquad
r_{{\rm cut},i}=r_{{\rm cut},{\rm ref}}
\left(\frac{L_i}{L_{\rm ref}}\right)^{0.7}.
$$

The exponent 0.25 follows the approximate Faber-Jackson scaling introduced in Tutorial 3. The truncation exponent
0.7 is an adopted prescription for this example, rather than a universal property of group members.

Unlike the previous tutorial's anchor-galaxy approach, we introduce a shared free normalization, `sigma_ref`,
at a fixed reference luminosity. We hold both exponents and `r_cut_ref` fixed.

With member centres fixed and spherical mass profiles, the entire member **mass population** therefore has one
non-linear parameter, regardless of how many members we include.

Their light parameters remain separate. Here, each member's fixed-centre linear `SersicSph` contributes two
non-linear parameters: effective radius and Sersic index. Its intensity is solved linearly.

For this simulated example, we obtain luminosities from the known member light profiles within a 3.0" aperture
centred on each galaxy. This photometric aperture is separate from the 7.5" fitting mask.

For real observations, use consistently measured luminosities and account for the measurement and population
assumptions discussed in Tutorial 3.
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
The tied members now appear under `scaling_galaxies`, while the individually modeled members previously appeared
under `extra_galaxies`. Both contribute to the ray-tracing.

Inspect `model.info`: both member `sigma` expressions depend on the same `sigma_ref` prior.

The model has **18 non-linear parameters**: seven for the BGG, six for the source, four for the member light
profiles and one shared member-mass normalization.

With N members, the corresponding count is 14 + 2N, because their light models still contribute two parameters
each. Only the member mass-parameter count remains constant.

Fixed-width, fixed-centre spherical MGEs could remove those non-linear member-light parameters where that light
description is appropriate. Elliptical MGEs with fixed centres would retain two per member.

The truncation radii also change: instead of fixing both to 10.0", we now assign luminosity-dependent values.
The comparison therefore changes both the mass-normalization freedom and the truncation prescription.
"""
print(model.info)

"""
The figure shows the two member `sigma` values linked to one shared prior. Each is computed from `sigma_ref`
and its fixed luminosity ratio.

Their truncation radii are also calculated from luminosity, but every input to that calculation is fixed.
Each member therefore receives a fixed `r_cut` value, which can differ between members.

Replacing two independent `sigma` parameters with one shared normalization reduces the count from 19 to 18.

For a larger population, the saving grows: replacing N independent member normalizations with one shared
parameter removes N - 1 sampling dimensions, provided the other model choices remain unchanged.
"""
af.ModelPlotter(model).figure()

print(f"Total free parameters: {model.prior_count}")

"""
We fit the scaling-relation model using the same dataset, analysis object and search settings as the galaxy-only
fit with independently modeled members.

Inspect the resulting parameters and residuals to assess whether the restricted member model describes the data
adequately.

Remember that this comparison changes the truncation prescription as well as tying the mass normalizations.
Any difference in the fit cannot be attributed solely to the reduction of one parameter.
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
    "The non-linear search has begun. Check the output/howtolens/chapter_4/"
    "tutorial_4_group_scale_scaling_relation folder for results and visualizations as the search progresses."
    " This cell will finish when the search completes."
)

result_scaling = search.fit(model=model, analysis=analysis)

print("The search has finished — you may now continue the notebook.")

print(result_scaling.info)

aplt.subplot_fit_imaging(fit=result_scaling.max_log_likelihood_fit)

"""
__The Group Scale Sweet Spot__

Group-scale lenses let us apply much of the toolkit developed for individual galaxies to a richer mass
distribution.

We still use `FitImaging`, `AnalysisImaging`, linear light profiles and a non-linear search. Pixelized source
reconstructions can also be used when the lensed source has structure that a simple parametric profile does
not capture.

Our examples remain compact: 19 non-linear parameters for independently normalized members, 21 after adding the
illustrative NFW halo, and 18 for the galaxy-only scaling-relation model.

These counts describe our chosen examples, rather than an upper limit for group modeling. Real systems may
require more galaxies, additional sources, more flexible mass profiles or further freedom in the scaling relation.

Groups are particularly useful for studying how mass is divided between a central galaxy, satellite galaxies
and a larger dark matter halo. As throughout this chapter, that decomposition can remain uncertain even when
the lensed images are fitted well.

At cluster scale, many multiply imaged sources can provide constraints across a much larger region. A common
approach is to fit their image positions, using a catalogue to describe the member population.

This is the approach introduced in the next tutorial. It complements surface-brightness modeling rather than
marking a fundamental scale beyond which pixel-based fitting is impossible.

__Wrap Up__

In this tutorial, we learned:

- Group lenses combine contributions from central galaxies, members and potentially an explicitly modeled shared
  dark matter halo.
- The dPIE profile provides a finite-mass description of members, with a smooth truncation scale whose
  interpretation depends on the adopted model.
- Measuring member positions first and fixing centres helps keep the inference manageable.
- Comparing fitted galaxy-only and galaxy-plus-NFW models tests whether the data support a separate halo
  component under the chosen priors.
- Luminosity-based scaling relations can replace independent member mass normalizations with shared parameters,
  while member-light freedom must still be counted separately.
- The familiar imaging toolkit extends naturally to groups, but reliable mass decomposition requires attention
  to degeneracies and modeling assumptions.

In the next tutorial, we move to **galaxy clusters**, fitting the positions of multiple images from background
sources and building member-galaxy models from a CSV catalogue.
"""
