r"""
Tutorial 3: Scaling Relations
=============================

In the previous tutorial, we saw how adding significant deflectors increases both the number of parameters and
the degeneracies between galaxy masses. Independently fitting the mass of every member becomes increasingly
demanding for groups and clusters.

A **scaling relation** reduces this freedom by linking galaxy mass properties to measured luminosities through
a relation shared by the population.

Here, one galaxy anchors the relation: its Einstein radius remains free, while the other galaxies' Einstein
radii follow from their luminosity ratios. With fixed positions, spherical mass profiles and a fixed scaling
exponent, each additional tied galaxy adds **zero non-linear mass parameters**.

The light-model choices still matter. As we discussed in Tutorial 2, a spherical MGE with fixed widths and centre
adds zero non-linear light parameters. An elliptical MGE with a fixed centre and free shared elliptical
components adds two, as in this tutorial.

Scaling relations therefore control the growth of the mass model; they do not automatically make every part of
a many-galaxy model independent of galaxy count. More galaxies also increase the computational work required to
evaluate the model.

We will introduce the method using the familiar two-galaxy dataset, compare parameter counts and examine the
assumptions we impose when tying galaxy masses together.

__Overview__

In this tutorial, we will:

- Explain the assumptions linking luminosity to mass normalization through the Faber-Jackson relation.

- Compose a lens model where one galaxy "anchors" the relation and the other galaxy's Einstein radius is tied to
  it via the ratio of their measured luminosities.

- Verify by explicit parameter counting that tied galaxies add zero additional non-linear mass parameters.

- Fit the model and inspect how a tied parameter appears in the results.

- Discuss the limitations: real galaxies scatter around any relation, and a member that deviates can bias the
  whole lens model.

__Contents__

- **Mass Follows Light:** The physics linking a galaxy's luminosity to its lensing mass.
- **Initial Setup:** Load the two-lens-galaxy imaging dataset from the previous tutorial.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Measured Luminosities:** The luminosities the relation needs, and where they come from.
- **The Anchor:** Identify the brightest galaxy, which anchors the relation.
- **Over Sampling:** Adaptive over-sampling centred on both lens galaxies.
- **Light Via MGE:** MGE light models for every galaxy, at a cost of two parameters each.
- **Scaling Relation:** Tie the fainter galaxy's Einstein radius to the anchor's.
- **Model:** Compose the model with the scaling tier as its own collection.
- **Parameter Counts:** Compare against the previous tutorial's per-galaxy model.
- **Scaling To Many Galaxies:** Show the model does not grow when 12 more members are tied.
- **Model Fit:** Fit the scaling-relation model with Nautilus.
- **Results:** How a tied parameter appears in the results, and how close the truth is to the relation.
- **Limitations:** Scatter, deviant members, and adding per-galaxy freedom selectively.
- **Lens Environments:** The same machinery models the environment of single-lens systems.
- **Wrap Up:** Summary of the script and next steps.

__Mass Follows Light__

Galaxy luminosity provides useful information about mass, but it does not determine an individual galaxy's mass
exactly.

For early-type galaxies, the empirical **Faber-Jackson relation** links luminosity to stellar velocity dispersion.
A commonly used approximate form is:

$$
L \propto \sigma_\star^4.
$$

For a spherical isothermal lens, the Einstein radius is related to its mass-model velocity-dispersion parameter by:

$$
\theta_{\rm E}
=4\pi\left(\frac{\sigma_{\rm SIS}}{c}\right)^2
\frac{D_{\rm ls}}{D_{\rm s}},
$$

where $D_{\rm ls}$ and $D_{\rm s}$ are angular-diameter distances from lens to source and observer to source.

Assuming that the mass-model velocity dispersion scales with the measured stellar velocity dispersion, galaxies
at the same lens redshift lensing the same source approximately satisfy:

$$
\theta_{\rm E}\propto L^{1/2}.
$$

We therefore adopt:

$$
\theta_{{\rm E},i}
=\theta_{{\rm E},{\rm anchor}}
\left(\frac{L_i}{L_{\rm anchor}}\right)^{0.5}.
$$

This is a population-level modeling assumption. The exponent need not be universal, individual galaxies scatter
around the relation, and stellar and mass-model velocity dispersions are not automatically identical.

The phrase "mass follows light" here refers to linking mass normalization to luminosity. We are not requiring
the spatial mass distribution to have the same shape as the observed light.

Only luminosity ratios enter this expression, so a common multiplicative calibration cancels. However, the
measurements must use a consistent waveband and luminosity definition. Changing waveband can change the ratios
because galaxies have different stellar populations and colours.

For galaxies at different redshifts, observed flux ratios are not directly luminosity ratios, and the
lensing-distance factors also differ. Our example avoids these complications by placing both deflectors at the
same redshift.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We use the same two-galaxy imaging dataset as in Tutorial 2:

- Both lens galaxies have `Sersic` light profiles.
- Both have `Isothermal` mass profiles.
- The source light follows an `ExponentialCoreSph` profile.

Reusing this dataset lets us compare independently fitted galaxy masses with masses linked through a scaling
relation.

The simulated masses were specified independently, so we should check how closely they satisfy our adopted
relation rather than assume that they do.
"""
dataset_name = "x2_lens_galaxies"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/lens_x2.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use the same 4.0" circular mask as in Tutorial 2, centred between the two lens galaxies.

Keeping the fitting region consistent helps us compare the effects of changing the model. Inspect the masked
subplot to check that the lensed emission we want to analyze is retained.
"""
mask_radius = 4.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

r"""
__Measured Luminosities__

Our relation uses a measured position and luminosity estimate for each galaxy. We treat these as fixed inputs
in this example.

For real observations, positions and luminosities can come from a preliminary light-profile fit or a suitable
photometric catalogue. In blended systems, separating each galaxy's light from its neighbours and the lensed
source is an important part of this measurement.

Magnitudes measured consistently in the same band can be converted to flux ratios using:

$$
\frac{F_i}{F_{\rm anchor}}
=10^{0.4(m_{\rm anchor}-m_i)}.
$$

For galaxies at the same distance, with consistent photometric treatment, these provide the luminosity ratios
used here.

Because this dataset is simulated, we can load the true light profiles from `tracer.json`. We obtain each
galaxy's centre and integrate its profile within a circular aperture of radius `mask_radius`, centred on that
galaxy.

These are **aperture luminosities**, not integrals over the entire profile. Using the same aperture definition
makes the measurements consistent, but does not guarantee that the aperture contains the same fraction of every
galaxy's total light.

This shortcut isolates the scaling-relation machinery. Real measurements have uncertainties from noise, blending
and light-model choices; treating their values as fixed does not propagate those uncertainties into the inferred
masses.
"""
tracer_true = al.from_json(file_path=dataset_path / "tracer.json")

lens_galaxies_true = [
    galaxy for galaxy in tracer_true.galaxies if galaxy.redshift == 0.5
]

centres = [
    tuple(float(value) for value in galaxy.bulge.centre)
    for galaxy in lens_galaxies_true
]

luminosities = [
    galaxy.bulge.luminosity_within_circle_from(radius=mask_radius)
    for galaxy in lens_galaxies_true
]

print(f"Lens galaxy centres:      {centres}")
print(f"Lens galaxy luminosities: {luminosities}")

"""
__The Anchor__

We choose one galaxy as the **anchor**: its Einstein radius is a free parameter, and the other galaxies' Einstein
radii are linked to it through their measured luminosity ratios.

Here, we select the galaxy with the largest measured aperture luminosity using `argmax`. This provides a
convenient reference; it does not establish that the brightest galaxy has a precisely known mass or must follow
the relation perfectly.

The anchor's Einstein radius sets the normalization of the relation and is inferred from the lensing data.
Because it is already a parameter of the anchor's mass profile, tying additional galaxies to it introduces no
separate normalization parameter.

We also fix the exponent to 0.5. Allowing the exponent or other properties of the relation to vary would add
shared parameters, rather than necessarily adding one parameter per galaxy.
"""
anchor_index = int(np.argmax(luminosities))

luminosity_anchor = luminosities[anchor_index]
centre_anchor = centres[anchor_index]

tied_indexes = [i for i in range(len(centres)) if i != anchor_index]

print(f"Anchor galaxy index: {anchor_index}, L_anchor = {luminosity_anchor:.4f}")

"""
__Over Sampling__

As in Tutorial 2, we apply adaptive over-sampling around both measured galaxy centres so that their steep central
light profiles are evaluated accurately.

These positions determine where the light calculation receives finer sampling. Fixing a light or mass profile's
centre remains a separate model-composition step, which we perform below.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Light Via MGE__

We represent each lens galaxy's light with an MGE, constructed using `al.model_util.mge_model_from`.

We fix its centre to the measured light position. The Gaussian widths are also fixed, and their intensities are
solved by the linear inversion.

If the Gaussians were spherical, this would leave **zero non-linear light parameters**, regardless of the number
of Gaussians.

Here, we retain two free elliptical components shared by the Gaussians, allowing the light model to be flattened
and rotated. Each galaxy therefore contributes **two non-linear light parameters**.

The anchor also receives an elliptical `Isothermal` mass profile with its centre fixed to the measured position.
Its two elliptical components and Einstein radius remain free.

The anchor's Einstein radius serves two purposes: it determines the normalization of its own mass profile and
sets the normalization of the scaling relation for the other galaxies.
"""
bulge_anchor = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=10,
    centre=centre_anchor,
    centre_fixed=centre_anchor,
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

mass_anchor = af.Model(al.mp.Isothermal)
mass_anchor.centre = centre_anchor

lens_anchor = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge_anchor,
    mass=mass_anchor,
)

"""
The source is also represented by an MGE, but its centre remains free because its unlensed position is inferred
through the lens model.

Its Gaussian widths are fixed and its intensities are solved linearly. The shared centre coordinates and
elliptical components contribute four non-linear source parameters.
"""
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=10,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

r"""
__Scaling Relation__

We now assign the other galaxy a spherical `IsothermalSph` mass profile and fix its centre to the measured
light position.

Rather than assigning its `einstein_radius` an independent prior, we express it in terms of the anchor's:

$$
\theta_{{\rm E},i}
=\theta_{{\rm E},{\rm anchor}}
\left(\frac{L_i}{L_{\rm anchor}}\right)^{0.5}.
$$

Whenever the search proposes an Einstein radius for the anchor, the tied galaxy's Einstein radius follows
automatically. It is a **derived parameter**, with no independent sampling dimension.

Three choices together give this galaxy zero non-linear mass parameters: its centre is fixed, its profile is
spherical, and its Einstein radius is tied to the anchor. The scaling relation itself removes only the independent
Einstein-radius parameter; fixing the centre and adopting a spherical profile provide the other reductions.

Its elliptical MGE still contributes two non-linear light parameters. A spherical MGE with fixed widths and
centre would reduce that light contribution to zero as well.

We use this familiar two-galaxy dataset to demonstrate the machinery. However, both galaxies contribute
substantially to the lensing, so tying their masses imposes a strong restriction on a quantity the previous
tutorial allowed the data to constrain.

An independent mass model provides a useful comparison for such important deflectors. A scaling relation may be
appropriate when supported by the scientific assumptions, but its apparent precision must not be mistaken for
information supplied entirely by the lensing data.

The simplification becomes especially useful for populations of less influential galaxies. We can tie that
population while retaining independent mass parameters for galaxies whose contributions need more flexibility.
"""
scaling_exponent = 0.5

scaling_galaxies_list = []

for i in tied_indexes:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre=centres[i],
        centre_fixed=centres[i],
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centres[i]
    mass.einstein_radius = (
        mass_anchor.einstein_radius
        * (luminosities[i] / luminosity_anchor) ** scaling_exponent
    )

    scaling_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
__Model__

We assemble the model using two collections:

- `galaxies` contains the anchor lens galaxy and the source.
- `scaling_galaxies` contains the galaxies whose mass normalizations follow the scaling relation.

Both collections contribute to the lens model. Keeping the tied population separate makes its role easier to
inspect and follows the convention used in the workspace examples.

Inspect `model.info` below. The tied galaxy's `einstein_radius` is expressed in terms of the anchor's parameter
rather than assigned an independent prior.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens_anchor, source=source),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
The figure shows how the tied galaxy's Einstein radius depends on the anchor's.

It has no independent prior or sampling dimension, but it remains a parameter of the galaxy's mass profile. Its
value changes as the search varies the anchor's Einstein radius.

With fixed luminosity ratios and a fixed exponent, uncertainty in the anchor propagates directly to the tied
Einstein radius. This does not include uncertainty in the luminosity measurements or intrinsic scatter around
the scaling relation.
"""
af.ModelPlotter(model).figure()

"""
__Parameter Counts__

We now compare models with different amounts of mass freedom.

First, we give each lens galaxy an independent elliptical `Isothermal` mass profile with a fixed centre. We
retain the same fixed-centre MGE light models and the same source model used above.

Each lens galaxy then contributes two non-linear light parameters and three mass parameters. Including the
source's four parameters gives **14** in total.

Our scaling-relation model has **11**: five for the anchor's light and mass, two for the tied galaxy's light,
and four for the source.

The reduction from 14 to 11 combines two changes to the second galaxy: adopting a spherical mass profile removes
two elliptical components, and tying its Einstein radius removes one further parameter.

This comparison differs from Tutorial 2's 18-parameter model because we have also fixed both lens galaxies'
light centres here.
"""
per_galaxy_lens_dict = {}

for i in range(len(centres)):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre=centres[i],
        centre_fixed=centres[i],
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre = centres[i]

    per_galaxy_lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy, redshift=0.5, bulge=bulge, mass=mass
    )

model_per_galaxy = af.Collection(
    galaxies=af.Collection(**per_galaxy_lens_dict, source=source),
)

print(
    f"Free parameters, independent elliptical masses: {model_per_galaxy.prior_count}"
)
print(f"Free parameters, scaling-relation model:               {model.prior_count}")

"""
To isolate the effect of the scaling relation itself, we build another model with exactly the same profiles,
fixed centres and light-model choices as the tied model.

The only change is that the second galaxy's Einstein radius receives an independent uniform prior instead of
being linked to the anchor.

This model has **12** free non-linear parameters, compared with **11** for the tied model. The scaling relation
therefore saves one parameter for this one tied galaxy.

The assertion below checks that the difference equals the number of tied galaxies.
"""
freed_galaxies_list = []

for i in tied_indexes:
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=10,
        centre=centres[i],
        centre_fixed=centres[i],
        sigma_min=dataset.pixel_scales[0] / 10.0,
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centres[i]
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    freed_galaxies_list.append(
        af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
    )

model_freed = af.Collection(
    galaxies=af.Collection(lens=lens_anchor, source=source),
    scaling_galaxies=af.Collection(freed_galaxies_list),
)

print(f"\nGalaxies tied by the relation:    {len(scaling_galaxies_list)}")
print(f"Free parameters, tier tied:       {model.prior_count}")
print(f"Free parameters, tier freed:      {model_freed.prior_count}")

assert model_freed.prior_count - model.prior_count == len(scaling_galaxies_list)

"""
__Scaling To Many Galaxies__

Saving one parameter becomes more consequential when we model a population of galaxies.

Below, we add twelve hypothetical members with specified positions and luminosities. These galaxies are not
present in our simulated image: we construct these models only to compare parameter counts, and do not fit them
to the data.

Each additional member receives a spherical mass profile with a fixed centre. In one model, its Einstein radius
follows the anchor through the scaling relation. In the comparison model, its Einstein radius has an independent
prior.

We omit light profiles for these hypothetical members to isolate the mass-parameter count. In a real analysis,
their light would need to be assessed separately, as in Tutorial 1. Being a faint or secondary deflector does
not automatically make its light negligible.

The original two-galaxy model is unchanged in both comparisons. In particular, its original tied galaxy remains
tied; only the twelve additional members differ.

The tied model therefore remains at **11** non-linear parameters, while giving the twelve new members independent
Einstein radii increases the count to **23**.

This demonstrates a constant mass-parameter count under our chosen restrictions. Evaluating the extra deflections
still adds computational work, and any additional light-model freedom or per-galaxy scatter parameters would
need to be counted separately.
"""
total_members = 12

member_centres = [
    (2.5 * float(np.sin(angle)), 2.5 * float(np.cos(angle)))
    for angle in np.linspace(0.0, 2.0 * np.pi, total_members, endpoint=False)
]

member_luminosities = [
    float(luminosity)
    for luminosity in np.linspace(0.05, 0.5, total_members) * luminosity_anchor
]

members_tied_list = []
members_freed_list = []

for centre, luminosity in zip(member_centres, member_luminosities):
    mass_tied = af.Model(al.mp.IsothermalSph)
    mass_tied.centre = centre
    mass_tied.einstein_radius = (
        mass_anchor.einstein_radius
        * (luminosity / luminosity_anchor) ** scaling_exponent
    )

    members_tied_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass_tied))

    mass_freed = af.Model(al.mp.IsothermalSph)
    mass_freed.centre = centre
    mass_freed.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    members_freed_list.append(af.Model(al.Galaxy, redshift=0.5, mass=mass_freed))

model_many_tied = af.Collection(
    galaxies=af.Collection(lens=lens_anchor, source=source),
    scaling_galaxies=af.Collection(scaling_galaxies_list + members_tied_list),
)

model_many_freed = af.Collection(
    galaxies=af.Collection(lens=lens_anchor, source=source),
    scaling_galaxies=af.Collection(scaling_galaxies_list + members_freed_list),
)

print(f"Free parameters, 2-galaxy scaling model:            {model.prior_count}")
print(
    f"Free parameters, 14-galaxy scaling model:           {model_many_tied.prior_count}"
)
print(
    f"Free parameters, 12 additional members with free Einstein radii: {model_many_freed.prior_count}"
)

assert model_many_tied.prior_count == model.prior_count

"""
Adding twelve mass-only galaxies through the fixed scaling relation leaves the non-linear parameter count
unchanged. Their positions are fixed, their profiles are spherical, and their Einstein radii all depend on the
existing anchor parameter.

This does not mean the additional galaxies are computationally free: their deflections must still be evaluated.
It also does not include any extra freedom we might introduce for their light, positions or deviations from
the relation.

__Model Fit__

We now fit the original **two-galaxy scaling-relation model** to the data. The twelve hypothetical members were
used only for the parameter-counting exercise and are not included in this fit.

We use the familiar `Nautilus` search and `AnalysisImaging` object. The model has 11 non-linear parameters, but its
smaller parameter space reflects stronger assumptions about the galaxies as well as fewer independent
quantities to infer.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_4",
    name="tutorial_3_scaling_relation",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun. Check the output/howtolens/chapter_4/tutorial_3_scaling_relation"
    " folder for results and visualizations as the search progresses."
    " This cell will finish when the search completes."
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished — you may now continue the notebook.")

"""
__Results__

Inspect `result.info` and the maximum log likelihood fit below.

The tied galaxy's Einstein radius is derived from the anchor's Einstein radius through the fixed luminosity
ratio and exponent. It has no independently sampled prior.

Its inferred uncertainty therefore reflects uncertainty in the anchor within this model. It does not include
measurement uncertainty in the luminosity ratio, uncertainty in the exponent or intrinsic scatter around
the relation.

Check the residuals around both galaxies and the lensed source. A compact model is useful only if its assumptions
provide an adequate description of the data.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
Because this dataset is simulated, we can check how closely the input galaxy masses satisfy the relation we
imposed.

Below, we use the anchor's true Einstein radius and the measured aperture-luminosity ratio to predict the other
galaxy's Einstein radius, then compare that prediction with its true value.

This tests the mass-normalization relation. It does not test every assumption in the fitted model: for example,
the simulated second galaxy has an elliptical mass profile, while our tied model represents it as spherical.

A visually satisfactory fit would not by itself demonstrate that either assumption is exact. Other parameters
may compensate for an imperfect scaling relation or profile choice.
"""
einstein_radii_true = [
    float(galaxy.mass.einstein_radius) for galaxy in lens_galaxies_true
]

for i in tied_indexes:
    einstein_radius_predicted = einstein_radii_true[anchor_index] * (
        (luminosities[i] / luminosity_anchor) ** scaling_exponent
    )

    print(f"Tied galaxy {i}: true einstein_radius      = {einstein_radii_true[i]:.3f}")
    print(
        f"Tied galaxy {i}: relation predicts         = {einstein_radius_predicted:.3f}"
    )

r"""
Use the printed values to assess the agreement for this dataset and our chosen luminosity aperture. The
simulated masses were specified independently, so agreement with the scaling relation is something to check,
rather than a built-in guarantee.

__Limitations__

A scaling relation reduces parameter freedom by introducing assumptions about the galaxy population.

**Intrinsic scatter.** Real galaxies do not lie exactly on a single luminosity-mass relation. Our deterministic
model allows no independent deviation for a tied member. Even when individual members have small effects, their
modeling errors do not necessarily cancel: the lensing impact depends on where those galaxies lie relative to
the images.

**Measurements and membership.** Luminosities and centres have measurement uncertainties, especially in blended
images. Incorrectly assigning a foreground or background galaxy to the lens population can also give it an
inappropriate luminosity and lensing geometry.

**The adopted profiles.** We assume spherical mass profiles for tied members. Deviations in shape, radial
structure or tidal truncation can matter independently of whether the luminosity-normalization relation is
accurate.

**The anchor.** The anchor's fitted Einstein radius normalizes the entire population. If the anchor is not
representative of that population, the imposed relation can give inappropriate masses to other members.

These restrictions can lead to biased inferences or uncertainties that are too small. Residuals are useful
diagnostics, but a good image fit does not guarantee that the assumed mass decomposition is correct.

A practical response is **selective freedom**: retain the scaling relation for a population of less influential
members while giving important galaxies independent mass parameters. Galaxies close to the lensed images may
warrant this even when they are faint.

Other approaches include fitting shared normalization or slope parameters, or explicitly modeling scatter.
Each changes the assumptions and parameter count, and should be assessed against the scientific goal and
available constraints.

__Lens Environments__

Scaling relations are also useful for the environment of a lens dominated by one galaxy. A population of nearby
galaxies can collectively perturb the lensing, even when no single member is a major deflector.

For galaxies belonging to the same physical population at approximately the same redshift, measured positions
and consistent luminosities can support a shared relation. The main lens can serve as an anchor if that
assumption is appropriate; it need not always be the best reference.

Galaxies projected nearby on the sky are not necessarily members of that population. Foreground and background
galaxies require appropriate redshifts, luminosity estimates and lensing-distance factors. We should not tie them
to the main lens using observed flux ratios alone.

As throughout this chapter, their light and mass require separate consideration. A mass-only member adds no light
parameters, but its emission must still be negligible, excluded appropriately or modeled.

__Wrap Up__

In this tutorial, we learned:

1. A luminosity-based scaling relation links galaxy mass normalizations through shared assumptions, rather than
   measuring each mass independently.
2. With fixed centres, spherical mass profiles, a fixed exponent and an existing anchor parameter, additional
   tied galaxies add zero non-linear mass parameters.
3. Light-model freedom must be counted separately. Fixed-centre elliptical MGEs add two parameters per galaxy;
   fixed-centre spherical MGEs with fixed widths add none.
4. Reduced dimensionality does not remove computational cost or uncertainty associated with the relation's
   assumptions.
5. Important galaxies can retain independent mass freedom while less influential members follow a shared relation.

In the next tutorial, we move to **group-scale lensing**, adding a shared dark matter halo alongside the
member galaxies.

We will also introduce a relation normalized at a fixed reference luminosity, rather than anchored to one galaxy,
and truncated mass profiles for describing member galaxies whose outer mass distributions may have been altered
by tidal stripping.
"""
