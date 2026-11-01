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

The relation needs an "anchor": one galaxy whose Einstein radius is a free parameter of the model, off which
every other galaxy's Einstein radius hangs. We use the brightest galaxy, identified by `argmax` over the
measured luminosities — a measurement, not an assumption about which galaxy happens to be listed first.

Anchoring on a galaxy the model is already fitting is the key trick: the anchor's `einstein_radius` is not a new
parameter, so the relation itself will add **zero** free parameters to the model.
"""
anchor_index = int(np.argmax(luminosities))

luminosity_anchor = luminosities[anchor_index]
centre_anchor = centres[anchor_index]

tied_indexes = [i for i in range(len(centres)) if i != anchor_index]

print(f"Anchor galaxy index: {anchor_index}, L_anchor = {luminosity_anchor:.4f}")

"""
__Over Sampling__

As in the previous tutorial, we apply adaptive over-sampling centred on both lens galaxies, so the steeply
varying central light of each is computed accurately.
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

Every galaxy's light is modelled with a Multi-Gaussian Expansion, composed via the `al.model_util.mge_model_from`
convenience function (this wraps the `Basis` of linear `Gaussian` profiles we built by hand in chapter 2 into a
single call — the composition it returns is the same).

Because the galaxy centres were measured above, we fix each MGE's centre to them (passing the measured centre as
both the `centre` and `centre_fixed` inputs, which fixes it rather than giving it a prior). The Gaussian
`sigma` values are fixed by the basis and the intensities are solved for by linear algebra, so each galaxy's
light costs just **two** non-linear parameters: its elliptical components. This is the "light adds (almost) no
parameters" half of the recipe from the introduction.

The anchor galaxy also gets its mass: a free `Isothermal` profile with its centre fixed at the measured light
centre. Its `einstein_radius` is an ordinary free parameter — but it is also what the scaling relation below
hangs off.
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
The source galaxy is also an MGE, with a free centre (we have no pre-measured position for the unlensed source).
"""
source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=10,
    gaussian_per_basis=1,
    centre_prior_is_uniform=False,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

"""
__Scaling Relation__

Now the relation itself. The fainter galaxy's mass is a spherical isothermal profile whose `einstein_radius` is
not given a prior — instead, it is *assigned an expression*: the anchor's `einstein_radius` multiplied by the
luminosity ratio raised to the Faber-Jackson exponent of 0.5.

This single line is the whole trick. Because `mass_anchor.einstein_radius` is the model's own free parameter,
multiplying it by a fixed number produces a **derived quantity**, not a new parameter. Whenever the non-linear
search proposes an Einstein radius for the anchor, every tied galaxy's Einstein radius follows automatically.

The tied galaxy keeps its MGE bulge (its light is inside the mask and must be fitted), so it costs two light
parameters and **zero** mass parameters.

An honest aside before we continue: this dataset's two galaxies are comparably bright, and for a genuine system
of two co-dominant deflectors the previous tutorial's model — full mass freedom for both — is the right choice.
A galaxy contributing half the lensing deserves its own parameters. We tie one here so you can learn the
machinery on a familiar dataset; the regime where the relation genuinely earns its keep is a *population* of
fainter members, which we build towards below and which dominates the next tutorial.
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

The model has two top-level collections: `galaxies` (the anchor and the source) and `scaling_galaxies` (the tied
population). This is not just presentational — `scaling_galaxies` is the collection name **PyAutoLens** expects a
scaling population in: the analysis appends it to the tracer's galaxy list when fitting, and results loaded back
via the aggregator restore it. It also keeps `model.info` readable, with the tied population visibly separate
from the freely modelled galaxies.

Inspecting `model.info` below, note that the tied galaxies have no `einstein_radius` prior of their own — it is
listed as a function of the anchor's.
"""
model = af.Collection(
    galaxies=af.Collection(lens=lens_anchor, source=source),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free,
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed
parameter.

The tied galaxy's `einstein_radius` is not sampled at all. It is computed from the anchor lens galaxy's via the
scaling relation, so it has no prior, because it is not a parameter.
"""
af.ModelPlotter(model).figure()

"""
__Parameter Counts__

Let's verify the claims above by counting, rather than believing. First we recompose the previous tutorial's
style of model on this dataset — every galaxy with its own free elliptical `Isothermal` mass (centres fixed at
the measured light centres, as above, so the comparison is like-for-like).
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
    f"Free parameters, per-galaxy model (previous tutorial): {model_per_galaxy.prior_count}"
)
print(f"Free parameters, scaling-relation model:               {model.prior_count}")

"""
The scaling-relation model is smaller. To isolate exactly what the relation saves, we compose one more variant:
identical to the scaling-relation model in every way, except each tied galaxy's `einstein_radius` is freed with
a uniform prior instead of tied. The difference in parameter count must equal the number of tied galaxies —
one saved `einstein_radius` each.
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

Saving one parameter looks modest — the point is what happens as the population grows. Below we imagine this
lens had twelve additional member galaxies (we invent centres on a ring and plausible luminosities; this model
is composed for counting only, **not** fitted — our dataset contains no such galaxies!).

Each imagined member is mass-only (faint distant perturbers whose light is negligible or outside the mask need
no light model — the mass-only counterpart of the intermediate options tutorial 1 discussed), with its Einstein
radius tied to the anchor by the same relation. We count the parameters of this 14-galaxy lens model, and of its nightmare per-galaxy twin.
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
    f"Free parameters, 14-galaxy per-galaxy-mass model:   {model_many_freed.prior_count}"
)

assert model_many_tied.prior_count == model.prior_count

"""
Twelve more galaxies, zero more parameters. The tied model's dimensionality is *independent of the number of
galaxies* — this is the property that makes group- and cluster-scale lens modeling possible at all, and it is
why the assertion above is worth having in the script rather than in prose.

__Model Fit__

We now fit the scaling-relation model to the data, using the familiar `Nautilus` search and `AnalysisImaging`
object. The parameter space is barely larger than a single-lens fit, which is the whole point.
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
    "The non-linear search has begun running - checkout the output/howtolens/chapter_4/tutorial_3_scaling_relation"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
__Results__

In `result.info`, the tied galaxy has no `einstein_radius` entry of its own — it is reported as a derived
function of the anchor's, which is what a tied parameter looks like in the output. Its posterior uncertainty
comes entirely from the anchor's.
"""
print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
The fit is good — but we should ask *why* it is good, because the answer is the honest heart of this tutorial.

The tied model can only fit well if the true galaxies actually lie close to the relation we imposed. We can
check, because this is simulated data and we know the truth. Below we compare the fainter galaxy's true Einstein
radius against the value the relation predicts from the anchor's true Einstein radius and the luminosity ratio.
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
The prediction lands within a few percent of the truth: this simulated pair happens to sit almost exactly on the
Faber-Jackson relation. That is why the tied fit succeeds. Real galaxies are not always so obliging.

__Limitations__

A scaling relation is an assumption, and it is worth being clear-eyed about what we assumed:

**Scatter.** Faber-Jackson is a statistical relation with real intrinsic scatter — at fixed luminosity,
galaxies show a spread in velocity dispersion, and therefore in Einstein radius. A tied model has no freedom to
absorb this scatter: it asserts every member sits exactly on the relation. For a population of faint members
whose individual lensing effects are small, the errors average down and this is fine. For any single galaxy
that matters a lot to the fit, it is a risk.

**Deviant members.** Worse than scatter is a member that is systematically off the relation — a tidally
stripped satellite that has lost dark matter but kept its stars, a galaxy with an unusual mass-to-light ratio,
or a misclassified foreground/background interloper assigned a luminosity at the wrong distance. Because the
tied model *cannot* adjust that one galaxy, the non-linear search compensates the only way it can: by biasing
the parameters it does control — the anchor's mass, the source — dragging the whole lens model away from the
truth. One bad member can poison an otherwise excellent fit, and nothing in the residuals will neatly point at
the culprit.

**The anchor itself.** Anchoring on the brightest galaxy assumes the anchor sits on the relation too. If it is
the anomalous one, every tied galaxy inherits its anomaly.

The defence against all three is **selective freedom**. The tiers can be mixed within one model: keep the long
tail of faint members tied, but promote any galaxy that is bright, close to the lensed images, or visibly
suspicious to its own free mass profile — exactly as the extra galaxies were treated earlier in this chapter. A
middle option bounds a member's free Einstein radius using its luminosity (free, but only within a factor of the
relation's prediction). And the relation itself can be loosened: freeing `scaling_exponent` as a fit parameter
costs a single dimension *for the entire population* and is a useful systematics test. The `autolens_workspace`'s
`imaging/features/scaling_relation` example shows all of these tiers working together.

__Lens Environments__

Everything above was framed around lenses with multiple deflectors, but the same machinery solves a problem we
met at the start of this chapter: the **environment** of an ordinary single-galaxy lens.

The first tutorial of this chapter modelled a nearby extra galaxy individually, which was fine for one.
But a deep image of a real lens can reveal dozens of galaxies near the line of sight, each contributing a small
perturbation to the deflection field. Modeling them individually is hopeless; ignoring them entirely can bias
precision measurements. With a scaling relation they cost nothing: measure each galaxy's centre and luminosity
from the image, tie the lot to the main lens galaxy (which is the natural anchor — it is usually the brightest
galaxy in the field), and the model gains dozens of physically motivated perturbers with zero added parameters.

__Wrap Up__

In this tutorial we learned:

1. Luminosity predicts lensing mass, via mass-follows-light arguments and the Faber-Jackson relation
   (`L ~ \sigma^4`, `\theta_E ~ \sigma^2`, hence `\theta_E ~ L^{0.5}`).

2. A scaling relation replaces per-galaxy mass freedom with a shared relation anchored on a parameter the model
   already fits, so tied galaxies add **zero** free parameters — verified by explicit parameter counting.

3. Combined with MGE light profiles, a many-galaxy lens model stays low-dimensional: dimensionality is
   independent of the number of galaxies.

4. The relation is an assumption: scatter and deviant members can bias the whole model, and the remedy is
   selective freedom — tie the faint majority, free the important few.

5. The same machinery models the environments of single-lens systems, at no parameter cost.

So far, we tied galaxy to galaxy. But the biggest lenses in the Universe — galaxy groups and clusters — contain
something the galaxies themselves cannot account for: a massive dark matter halo enveloping the whole system,
holding most of its mass. In the next tutorial we step up to group-scale lenses, where the scaling relation
becomes essential (there are simply too many member galaxies to free) and a dark matter halo joins the model as
a new component. There, we will also meet a different way to normalise the relation — a shared free
normalization at a fixed reference luminosity, rather than an anchor galaxy — and truncated mass profiles, which
describe members whose outer dark matter has been tidally stripped by the very halo we are adding.
"""
