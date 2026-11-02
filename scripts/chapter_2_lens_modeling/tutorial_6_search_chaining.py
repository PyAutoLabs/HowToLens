"""
Tutorial 6: Search Chaining
===========================

Look again at our lens image. The foreground galaxy's light, its gravitational deflection and the background source all
contribute to what we observe. Fitting them together requires the search to explore several connected parts of the
model.

Yet we do not always need to keep every part free throughout an analysis. Once we have obtained a satisfactory
description of the foreground light, we can hold aspects of that description fixed while investigating the mass
distribution in greater detail.

**Search chaining** connects a sequence of fits in this way. Each search has a particular purpose, and its result
supplies information used to construct a later analysis.

In this tutorial, an initial fit will describe the lens and source light using multi-Gaussian expansions, alongside an
isothermal mass profile and external shear. We will then fix the fitted lens-light geometry and replace the isothermal
mass profile with a power-law profile whose slope is free.

The second search can therefore investigate a more flexible mass model without simultaneously rediscovering the
foreground light geometry. We will also use the first result to estimate the positions constraint introduced in the
previous tutorial.

__Contents__

- **The Search Chain:** Give each search a distinct role.
- **Dataset and Mask:** Reuse the imaging dataset and circular mask.
- **Multi-Gaussian Expansions:** Construct lens and source light models with the MGE helper.
- **Model 1:** Compose the initial isothermal lens model.
- **Search 1:** Find a candidate solution with Prodigy.
- **Inspecting Search 1:** Check the fit before carrying information forward.
- **Fixing the Lens-Light Geometry:** Use fitted instances while retaining linear intensities.
- **The Power-Law Mass Model:** Allow the radial mass slope to vary.
- **Model 2:** Reduce the search to twelve non-linear parameters.
- **Automatically Estimated Positions:** Derive a positions constraint from the first fit.
- **Search 2:** Explore the revised model and its uncertainties with Nautilus.
- **Result 2:** Inspect residuals and posterior distributions.
- **Conditional Uncertainties:** Understand the consequences of fixed geometry.
- **Search Chaining and SLaM:** Connect this example to reusable pipelines.

__The Search Chain__

Our first search aims to obtain a useful description of the image relatively quickly. We use `MultiStartProdigy`, which
runs gradient-based optimisations from multiple starting points.

This gives us a candidate solution to inspect. It does not provide the posterior samples needed to describe parameter
uncertainties, and we must check its fitted image and residuals before using its result.

Our second search uses Nautilus to explore the revised model more thoroughly. The lens-light centres and shapes will be
fixed to their fitted values, while the mass and source parameters remain free. The light-profile intensities will
still be solved during each likelihood evaluation.

This distinction matters: we are fixing the geometry of the foreground light, rather than subtracting a fixed image and
assuming that its brightness can never change.

Holding fitted quantities fixed is also an assumption. The second result is conditional on that choice and does not
propagate uncertainty in the fixed lens-light geometry. Later in the tutorial, we will discuss when that approximation
is useful and when those parameters should be allowed to vary again."""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset and Mask__

We use the same `lens_sersic` imaging dataset and 3.0-arcsecond circular mask as the previous tutorial.

Inspect the central lens galaxy and the surrounding arcs. The first fit must describe both: an inaccurate source or
mass model can affect how the overlapping light is assigned to the foreground galaxy."""

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

al.util.dataset.simulate_if_needed(
    dataset_path,
    simulator="scripts/simulator/lens_sersic.py",
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Multi-Gaussian Expansions__

We use the MGE construction introduced in *Linear Profiles*. Within each galaxy's Gaussian group, the profiles share a
centre and elliptical shape. Their widths are fixed, and their intensities are solved together during the fit.

The lens and source have separate groups, so their centres and shapes remain independent.

To avoid repeating the construction loops from Tutorial 4, we use `al.model_util.mge_model_from`. This helper builds an
ordinary model from linear Gaussian profiles; it does not fit the data or introduce a different model-composition API.

The lens uses 30 Gaussians, with widths spanning 0.01 to 3.0 arcseconds. The source uses 15 Gaussians spanning 0.0001
to 1.0 arcseconds in the source plane. These ranges provide components on different scales; they do not imply that the
smallest components are individually resolved by the observations.

In this helper, `mask_radius` sets the largest Gaussian width. The source's value of 1.0 therefore controls its
Gaussian widths—it does not apply another mask to the image."""

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=30,
    centre_prior_is_uniform=True,
    sigma_min=dataset.pixel_scales[0] / 10.0,
)

source_bulge = al.model_util.mge_model_from(
    mask_radius=1.0,
    total_gaussians=15,
    centre_prior_is_uniform=False,
    sigma_min=1.0e-4,
)

"""
__Model 1__

Combine these light profiles with an elliptical isothermal mass profile and external shear.

The isothermal profile gives the first search a mass model with a fixed radial slope. In the second search, we will
relax this restriction using a power-law profile.

We do not force the mass to share the lens light's centre or elliptical shape. The first fit must determine those
properties independently from the image.

Despite containing 45 Gaussian light profiles, this model has only 15 non-linear parameters: four describe each
galaxy's shared light geometry, five describe the isothermal mass profile, and two describe external shear. The
Gaussian widths are fixed and their intensities are solved linearly.

Inspect `model_1.info` and its diagram to identify those independent parameters before fitting.
"""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge,
    mass=al.mp.Isothermal,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

model_1 = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

print(model_1.info)

af.ModelPlotter(model_1).figure()

"""
__Search 1__

We begin with `MultiStartProdigy`, using the settings introduced in the optional searches tutorial: 48 starting points
and up to 300 gradient steps per start.

Multiple starting points give the optimiser opportunities to find different solutions. They do not guarantee that it
finds the best one, so we will inspect the result before carrying any information forward.

`batch_size=4` limits how many starts are processed together, reducing memory use. It does not reduce the total number
of starts.

The analysis fits the imaging data without a positions constraint. After this search, we will use the fitted model to
estimate positions for the next analysis.

Prodigy's job here is to find a good-fitting set of parameter values. It does not attempt to map out the full range of
plausible solutions or calculate posterior uncertainty intervals—the parameter "errors".

This narrower task is one reason we expect it to finish sooner than a thorough Nautilus run, alongside its use of
gradients. It is therefore useful for our first stage, where we need a satisfactory model to build on rather than a
final measurement of every parameter's uncertainty. The speed advantage still depends on the problem and settings."""

search_1 = af.MultiStartProdigy(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_6_search_1",
    unique_tag=dataset_name,
    n_starts=48,
    n_steps=300,
    batch_size=4,
)

analysis_1 = al.AnalysisImaging(dataset=dataset, use_jax=True)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
JAX supplies the gradients used by Prodigy. Its initial compilation adds a setup cost; later evaluations reuse the compiled calculation. The total run time depends on the hardware, model and number of optimisation steps required.

__Inspecting Search 1__

Inspect the fitted image and normalised residuals before proceeding.

Look first at the foreground galaxy. Does the MGE reproduce its centre, shape and extended light? Then examine the
arcs. A poor description of the source can affect how the fit divides overlapping light between the two galaxies.

Search chaining carries information forward, including mistakes. Fixing the foreground geometry from a poor fit can
prevent the next search from correcting it.

If substantial residual structure remains, investigate the first fit using the approaches from *Why Modeling Is Hard*.
More optimisation effort may help, but the model may also need additional freedom.

A satisfactory result gives us a candidate lens-light geometry for the next stage. It does not establish its
uncertainty or prove that the solution is unique."""

print(result_1.info)

aplt.subplot_fit_imaging(fit=result_1.max_log_likelihood_fit)

"""
__Fixing the Lens-Light Geometry__

Until now, we have mainly constructed components using `af.Model`, which leaves selected parameters free for a search
to fit.

`result_1.instance` instead contains components with numerical parameter values from the highest-likelihood solution
stored in the first result. We can take its fitted lens-light basis and place that instance into our next model.

Its Gaussian centres and elliptical shapes are then fixed. The widths were already fixed in the first search.

The Gaussian profiles remain **linear light profiles**, however. Their intensities will still be solved during each
likelihood evaluation of the second fit. We therefore preserve the fitted geometry while allowing the brightness
assigned to the lens and source to adjust together.

This differs from subtracting the first fit's lens-light image from the data. We continue to fit the original image,
with foreground and background light included in the prediction."""

lens_bulge_fixed = result_1.instance.galaxies.lens.bulge

"""
__The Power-Law Mass Model__

The first search used an isothermal mass profile, whose radial density slope is fixed. We now replace it with
`PowerLaw`, allowing the slope to vary.

The slope controls how quickly the mass density falls with radius. The isothermal case corresponds to `slope=2.0`;
allowing this parameter to vary lets the second search investigate steeper and shallower profiles.

This extra freedom is scientifically useful, but it also introduces relationships with other parameters, including the
source size. We should not assume that adding the slope makes it independently or precisely measurable.

We retain free mass-centre coordinates, ellipticity and Einstein radius, alongside external shear. The lens-light
geometry is fixed separately: we are not forcing the mass distribution to follow the light."""

lens_2 = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge_fixed,
    mass=al.mp.PowerLaw,
)

field_2 = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

"""
__Model 2__

The source MGE remains free to change its centre and shape. This is important because changing the mass profile changes
how the observed arcs map back to the source plane.

We construct a fresh source model with the same settings as before. Its priors are not replaced with narrow
distributions derived from the first result.

Combining it with the revised lens gives 12 non-linear parameters: six for the power-law mass profile, two for external
shear and four for the source geometry.

We have added one mass parameter but removed four lens-light geometry parameters from the search. The second model
therefore has a more flexible mass profile and fewer non-linear parameters overall.

Twelve non-linear parameters is fewer than we might expect for a model that includes foreground lens light, a flexible
mass profile, external shear and source light. The lens light is still present, but its fitted geometry no longer
contributes free parameters to the search, and its intensities are handled by the linear solve.

This is a central benefit of the chain: we can retain a detailed description of the image while asking the second
search to explore fewer parameters.

Compare the model summary and diagram with those from the first search. The foreground Gaussian geometry should now
have fixed numerical values, while the source geometry, mass parameters and shear remain free. The lens and source
intensities are still solved linearly."""

source_bulge_2 = al.model_util.mge_model_from(
    mask_radius=1.0,
    total_gaussians=15,
    centre_prior_is_uniform=False,
    sigma_min=1.0e-4,
)

source_2 = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_2)

model_2 = af.Collection(
    galaxies=af.Collection(lens=lens_2, source=source_2),
    fields=field_2,
)

print(model_2.info)

af.ModelPlotter(model_2).figure()

"""
__Automatically Estimated Positions__

In the previous tutorial, we supplied image positions explicitly. We can now estimate them from the first fitted model.

The result uses its fitted source location and mass distribution to predict the corresponding multiple-image positions.
It then constructs a `PositionsLH` constraint for the next analysis.

This carries a second kind of information between searches. Alongside the fitted lens-light geometry, we use the first
model's lensing geometry to help guide the next search.

These positions are predictions from the first fit, rather than independently measured coordinates. Plot them over the
image and check that they correspond sensibly to the observed arcs. A poor first model can generate misleading
positions as well as an inaccurate light distribution."""

positions_likelihood = result_1.positions_likelihood_from(
    factor=3.0,
    minimum_threshold=0.2,
)

aplt.subplot_imaging_dataset(
    dataset=dataset,
    positions=positions_likelihood.positions,
)

print("Source-plane threshold:", positions_likelihood.threshold)

"""
The threshold is three times the source-plane separation calculated from the first model, with a minimum of 0.2 arcseconds. The minimum prevents the numerical precision of the predicted positions from imposing an unnecessarily tight constraint.

As before, this is a broad consistency check rather than a positional-error measurement. Its calculation during the
search is extremely fast compared with evaluating the full imaging likelihood.

__Search 2__

We now return to Nautilus because we want more than a single good-fitting solution. We want posterior samples that
describe the range of plausible parameter values and allow us to calculate uncertainty intervals—the parameter
"errors".

Exploring those distributions requires more work than locating one solution, so we generally expect this stage to run
more slowly than the initial Prodigy optimisation. Fixing the lens-light geometry reduces that burden.

Pass the positions constraint to a new Analysis object and fit `model_2`. We continue to use the original imaging data,
including the foreground light.

The resulting uncertainties are conditional on the geometry we fixed. Nautilus explores the free parameters of this
second model; it cannot recover uncertainty in parameters we have removed from the search."""

analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    use_jax=True,
)

search_2 = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_6_search_2",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result 2__

Inspect the fitted image and normalised residuals again. Check both the arcs and the foreground galaxy: the fixed
lens-light geometry is still an assumption that the data can reveal to be inadequate.

Compare the result with the first fit. The power-law slope is now free, so the mass model and reconstructed source may
change together. A change in the inferred source size need not indicate a failure; it can reflect a relationship
between source structure and the radial mass profile.

This dataset was simulated with an isothermal mass profile, corresponding to a power-law slope of 2.0. Use that known
value as a check, while recognising that noise, the adopted light models and the fixed lens-light geometry can affect
the inferred result.

Examine the posterior distributions as well as the highest-likelihood fit. Look for broad intervals and correlations,
particularly between the mass slope and source parameters. Those distributions contain information that the first
optimisation did not provide."""

print(result_2.info)

aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)

aplt.corner_anesthetic(samples=result_2.samples)

"""
__Conditional Uncertainties__

The second search answers a specific question: which mass and source parameters are consistent with the data **given
the lens-light geometry we carried forward**?

It allows the Gaussian intensities to adjust, but it does not explore alternative foreground centres or shapes. If
those geometric quantities are uncertain, their uncertainty can affect how the overlapping light is divided between the
lens and source.

Fixing them can therefore make the reported mass and source uncertainties narrower than those from an analysis that
also explores the foreground geometry. It can also shift the inferred parameters if the fixed geometry is inaccurate.

Search chaining makes this trade-off useful by allowing us to investigate the model in stages. It does not make the
trade-off disappear.

For a final scientific analysis, we can test its importance by allowing the lens-light geometry to vary again,
repeating the fit with other plausible foreground descriptions, or constructing an analysis that propagates the
relevant uncertainty.

We should also investigate whether the automatically estimated positions constraint is restrictive enough to influence
our conclusions. It was derived from an earlier fitted model, not from an independent observation.

**🤖 Ask AI:** My chained fit gives a narrow uncertainty interval on the mass slope. How could fixing the foreground light geometry affect that interval, and what comparisons would help me assess it?

__Search Chaining and SLaM__

Our two searches had different jobs. The first used a relatively fast optimisation to obtain a description of the
foreground light and an initial lens model. The second fixed the foreground geometry, introduced a more flexible mass
profile and used Nautilus to explore the remaining parameter uncertainties.

The second model contained fewer non-linear parameters even though its mass profile was more flexible. This is one of
the main benefits of fitting different parts of the system in stages.

More elaborate chains follow the same principle. An initial source model can help configure a later source
reconstruction, a foreground-light fit can support a more detailed mass analysis, and an earlier lens model can supply
positions constraints for subsequent searches.

PyAutoLens provides **SLaM**, short for **Source, Light and Mass**, to organise these stages into reusable modeling
pipelines. The workspace's `scripts/imaging/features/multi_gaussian_expansion/slam.py` provides a concrete example.

A pipeline automates the sequence, but each stage still needs a suitable model and a result worth carrying forward.
Inspecting the intermediate fits remains essential.

In Chapter 3, we will introduce pixelized source reconstructions, which describe source structure beyond the smooth
profiles used here. Search chaining will provide a way to connect those more detailed reconstructions to the tools we
have already learnt."""
