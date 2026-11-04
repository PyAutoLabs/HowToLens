"""
Optional Tutorial: Historical Prior Passing
===========================================

This optional tutorial preserves a technique used in earlier **PyAutoLens** modelling workflows, up to around 2025:
passing parameter constraints from one search into the priors of a later search.

For example, an initial search might constrain the Einstein radius. A subsequent search could use a narrower prior
centred on that inferred value, reducing the region it needs to explore. The prior-passing API automated this
construction using the results of the earlier fit.

This approach is no longer part of the recommended workflow taught in this chapter. Tutorial 6 instead demonstrates
a chain that fixes selected lens-light geometry, retains fresh priors for the parameters being explored and uses
the first result to construct a positions penalty. Search chaining does not require passing narrowed priors
between searches.

The distinction matters because an earlier fit can be incomplete or trapped at a local maximum. Restricting a later
search around that result can prevent it from finding alternatives. Using the same data again with priors derived
from its earlier fit also requires care when interpreting the resulting uncertainties.

New readers can skip this tutorial and continue to chapter 3. It is retained to help readers understand older
scripts and analyses, rather than as an additional technique they need to learn before modelling their own lenses.

Prior passing is still supported through `result.model_centred`. This tutorial runs two searches to demonstrate
that API. The historical distinction is its role in our workflows: it used to be central to search chaining,
whereas the main chapter now uses fresh priors and fixes selected components instead.

`result.model` preserves the input priors. `result.model_centred` creates new priors centred on the inferred
parameter values. `result.instance` supplies fitted values that are held fixed. We will inspect the second model
to see which parameters receive centred priors and which receive fresh priors.

__Contents__

- **Initial Setup:** Load the lens_sersic imaging dataset.
- **Dataset Auto-Simulation:** Simulate the dataset via its simulator script if it is not on your hard-disk.
- **Model:** Compose the lens model fitted to the data.
- **Search:** Configure the non-linear search used to fit the model.
- **Result (Search 1):** The results of search 1, which are used for prior passing.
- **Prior Passing:** Use the prior passing API to pass the results of search 1 without writing out values manually.
- **Result:** Overview of the results of the model-fit.
- **Wrap Up:** Summary of the script and next steps.
- **Detailed Explanation Of Prior Passing:** A detailed overview of how prior passing works and tools to customize it.
- **EXAMPLE:** Compare absolute and relative widths for a Sersic index.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
__Initial Setup__

We use the lens_sersic simulated imaging dataset, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens system uses an `Isothermal` galaxy mass profile and a separate `ExternalShear` field.
 - The source galaxy's light is an `ExponentialCore`.

All the usual steps for setting up a model fit (masking, analysis, etc.) are included below.
"""
dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/lens_sersic.py"
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
__Model__

The first search fits a simpler model: the lens-light centre and Sersic index are fixed, the mass centre is fixed,
and the mass and light share their ellipticity. The second search will free some of these choices and use a more
flexible source profile. This tutorial performs its own fits; it does not require saved results from Tutorial 6.
"""
bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

bulge.centre_0 = 0.0
bulge.centre_1 = 0.0
mass.centre_0 = 0.0
mass.centre_1 = 0.0

mass.ell_comps = bulge.ell_comps

bulge.sersic_index = 4.0

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.ExponentialCore)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free,
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed
parameter.

Keep this model in mind when inspecting the second model below: changing priors and changing the physical
components are separate choices.
"""
af.ModelPlotter(model_1).figure()

"""
__Search__

We run Nautilus to obtain a fit and posterior samples for the first model. Inspect its inferred parameters before
using them to construct priors for the second search. As with the earlier tutorials, a completed search does not
by itself establish that the solution is the global maximum.
"""
search_1 = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_optional"),
    name="tutorial_prior_passing_1",
    unique_tag=dataset_name,
    n_live=100,
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarised in the `info` attribute.
"""
print(result_1.info)

"""
__Prior Passing__

We are now going to use the prior passing API to pass these results, in a way which does not require us to manually
write out the inferred parameter values of each component. The details of how prior passing is performed will be
expanded upon at the end of the tutorial.

We start with the lens's bulge, which in the previous search was an `Sersic` with its centre fixed to (0.0, 0.0)
and its `sersic_index` fixed to 4.0. The API for passing priors is shown below and there are two things worth noting:

 1) We pass the priors using the `model_centred` attribute of the result. This informs **PyAutoLens** to pass the result as a
 model component that is to be fitted for in the next search, using priors that are initialized from the previous
 search's result. Note, if we pass through `model_centred` a parameter that was fixed in search 1 (e.g. the `sersic_index`) it
 will be fixed to the same value in search 2.

 2) We do not pass the `centre` or `sersic_index` using `model_centred`, because it would be fixed to the values that it was in
 the first search. By omitting the centre, it uses the default priors on a lens galaxy, whereas we manually tell the
 Sersic index to use a `TruncatedGaussianPrior` centred on 4.0.
"""
bulge = af.Model(al.lp_linear.Sersic)

bulge.ell_comps = result_1.model_centred.galaxies.lens.bulge.ell_comps
bulge.effective_radius = result_1.model_centred.galaxies.lens.bulge.effective_radius
bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=4.0, sigma=2.0, lower_limit=0.0, upper_limit=5.0
)

"""
For the mass, we again must account for how its centre was fixed to (0.0", 0.0") and therefore not pass the centre.
The external shear belongs to its own field, so we pass that field model separately. Taking it from
`result_1.model_centred.fields` keeps its parameters free with the priors learned in the first search; taking
`result_1.instance.fields` would fix them to the fitted values instead.
"""
mass = af.Model(al.mp.Isothermal)
mass.einstein_radius = result_1.model_centred.galaxies.lens.mass.einstein_radius
mass.ell_comps = result_1.model_centred.galaxies.lens.mass.ell_comps
field = result_1.model_centred.fields

"""
For the source's bulge, we are passing the result of an `ExponentialCore` to an `Sersic`.

We can use a special prior passing method to do this, called `take_attributes`. This scans the `ExponentialCore`
passed to the `take_attributes` method for all parameters which have the same name as the `Sersic` model,
and if their names are the same it copies their prior from the centred model. Thus, it will locate the
parameters in common between the two profiles (`centre`, `ell_comps`, `effective_radius`) and pass those,
leaving the `sersic_index`'s priors as the default values.

The `take_attributes` method is used in many examples of prior passing, when we pass a simpler parameterization of a
model to a more complex model. Another good example would be passing the result of a `IsothermalSph` to an
`Isothermal`.
"""
source_bulge = af.Model(al.lp_linear.Sersic)
source_bulge.take_attributes(result_1.model_centred.galaxies.source.bulge)

"""
We now compose the model with these components that have had their priors customized.
"""
lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

"""
The `info` attribute shows the model, including how all priors are updated via prior passing.
"""
print(model_2.info)

"""
Compare the priors in `model_2.info` with those in `model_1.info`. Transferred free parameters now have truncated
Gaussian priors centred on the first result. Newly freed parameters retain the priors we assigned separately.
The widths come from configuration; they need not all be narrower than the original priors.

The two changes we made deliberately, and separately from the passing itself, are structural: `centre` and
`sersic_index` are free again, because we chose not to pass those, and the source is now a `Sersic` rather than
an `ExponentialCore` — the upgrade `take_attributes` exists to make possible.
"""
af.ModelPlotter(model_2).figure()

"""
__Search__

We now fit the second model, using a separate output name so we can inspect both searches and their priors.
"""
search_2 = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_optional"),
    name="tutorial_prior_passing_2",
    unique_tag=dataset_name,
    n_live=100,
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the output/howtolens/chapter_optional"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

print("Search has finished run - you may now continue the notebook.")

"""
__Result__

We can again inspect the results via the `info` attribute.
"""
print(result_2.info)

"""
Inspect the fitted image and residuals to assess how well the second model explains the data.
"""
aplt.subplot_fit_imaging(fit=result_2.max_log_likelihood_fit)

"""
__Wrap Up__

We have fitted a simpler model, used `result_1.model_centred` to transfer priors for selected components, and fitted
a second model with additional freedom. Entire profiles or galaxies can also be transferred when appropriate:

```python
bulge = result_1.model_centred.galaxies.lens.bulge
lens = result_1.model_centred.galaxies.lens
source = result_1.model_centred.galaxies.source
```

Passing `result_1.instance` instead supplies fixed fitted values. Tutorial 6 uses this to fix lens-light geometry
while continuing to solve linear intensities. The later uncertainties are conditional on the quantities held fixed.

This optional technique remains available, but it is not required for the search-chaining workflow in the main
chapter. Inspect the first fit carefully: centring priors on a local maximum can make other solutions harder to find.
Using constraints inferred from the same data as new priors also requires care when interpreting later uncertainties.

__Detailed Explanation Of Prior Passing__

For each transferred free parameter, `model_centred` creates a `TruncatedGaussianPrior`. Its mean is the posterior
median when available, falling back to the maximum-likelihood value for results without a posterior median.
Parameter limits are taken from configuration, falling back to the original prior's limits.

The width is determined by the parameter's configured width modifier. In this implementation it is not taken
from the posterior error bars. The new priors therefore concentrate exploration near inferred values without
reproducing the complete posterior distribution or its correlations.

An absolute width modifier specifies sigma in the parameter's units. A relative width modifier scales with the
inferred value, with any configured floor providing a minimum width. A parameter near zero can otherwise produce
a vanishingly small relative width, so centres and other parameters that can cross zero require particular care.
The linear intensities used here are solved inside the likelihood and do not receive non-linear priors.

Explicit alternatives are `result_1.model_centred_absolute(a=...)` and `result_1.model_centred_relative(r=...)`.
The absolute variant sets sigma to `a`; the relative variant sets sigma to `r` times the inferred value. Choose
positive, meaningful widths for the components being transferred: a single relative factor is unsuitable for
parameters that can be zero or negative, and a single absolute width need not suit parameters with different units.

These centred priors need not match a strongly asymmetric or multimodal posterior. Even a broad width does not
guarantee that the second search can recover solutions missed by the first.

__EXAMPLE__

Suppose a first search gives a posterior median Sersic index of 4.0. With an absolute width modifier of 0.8,
`model_centred` constructs a truncated Gaussian prior with mean 4.0 and sigma 0.8, subject to the parameter limits.
With a relative width modifier of 0.8 and no active floor, sigma would instead be 4.0 * 0.8 = 3.2.

These values illustrate the distinction; they are not recommended settings or a statement of the current defaults.
The centre of a new prior comes from the result, while its width expresses how much freedom we give the next search.

**🤖 Ask AI:** In this two-search lens fit, some parameters receive priors from `model_centred`, others receive fresh
priors, and some remain fixed. How could each choice affect the solutions found and the uncertainties reported?

Return to Tutorial 6 for the main chapter's search-chaining approach, or continue to chapter 3 for pixelized source
reconstructions.
"""
