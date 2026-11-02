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

The automatic narrowing described below is historical behaviour. In the current implementation, `result.model`
preserves the input priors rather than replacing them with constraints inferred from the posterior. The code
excerpts below are therefore for reading older analyses, not executable exercises for the current version.

__Contents__

- **Passing a Model:** How earlier workflows carried parameter constraints between searches.
- **Changing the Model:** Which choices were separate from prior passing.
- **Prior Widths:** How the historical Gaussian approximation was configured.
- **Fixed Instances:** The distinction between passing priors and fixing fitted values.
- **Wrap Up:** Interpreting older analyses and returning to the current workflow.

__Passing a Model__

In the historical workflow, an initial search fitted a simpler lens model. Its posterior constraints were then used
to construct Gaussian priors for a second search. The aim was to concentrate exploration near solutions that the
first search had already identified.

For example, older scripts assigned a mass parameter using:

```python
mass.einstein_radius = result_1.model.galaxies.lens.mass.einstein_radius
```

Under the old behaviour, this transferred an inferred constraint on the Einstein radius. In the current version,
the same expression transfers the input prior; it does not automatically centre a new Gaussian on the fitted value.
This distinction is essential when interpreting an old script that still runs without raising an error.

Entire components could also be transferred:

```python
lens = result_1.model.galaxies.lens
source = result_1.model.galaxies.source
```

The model composition API was the same `af.Model` and `af.Collection` API used throughout this chapter. Prior
passing changed the distributions assigned to free parameters; it did not itself select new physical components.

__Changing the Model__

An older chain might fit a lens Sersic profile with its centre and Sersic index fixed in the first search, then
free those quantities in the second search. It could retain constraints on the profile's ellipticity and effective
radius while assigning fresh priors to the newly freed parameters.

For example, the historical construction included:

```python
bulge = af.Model(al.lp_linear.Sersic)
bulge.ell_comps = result_1.model.galaxies.lens.bulge.ell_comps
bulge.effective_radius = result_1.model.galaxies.lens.bulge.effective_radius
bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=4.0, sigma=2.0, lower_limit=0.0, upper_limit=5.0
)
```

Creating a new profile supplied fresh centre priors. The Sersic-index prior was chosen explicitly. Neither change
was inferred automatically by prior passing. Copying a component with fixed parameters would keep those values fixed.

Another example replaced an ExponentialCore source with a Sersic source. The `take_attributes` method transferred
matching attributes, such as the centre, ellipticity and effective radius, while the new Sersic index retained its
default prior:

```python
source_bulge = af.Model(al.lp_linear.Sersic)
source_bulge.take_attributes(result_1.model.galaxies.source.bulge)
```

Transferring matching attributes and deriving new priors from posterior constraints are separate operations.
The historical interpretation depended on what `result_1.model` returned at the time.

__Prior Widths__

The historical prescription approximated each parameter's marginal posterior with a Gaussian prior centred on its
posterior median. Width settings controlled how tightly the next search was concentrated around that value.
This was an approximation: asymmetric or multimodal posteriors are not fully represented by one Gaussian, and
separate one-dimensional priors do not preserve correlations between parameters.

Older configurations used `width_modifier` entries and `prior_passer` settings. Absolute widths supplied a scale
in the parameter's units; relative widths supplied a fraction of the inferred value. Depending on the configuration,
posterior errors could also contribute to the adopted width. Read the configuration saved with an old result when
reconstructing how its next-stage priors were made.

For an illustrative Sersic-index median of 4.0, an absolute width of 0.8 would give a Gaussian with mean 4.0 and
sigma 0.8. A relative width of 0.8 would instead give sigma 3.2. These numbers illustrate the distinction; they
are not recommended settings or a statement of the current defaults.

Relative widths are unsuitable for some parameters. A centre coordinate near zero, for example, could be given an
arbitrarily small width even when its uncertainty is substantial. An absolute angular scale is easier to interpret
in that case. Conversely, a parameter whose numerical scale depends strongly on units may motivate a relative width.
The linear intensities in our current examples are solved within the likelihood and do not need such non-linear priors.

A broad width gave the next search more freedom to move, whereas a narrow width concentrated it near the earlier
solution. Neither choice guaranteed that other important solutions were retained. If the first search missed a
peak, narrowing priors around its result could make that peak even harder to discover.

__Fixed Instances__

Passing a fitted instance is different from passing priors:

```python
lens_bulge_fixed = result_1.instance.galaxies.lens.bulge
```

An instance supplies fitted values instead of free non-linear parameters. Tutorial 6 uses this distinction to fix
lens-light geometry while continuing to solve the linear intensities. The later uncertainties are conditional on
that fixed geometry; they do not include uncertainty in the quantities held fixed.

This remains useful in current search chains, but it should not be described as automatically narrowing priors.

__Wrap Up__

When reading an older chained analysis, establish which parameters stayed free, which values were fixed, and how
any new priors were constructed. Check the software version and saved configuration rather than inferring behaviour
from the presence of `result.model` alone.

Using posterior constraints from an earlier fit to the same data as new priors can also affect the interpretation
of later uncertainties. A computational shortcut does not make those constraints independent new information.

**🤖 Ask AI:** An older lens-modelling script uses `result_1.model` in a second fit to the same image. What would I
need to inspect to establish whether it reused the original priors, narrowed them using the first posterior, or
fixed parameters, and how would those choices affect the reported uncertainties?

Return to Tutorial 6 for the current search-chaining example, or continue to chapter 3 for pixelized source
reconstructions. Historical prior passing is not a prerequisite for either.
"""
