"""
Tutorial 7: Summary
===================

You have now fitted a lens model by searching through its possible parameter values, rather than choosing those
values by hand.

We began by connecting a point in parameter space to a predicted image and its likelihood. We then explored why
finding a convincing fit can be difficult: parameters can compensate for one another, different solutions can
produce similar images, and a search can settle at a local maximum. Inspecting the fit, changing search settings
and trying other search methods help us investigate these possibilities.

Linear light profiles simplified part of this problem by solving for intensities within each likelihood evaluation.
Search chaining then showed how an initial fit can support a more detailed analysis, while making clear which
quantities remain free and which assumptions are carried forward.

We also separated the ingredients of a modelling calculation. The dataset supplies the observations; the model
describes the objects and parameters; the analysis defines how their predictions are compared with those
observations; and the search explores the resulting parameter space. The same `af.Model` and `af.Collection`
composition API is used across datasets. Different observations require different predictions and likelihoods,
but much of the modelling framework remains familiar.

You can now adapt the examples in `autolens_workspace` to your own data, or use the simulators to investigate a
lens whose input properties you know. Start by inspecting the image and the fitted residuals, and treat the first
plausible solution as something to investigate rather than proof that the search has found the global maximum.

Chapter 3 develops a more flexible description of the background source: reconstructing its light on a pixelization
instead of representing it with analytic light profiles.
"""
