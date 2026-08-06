"""
Tutorial 11: SLaM
=================

In the previous two tutorials, we learnt how search chaining breaks a lens model-fit into a sequence of simpler
non-linear searches, and how prior passing carries the results of each search into the next. Together, they give us
the flexibility to juggle the dimensionality, priors and settings of each search — the three drivers of run-time we
met in tutorial 8 — whilst still fitting a complex and realistic lens model at the end.

You could now write your own chained sequences of searches, tailored to whatever lens model is of particular interest
to your scientific study. However, for most lens models there are standardized approaches one can take to fitting
them. For example, an effective approach is to first fit a model for the lens's light, then a model for its mass and
the source. It would be wasteful for all **PyAutoLens** users to have to write their own chained pipelines to perform
the same tasks.

For this reason, the `autolens_workspace` comes with standardized chained pipelines, which fit common lens models in
ways we have tested are efficient and robust. This closing tutorial does not run any code; it is a pointer to where
that material lives, so you know where to go once you want to scale your lens modeling up.

__Contents__

- **Search Chaining In The Workspace:** Where the workspace documents the chaining API introduced in tutorials 9 and 10.
- **SLaM (Source, Light and Mass):** The standardized pipelines that chain searches for you.
- **Wrap Up:** Choosing between single searches, your own chained fits and the SLaM pipelines.

__Search Chaining In The Workspace__

The chaining and prior passing API we introduced in tutorials 9 and 10 is documented in full in the workspace guide
`autolens_workspace/scripts/guides/modeling/chaining.py`. It covers the concise model composition API used in chained
fits, prior passing and the tools for customizing it, and how chaining is used in practice for automated lens
modeling.

__SLaM (Source, Light and Mass)__

The workspace's standardized pipelines are called the **SLaM** (Source, Light and Mass) pipelines. They are composed
of a specific sequence of chained searches:

 - `Source`: A pipeline that focuses on producing a robust model for the source's light, using simpler models for the
 lens's light and mass.

 - `Light`: A pipeline that fits a complex lens light model (e.g. one with many components), using the initialized
 source model to cleanly deblend the lens and source light.

 - `Mass`: A pipeline that fits a complex lens mass model, benefitting from the good models for the lens's light and
 source.

For fitting very complex lens models, for example ones which decompose the mass into its stellar and dark components,
the **SLaM** pipelines have been carefully crafted to do this in a reliable and automated way that is still efficient.
They also make fitting many different models to a single dataset efficient, as they reuse the results of earlier
searches (e.g. in the Source pipeline) to fit different models in the `Light` and `Mass` pipelines for the lens's
light and mass.

The canonical introduction is `autolens_workspace/scripts/guides/modeling/slam_start_here.py`, which walks through the
pipeline structure end-to-end. Runnable SLaM examples then live as `slam.py` scripts inside the feature folders of
each topic, for example `autolens_workspace/scripts/imaging/features/pixelization/slam.py` and
`autolens_workspace/scripts/imaging/features/multi_gaussian_expansion/slam.py`.

__Wrap Up__

Whether you should use individual searches, your own chained fits, the SLaM pipelines or write your own model-fitting
script depends on the scope of your scientific analysis. I would advise you begin by trying to adapt the scripts in
the `autolens_workspace` to fit your data, and also try using the SLaM pipelines once you are a confident
**PyAutoLens** user.

This tutorial concludes the lens modeling chapter. In the next chapter, we introduce pixelizations, which reconstruct
the source galaxy on a pixel-grid rather than with light profiles.
"""
