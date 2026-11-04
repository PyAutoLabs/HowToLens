"""
Optional Tutorial: Need for Speed
================================

Two fits to the same image can take very different amounts of time. Finding a good fit with a simple model may be
quick, whereas exploring the uncertainties of a more flexible model can take much longer. Both the calculation
performed for each model and the search through parameter space contribute to this difference.

This optional tutorial looks at these costs, including JAX compilation, batching and hardware. It builds on the fits
we have performed in chapter 2, but is not a prerequisite for the pixelized source reconstructions of chapter 3.
Return to it when planning a larger analysis, or when a fit takes longer than you expected.

__Contents__

- **Searching Parameter Space:** What determines how much exploration a search needs.
- **Cost Per Evaluation:** The calculations required to compare one model with the data.
- **JAX:** Compilation and the cost of the first evaluation.
- **Batches and GPU Memory:** Evaluating models together and choosing a batch size.
- **Run Time Estimation:** Measuring the costs relevant to your analysis.
- **Data Quantity:** How image resolution and the fitted region affect computation.
- **Wrap Up:** Putting these considerations together.

__Searching Parameter Space__

A non-linear search evaluates models until its stopping criteria are met. The number of evaluations depends on more
than the number of free parameters. A narrow, curved region of high likelihood can be difficult to explore, as can
several separated peaks. Two models with the same number of parameters need not take similar amounts of time to fit.

Broad priors can require the search to explore large regions that fit the data poorly. However, narrowing a prior
just to make a fit faster risks excluding plausible solutions. The prior should express a justified assumption
about the parameter, rather than the region we hope will give the quickest result.

Search settings also determine how much work is performed. More live points in Nautilus, or more starting points
and optimisation steps in Prodigy, cost computation but may improve exploration. Tutorial 3 showed why reducing
this effort too far can leave us with a local maximum.

The searches also answer different questions. Prodigy seeks a good-fitting model without exploring the posterior
to measure parameter uncertainties. Nautilus performs that additional exploration, so we should expect uncertainty
estimation to take longer. The search chain in Tutorial 6 uses these different strengths at different stages.

The optional tutorial `chapter_optional/tutorial_searches.py` introduces other search methods. Trying a different
method can help with a difficult fit; the appropriate choice depends on both the parameter space and whether we
need a best-fitting model or a description of its uncertainties.

__Cost Per Evaluation__

For every model proposed by the search, we must predict the observed image and compare it with the data. This
includes calculating deflection angles, ray-tracing the image-plane grid, evaluating light profiles and convolving
the predicted image with the telescope's PSF.

Over-sampling adds calculations within image pixels, so that rapidly changing light profiles are represented
accurately. Linear light profiles also require a linear solve for their intensities. An MGE includes many such
profiles, while the pixelized source reconstructions introduced in chapter 3 require their own reconstruction.

The cost of these operations depends on the number of pixels, profiles and source elements, as well as the numerical
methods and their implementation. There is no universal ordering in which every MGE is slower than a few light
profiles, or every pixelized reconstruction is slower than an MGE.

Adding more Gaussian components to an MGE increases the work in the linear calculation. However, solving their
intensities linearly can make the non-linear parameter space easier to explore, as we saw in Tutorial 4. A model
that costs more per evaluation can therefore finish sooner if the search needs substantially fewer evaluations.
Timing one evaluation alone does not tell us how long the whole fit will take.

__JAX__

**PyAutoLens** uses JAX (https://github.com/jax-ml/jax) to compile supported likelihood calculations and to calculate
the derivatives used by gradient-based searches such as Prodigy.

The first evaluation includes tracing the calculation and compiling it into a program that can be reused. Later
evaluations with new parameter values can then use that compiled program. This is why the beginning of a fit may
feel slow even when subsequent evaluations are fast.

Reuse depends on the calculation retaining a compatible structure. Changing array shapes, data types or settings
treated as fixed during compilation can require a new compilation. We should therefore distinguish the initial
setup cost from the cost of repeated evaluations, rather than assuming compilation happens only once across an
entire sequence of fits.

The imaging analyses in this chapter use JAX by default. Setting `use_jax=False` in `al.AnalysisImaging` selects the
NumPy path, which can be useful when diagnosing a calculation or comparing implementations. This does not make a
gradient-based search independent of derivatives: such a search still needs a supported differentiable calculation.

__Batches and GPU Memory__

Instead of evaluating one model at a time, a search can evaluate a batch of models together. This can increase the
number of evaluations completed per second, particularly on a GPU, but it also requires memory for several
calculations at once.

For Nautilus, `n_batch` controls the batch of likelihood evaluations. For MultiStartProdigy, `batch_size` controls
how many optimisation starts are processed together. These settings play different roles in the searches, but
both affect the amount of work held in memory.

A larger batch may improve throughput, but eventually the improvement can diminish or the calculation can exceed
the available GPU memory. The largest possible batch is not automatically the best setting.

A GPU is also not guaranteed to outperform a CPU for every fit. The outcome depends on the model, dataset, batch
size, numerical precision and hardware. Compilation and moving data to the device contribute to elapsed time too.
Compare the same analysis on the hardware available to you, including the costs relevant to your use: a short
interactive calculation and a long modelling campaign can favour different choices.

__Run Time Estimation__

To estimate how long a fit will take, start with a representative analysis: the model, dataset, search settings and
hardware should resemble those you intend to use. A timing measured with a simpler model or a smaller image may
give a misleading impression of the work ahead.

Separate the time spent preparing and compiling the calculation from the time spent evaluating models afterwards.
JAX can dispatch calculations before they have finished, so a measurement must wait for the calculation to complete.
Otherwise, we may time only the act of submitting the work.

For a batched search, the useful quantity is how many likelihood evaluations are completed per second. Timing a
standalone `FitImaging` calculation does not necessarily measure the compiled, batched calculation used by the search.

Even with this measurement, we do not yet know how many evaluations the search will need. A short run can reveal
expensive calculations or memory problems, but it cannot reliably predict how long exploration of a difficult
parameter space will take. Completed fits to similar datasets provide a more useful starting point.

When comparing approaches, inspect their results alongside their durations. A fit that finishes quickly at a poor
local maximum has not achieved the same outcome as one that finds a convincing solution. Likewise, finding one
good model and measuring its uncertainties are different amounts of work.

__Data Quantity__

Consider two images covering the same region of sky. One has pixels that are 0.1 arcseconds across, while the other
has pixels that are 0.05 arcseconds across. Halving the pixel scale doubles the number of pixels along each axis,
giving approximately four times as many pixels in the fitted region.

The finer image therefore requires calculations at more positions. The precise increase in runtime also depends
on over-sampling, the PSF, the model and how the calculation is implemented; four times as many pixels does not
necessarily mean a fit takes exactly four times as long.

The circular mask determines how much of the image enters the fit. However, as we discussed in Tutorial 5, faint
lens and source emission still contains information, including where a proposed model predicts light that the
image does not contain. Shrinking the circle until only the brightest features remain can change the inferred model.

We therefore choose a circle large enough to contain the relevant emission, including the lens galaxy's light.
If that calculation is expensive, we investigate the model, search and computational settings while retaining
the region needed for the scientific analysis.

__Wrap Up__

A lens fit takes time both because each proposed model requires a calculation and because the search must explore
many possible models. These costs interact: solving light-profile intensities linearly adds work to each evaluation
but can substantially simplify the search through parameter space.

JAX, batching and suitable hardware can accelerate the calculations. Search chaining can organise the exploration
into stages, using an initial fit to inform a more detailed analysis. Neither removes the need to inspect the
result and investigate alternative solutions.

When a fit is slow, identify where the time is being spent before changing settings. Compilation, expensive
evaluations and prolonged exploration call for different responses. Compare changes using both elapsed time and
the quality of the inference you obtain.

**🤖 Ask AI:** Two searches fit the same lens image. One quickly finds a good-looking model, while the other takes
much longer and returns parameter uncertainties. What should I compare before deciding whether the faster search
is sufficient for my scientific question?

"""

# from autolens import setup_notebook; setup_notebook()
