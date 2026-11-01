"""
Tutorial: Alternative Searches
==============================

Chapter 2 introduces lens modeling using Nautilus. This optional tutorial explores alternative search settings,
optimisation and Markov chain Monte Carlo (MCMC) after you have completed HowToLens.

The dataset, model and Analysis object retain the same roles. Changing the search changes how parameter space is
explored and which information is returned. Inspect the fits and assess the exploration before trusting the results.

__Contents__

- **Nested Sampling:** Lets first perform the model-fit using Nautilus, but look at different parameters that control how.
- **Optimisation:** Improve parameter values from multiple starting points using Prodigy.
- **MCMC:** For users familiar with Markov Chain Monte Carlo (MCMC) non-linear samplers, PyAutoFit supports the.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
we'll use new strong lensing data, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens system uses an `Isothermal` galaxy mass profile and a separate `ExternalShear` field.
 - The source galaxy's light is an `Sersic`.
"""
dataset_name = "lens_sersic"
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
        [sys.executable, "scripts/simulator/lens_sersic.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
we'll create and use a smaller 2.0" `Mask2D` again.
"""
mask_radius = 2.6

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Nested Sampling__

Lets first perform the model-fit using Nautilus, but look at different parameters that control how long it takes to run. 
We'll therefore discuss in a bit more detail how Nautilus works, but still keep this description conceptually simple  
and avoid technical terms and jargon. For a complete description of Nautilus you should check out the Nautilus 
publication `https://arxiv.org/abs/2306.16923`.

nlive:

Nautilus is a `nested sampling` algorithm. As we described in chapter 2, it throws down a set of `live points` in 
parameter space, where each live point corresponds to a lens model with a given set of parameters. These points are
initially distributed according to our priors, hence why tuning our priors allows us to sample parameter space faster.
 
The number of live points is set by the parameter `n_live`. More points provide a more thorough sampling of 
parameter space, increasing the probability that we locate the global maxima solution. Therefore, if you think your 
model-fit has gone to a local maxima, you should try increasing `n_live`. The downside of this is Nautilus will 
take longer to sample parameter space and converge on a solution. Ideally, we will use as few live points as possible 
to locate the global maxima as quickly as possible.

f_live:

A nested sampling algorithm estimates the *Bayesian Evidence* of the model-fit, which is quantity the non-linear 
search algorithms we introduce later do not. The Bayesian evidence quantifies how well the lens model as a whole fits
the data, following a principle called Occam's Razor (`https://simple.wikipedia.org/wiki/Occam%27s_razor`). This 
penalizes models for being more complex (e.g. more parameters) and requires that their additional complexity improve 
their overall fit to the data compared to a simpler model. By computing the comparing the Bayesian evidence of 
different models one can objectively choose the lens model that best fits the data.

A nested sampling algorithm stops sampling when it estimates that continuing sampling will not increase the Bayesian 
evidence (called the `log_evidence`) by more than the `f_live`. As Nautilus progresses and converges on the
solution, the rate of increase of the estimated Bayesian evidence slows down. Therefore, higher `f_live`s 
mean Nautilus terminate sooner.
    
A high `f_live` will make the errors estimated on every parameter unreliable and its value must be kept 
below 0.8 for reliable error estimates. However, when chaining searches, we typically *do not care* about the errors 
in the first search, therefore setting a high evidence tolerance can be an effective means to make Nautilus converge
faster (we'll estimate reliable errors in the second search when the `f_live is 0.8 or less). 

Lets perform two fits, where:

 - One has many live points and a higher evidence tolerance, causing the non-linear search to
 take a longer time to run.
      
 - One has few live points, a high sampling efficiency and evidence tolerance, causing the non-linear search to
 converge and end quicker.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy, redshift=0.5, bulge=al.lp.Sersic, mass=al.mp.Isothermal
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_optional"),
    name="tutorial_searches_slow",
    unique_tag=dataset_name,
    n_live=400,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the output/howtolens/chapter_optional/tutorial_searches_slow"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_slow = search.fit(model=model, analysis=analysis)

"""
Lets check that we get a good model and fit to the data.
"""
aplt.subplot_fit_imaging(fit=result_slow.max_log_likelihood_fit)

"""
We can use the result to tell us how many iterations Nautilus took to convergence on the solution.
"""
print("Total Nautilus Iterations (If you skip running the search, this is ~ 500000):")
print(result_slow.samples.total_samples)

"""
Now lets run the search with fast settings, so we can compare the total number of iterations required.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_optional",
    name="tutorial_searches_fast",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

print(
    "The non-linear search has begun running - checkout the output/howtolens/chapter_optional/tutorial_searches_fast"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_fast = search.fit(model=model, analysis=analysis)

print("Search has finished run - you may now continue the notebook.")

"""
Lets check that this search, despite its faster sampling settings, still gives us the global maxima solution.
"""
aplt.subplot_fit_imaging(fit=result_fast.max_log_likelihood_fit)

"""
And now lets confirm it uses significantly fewer iterations.
"""
print("Total Nautilus Iterations:")
print("Slow settings: ~500000")
print(result_slow.samples.total_samples)
print("Fast settings: ", result_fast.samples.total_samples)

"""
__Optimisation__

An optimiser repeatedly tries to improve a model. Starting from one set of parameter values, it works out which
changes are likely to produce a better fit, updates the parameters and evaluates the new model. Repeating this process
moves it towards a peak: a region where nearby parameter values no longer improve the fit.

You can think of this as making increasingly informed guesses, aiming for higher-likelihood solutions. Prodigy also
takes the priors into account, so it balances how well the model fits with the probability assigned to its parameter
values. This is maximum a posteriori (MAP) optimisation.

A peak need not be the best solution overall. Prodigy therefore repeats this process from several starting points,
giving it more opportunities to find a good model.

Prodigy uses gradients to guide its updates: these describe how changing each parameter affects the posterior
probability. PyAutoLens calculates these gradients automatically using a numerical computing library called **JAX**.
You do not need to use JAX directly to run this example.

We use 48 starting points, with up to 300 optimisation steps for each. The search can stop earlier when its stopping
criteria are met. Increasing either setting allows more exploration but also requires more computation. The batch size
of 4 limits how many starts are processed together; it does not reduce the total number of starts.
"""
search = af.MultiStartProdigy(
    path_prefix=Path("howtolens") / "chapter_optional",
    name="tutorial_searches_prodigy",
    unique_tag=dataset_name,
    n_starts=48,
    n_steps=300,
    batch_size=4,
)

result_prodigy = search.fit(model=model, analysis=analysis)

print(result_prodigy.info)

aplt.subplot_fit_imaging(fit=result_prodigy.max_log_likelihood_fit)

"""
Prodigy finds a best-fitting set of parameter values, but it does not tell us how uncertain those values are. An
Einstein radius of 1.6 arcseconds, for example, does not tell us whether values of 1.5 or 1.7 arcseconds would also be
consistent with the data.

Look for whether the model reproduces the location, shape and brightness of the lensed emission. In the normalised
residuals, inspect whether coherent structures remain that the model has failed to explain.

A plausible parameter value does not establish that the fit is good. Likewise, small residuals do not prove that the
search has found the best possible solution or measured parameter uncertainty.

**🤖 Ask AI:** My optimised lens model leaves an arc-shaped pattern in the normalised residuals. What should I
investigate before trusting its inferred parameters?

__MCMC__

For users familiar with Markov Chain Monte Carlo (MCMC) non-linear samplers, PyAutoFit supports the non-linear
searches `Emcee` and `Zeus`. These also need a good starting point, and are generally less effective at
lens modeling than Nautilus. 

I've included an example runs of Emcee and Zeus below, where the model is set up using `UniformPriors` to give
the starting point of the MCMC walkers. 
"""
lens_bulge = af.Model(al.lp.Sersic)
lens_bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
lens_bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
lens_bulge.intensity = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
lens_bulge.effective_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.6)
lens_bulge.sersic_index = af.UniformPrior(lower_limit=3.0, upper_limit=5.0)


mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)

field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)
field.shear.gamma_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
field.shear.gamma_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge = af.Model(al.lp.Sersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.intensity = af.UniformPrior(lower_limit=0.1, upper_limit=0.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

search = af.Zeus(
    path_prefix=Path("howtolens") / "chapter_optional",
    name="tutorial_searches_zeus",
    unique_tag=dataset_name,
    nwalkers=50,
    nsteps=1000,
)

print(
    "Zeus has begun running - checkout the output/howtolens/chapter_optional/tutorial_searches_zeus"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_zeus = search.fit(model=model, analysis=analysis)

print("Zeus has finished run - you may now continue the notebook.")

aplt.subplot_fit_imaging(fit=result_zeus.max_log_likelihood_fit)


search = af.Emcee(
    path_prefix=Path("howtolens") / "chapter_optional",
    name="tutorial_searches_emcee",
    unique_tag=dataset_name,
    nwalkers=50,
    nsteps=1000,
)

print(
    "The non-linear search has begun running - checkout the output/howtolens/chapter_optional/tutorial_searches_emcee"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_emcee = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

aplt.subplot_fit_imaging(fit=result_emcee.max_log_likelihood_fit)

"""
__Wrap Up__

This tutorial showed how the same lens model can be fitted using different non-linear searches (e.g. `Nautilus`,
`Emcee`, `Zeus`). Each search explores parameter space in a different way, and the best choice depends on the
dimensionality and complexity of the model you are fitting.

For the vast majority of lens models in **PyAutoLens**, the default nested sampling search `Nautilus` is recommended,
as it is robust, efficient and requires little manual tuning.
"""
