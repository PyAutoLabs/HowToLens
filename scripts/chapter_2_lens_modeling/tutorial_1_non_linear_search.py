r"""
Tutorial 1: Non-linear Search
=============================

In Chapter 1, we predicted what a strong lens would look like for a given set of model parameters and compared that
prediction with an observed image.

By changing the parameters yourself, you could improve the fit and increase its log likelihood. This demonstrated how
the data constrain a lens model, but manually adjusting parameters becomes difficult as the model grows.

In this chapter, we use a non-linear search to explore possible parameter values automatically. We will learn how to
find models that explain the observations, assess their uncertainties and recognise when a fit needs further attention.

We begin with simulated imaging, where we know the model used to generate the data. This lets us examine the fitting
process before tackling the additional challenges of real observations.

__Contents__

- **Overview:** Introduce the experiment and the results to inspect.
- **Parameter Space:** Understand how free parameters define the models a search can explore.
- **More Dimensions:** Understand how model parameters work together.
- **Automated Exploration:** Use a non-linear search to explore possible model parameters.
- **Deeper Background:** Find further explanations of likelihoods and Bayesian inference.
- **PyAutoFit:** Introduce the tools used to define models, priors and searches.
- **Initial Setup:** Introduce the simulated imaging dataset.
- **Dataset Auto-Simulation:** Create missing simulated data and load the imaging files.
- **Mask:** Select the image pixels used in the fit.
- **Model:** Define the components and free parameters of the model.
- **Priors:** Describe the allowed values of each free parameter.
- **Analysis:** Connect model predictions to the data and evaluate their likelihood.
- **Non-linear Search:** Introduce the search used to explore the model parameters.
- **Search Settings:** Choose live-point, batching and output settings.
- **Model Fit:** Run the search using the model and analysis.
- **Result:** Inspect the inferred parameters and fitted image.
- **Inspecting the Fit:** Compare the predicted image with the data and residuals.
- **Other Searches:** Find the optional optimisation and MCMC examples.
- **Wrap Up:** Review the lesson and its connection to the next topic.


__Overview__

This tutorial introduces the main pieces of an automated model fit:

- The **dataset** contains the measurements and their uncertainties.
- The **model** describes the physical components and identifies which parameters are free to vary.
- The **Analysis** object connects the model to the dataset: it predicts the observables for a particular set of parameter values and evaluates their log likelihood.
- The **search** explores parameter space using this likelihood and the specified priors.

We will explain parameter spaces, likelihood surfaces and priors, then use Nautilus to fit the model and inspect its
inferred parameters and uncertainties.

__Parameter Space__

Imagine that we fix every property of our lens model except its Einstein radius, $\theta_{\rm E}$. We can calculate a predicted image for each value of this parameter, compare that image with our observations, and calculate its log likelihood:

$$
\ln \mathcal{L} = f(\theta_{\rm E}).
$$

This is a one-dimensional parameter space. Each position along the horizontal axis represents a different Einstein radius, and the height of the curve tells us how well that model fits the data.

Changing the Einstein radius changes the deflection of the source light and therefore the positions and appearance of its lensed images. Values that place the predicted arcs close to the observed arcs can give a higher likelihood; values that put light where the data contain none give a poorer fit.

Now allow the source's effective radius, $R_{\rm e}$, to vary too:

$$
\ln \mathcal{L} = f(\theta_{\rm E}, R_{\rm e}).
$$

Every pair of values specifies a different lens model. We can imagine these pairs laid out across a sheet, with Einstein radius along one axis and source size along the other. The log likelihood gives a height above each point, forming a surface.

Changing the source size changes how extended its predicted light is. After lensing, this affects the widths and brightness distribution of the arcs. The best-fitting source size can depend on the mass model: changing one parameter can sometimes be partly compensated by changing another. On our likelihood surface, this can produce an elongated or curved ridge of relatively good fits.

__More Dimensions__

Our actual model has more than two free parameters. The lens has a centre, and the source has its own centre, size and intensity. We can write the same idea more generally as:

$$
\ln \mathcal{L} = f(p_1, p_2, \ldots, p_N),
$$

where each $p_i$ is one of the model's free parameters.

Each point in this $N$-dimensional space specifies all the values needed to construct a model and predict an image. A non-linear search explores these points to find models that explain the observations and investigate the range of parameter values consistent with them.

Although we cannot draw the full surface when there are many parameters, its structure still matters. There can be broad regions of poor fits, narrow ridges of good fits, and separate peaks corresponding to different explanations of the image.

For example, moving the source changes the arrangement and relative brightness of its lensed images. Changing the lens mass can also change those images. Several combinations may reproduce some features reasonably well, while only a smaller region reproduces all of them together.

This is why automated modeling involves more than adjusting each parameter independently until the image looks right. The search must explore how the parameters work together.

**🤖 Ask AI:** In strong-lens modeling, why might changing the lens mass and the source size together produce similar images, and how could this appear on a likelihood surface?

__Automated Exploration__

A non-linear search automates the exploration you performed manually in Chapter 1.

The search proposes parameter values. The Analysis object uses the corresponding model to predict the data and returns
a log likelihood. The search then uses its algorithm to decide which parameter values to evaluate next.

Automating this process allows us to evaluate many more models than we could inspect by hand. It does not guarantee
that every search finds the best solution: its starting conditions, settings and the structure of parameter space all
matter. We will see examples of this later in the chapter.

__Deeper Background__

The explanations here introduce what the search does and how to interpret its results. You do not need to understand
every detail of an algorithm to start using it, but you should understand what it estimates and how it can fail.

For now, work through the example and inspect its fit. After completing HowToLens, the optional searches tutorial and
the background resources linked below provide further detail on the methods.

- [Nested Sampling](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/astrophysics/public/icic/data-analysis-workshop/2016/NestedSampling_JRP.pdf)
- [Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
- [Introduction to MCMC Sampling](https://twiecki.io/blog/2015/11/10/mcmc-sampling/)
- [A Zero-Math Introduction to MCMC Methods](https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50)
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__PyAutoFit__

PyAutoLens uses [PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit) to define free parameters and priors, run non-linear searches and organise their results.

We import PyAutoFit as `af`. Its modeling and search tools work alongside the lensing components and Analysis objects
provided by PyAutoLens, which we import as `al`.

The same PyAutoFit tools can be used for other scientific problems by supplying a suitable model and Analysis object.
"""

import autofit as af

"""
__Initial Setup__

We begin with a simulated image containing a lensed source and no foreground lens light. The simulation uses an
`IsothermalSph` mass profile for the lens and an `ExponentialCoreSph` light profile for the source.

We will fit the same profile families, allowing their parameters to vary. This gives us a controlled example: the model
can represent the system that generated the data, so we can focus on how the search explores its parameters.

Noise still affects the inferred values. Even a successful search need not recover the simulation’s input parameters
exactly.

The following code selects the dataset folder. The next cell generates the data if needed and loads the image, noise
map and point-spread function (PSF).
"""

dataset_name = "simple__no_lens_light__mass_sis"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset is missing, this cell runs the simulator to create it. It then loads the files into an `Imaging` object
and plots the dataset.

Inspect the lensed emission in the image and its strength relative to the noise map. The PSF describes the telescope’s
blurring, which must be included when predicting the observed image.

Loading the dataset does not fit a model. It supplies the measurements and observational information that the Analysis
object will use later.
"""

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/no_lens_light__mass_sis.py"
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

We apply a circular mask with a radius of 3.0 arcseconds to select the region used for the fit.

Check the plotted mask against the image: the selected region should contain the lensed emission whose structure we
want to explain.

Masking changes which measurements enter the fit. It does not change the lens model or fix any of its parameters. Later
in this chapter, *Data and Likelihoods* explores mask choices and the treatment of contaminating objects.
"""

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

In Chapter 1, we created profile instances by supplying specific parameter values. To fit those parameters, we instead
wrap the profile class in `af.Model`.

For example, `af.Model(al.mp.IsothermalSph)` describes a spherical isothermal mass profile whose centre and Einstein
radius are free to vary. During the search, specific values are supplied to create an instance of this profile.

We place the mass profile inside a lens galaxy at a fixed redshift of 0.5. We do not include lens light because this
simulated image contains only the lensed source.
"""

mass = af.Model(al.mp.IsothermalSph)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

"""
We describe the source with an `ExponentialCoreSph` light profile at a fixed redshift of 1.0.

Its free parameters include its centre, effective radius and intensity. The search varies these quantities together with the lens mass parameters. Changing the source intensity changes the brightness of the predicted lensed emission, allowing the model to match the observed flux as well as the shape and location of the arcs.
"""

bulge = af.Model(al.lp.ExponentialCoreSph)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

"""
We combine the lens and source into an `af.Collection`, giving each galaxy a descriptive name.

The resulting structure lets us refer to individual components through paths such as `model.galaxies.lens.mass` and
`model.galaxies.source.bulge`. We will use these paths shortly to customise their priors.

The outer collection contains a `galaxies` collection. Keeping galaxies together leaves room for other kinds of
components, such as external shear fields, alongside them in later models.
"""

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `model.info` attribute summarises the model before we fit it.

Look for the lens mass and source light components and the priors assigned to their free parameters. The source intensity is one of the quantities the search will fit; the galaxy redshifts are fixed.

Compare these same components in the model diagram. Together, the summary and diagram describe what we intend to fit. They do not contain inferred parameter values yet, because the search has not run.
"""

print(model.info)

"""
The model diagram shows the same structure visually. Follow the branches from `galaxies` to the lens and source, then
to their mass and light profiles.

Compare the free profile parameters, including the source intensity, with the fixed redshifts. These distinctions determine which quantities the non-linear search must explore.

The lens redshift of 0.5 places it in front of the source at redshift 1.0. PyAutoLens uses these redshifts to establish
the ordering and distances needed for ray tracing.

Like `model.info`, this diagram describes the model’s structure; it does not show whether the model fits the data.
"""

af.ModelPlotter(model).figure()

"""
__Priors__

Each free parameter in `model.info` has a **prior**, which describes the probability assigned to its possible values
before using this dataset’s likelihood.

A `UniformPrior` assigns equal probability density within specified limits and excludes values outside them. A
`GaussianPrior` places more probability near its mean, with its width controlled by `sigma`.

PyAutoLens supplies default priors, but we should check whether they are appropriate for our dataset. For example, a
prior centred on zero is suitable for a galaxy’s position only if the image coordinates place it near the origin.

Below, we assign Gaussian priors to the lens and source centres and uniform priors to the Einstein radius and source
effective radius. Positions and radii are expressed in arcseconds.

These choices are part of our inference assumptions. A narrow prior can exclude plausible solutions, while a broad
prior gives the search more parameter space to explore. Priors should therefore reflect justified information about the
system, rather than simply being adjusted until a preferred answer appears.

Combining the priors with the likelihood gives the **posterior distribution**: the probability distribution of the
parameters given the data and our modeling assumptions.
"""

model.galaxies.lens.mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.lens.mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
model.galaxies.lens.mass.einstein_radius = af.UniformPrior(
    lower_limit=0.0, upper_limit=8.0
)

model.galaxies.source.bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
model.galaxies.source.bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
model.galaxies.source.bulge.effective_radius = af.UniformPrior(
    lower_limit=0.0, upper_limit=1.0
)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
Print `model.info` again and find the priors we have changed.

The lens-centre coordinates now have Gaussian priors with a mean of 0.0 and a standard deviation of 0.1 arcseconds. The
source-centre priors are broader, with a standard deviation of 0.3 arcseconds.

The Einstein radius has a uniform prior between 0.0 and 8.0 arcseconds, and the source effective radius has a uniform
prior between 0.0 and 1.0 arcseconds. These are inputs to the fit, not uncertainties measured from the data.
"""

print(model.info)

"""
The model diagram still contains the same galaxies, profiles and free parameters.

Changing the priors has not changed how those profiles predict an image. It has changed the probability assigned to
their parameter values, which can affect both the search and the inferred posterior.

This is different from changing the model’s physical components—for example, replacing a spherical mass profile with an
elliptical one.
"""

af.ModelPlotter(model).figure()

"""
__Analysis__

We have defined the dataset and the model separately. The `AnalysisImaging` object provides the interface between them.

When the search supplies a model instance with specific parameter values, the analysis constructs its predicted image,
including ray tracing and PSF convolution. It compares that prediction with the masked imaging data, using the noise
map to evaluate the log likelihood.


You performed the same essential comparison in Chapter 1 when you created a `FitImaging` object and inspected its
`log_likelihood`. The Analysis object makes that calculation available to the search so it can be repeated for many
model instances.

Creating the Analysis object does not start a search. It defines how proposed models will be evaluated.

A detailed step-by-step visual guide is provided at `autolens_workspace/scripts/imaging/likelihood_function.py`.
"""

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Non-linear Search__

We are ready to fit the lens model using **Nautilus**, a non-linear search.

Nautilus explores combinations of parameter values within the priors. For each combination, the Analysis object
predicts the image and calculates its log likelihood. This tells the search how well that model explains the data.

As the search progresses, it concentrates its exploration on regions containing higher-likelihood models. It explores a
range of possible solutions, allowing us to find a best-fitting model and estimate how uncertain its parameters are.

Nautilus uses a method called **nested sampling**. You do not need to understand its statistical details to work
through HowToLens. Here, we will focus on configuring a fit and interpreting what it returns.

__Search Settings__

One of Nautilus’s main settings is `n_live`, the number of “live points”. Each live point represents a set of model
parameters being used to explore parameter space.

Using more live points generally allows a more thorough exploration, but requires more computation. Too few can leave
important regions unexplored, particularly for complex models.

We use 100 live points for this introductory example. This is a starting setting, not a guarantee that every fit will
succeed. The following tutorials explain how to recognise difficulties and adjust the search.

The `n_batch` setting controls how many proposed models are evaluated together. It affects computational speed and
memory use, rather than setting the number of live points.
"""

search = af.Nautilus(
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=2500,
)

"""
__Model Fit__

We start the fit by passing our model and analysis to the search.

The model supplies the components, free parameters and priors. The analysis defines how each proposed model is compared
with the data. Nautilus controls which parameter values are explored.

When using JAX, the numerical computing library used by PyAutoLens, the calculations are first compiled for repeated
use. You may therefore see an initial pause before the search begins reporting progress.

Let the search finish before continuing. Its completion means that it has met its stopping criteria; we must still
inspect the result to assess whether the fit is satisfactory.

This is the one search in this chapter that you run in full, so you can see a fit in progress. From *Practicalities*
onwards, HowToLens loads stored results for each fit, so later tutorials do not keep you waiting.
"""

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a `Result` object. Print its `info` attribute to see the inferred parameter values and their
estimated uncertainties.

Compare this with the earlier `model.info`: that described what we allowed to vary, whereas `result.info` summarises
what we have inferred from the data.

An uncertainty describes the range of parameter values supported by the fit. For example, the data might constrain the
Einstein radius closely while allowing a wider range of source sizes.

These estimates depend on our model, priors and noise assumptions, as well as the search having explored the relevant
solutions. We should inspect the fit before trusting the reported values.
"""

print(result.info)

"""
__Inspecting the Fit__

Plot the highest-likelihood model found by the search alongside the data and residuals.

Check whether it reproduces the position, shape and brightness of the lensed emission. Then examine the normalised
residuals for coherent patterns, such as an arc that the model has failed to reproduce.

Our simulated data were generated using the same profile families we fitted. A successful fit should therefore leave
residuals broadly consistent with the noise. We do not expect every pixel to have a residual of zero.

Good residuals and sensible parameter estimates provide useful checks, but neither guarantees that every relevant
solution has been explored. The next tutorials develop the practical skills needed to assess this.

**🤖 Ask AI:** My fitted lens model leaves an arc-shaped pattern in the normalised residuals. What could cause this, and
what should I check before trusting the inferred parameters?
"""

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Other Searches__

Nautilus provides the model fit and parameter uncertainties we need for this introductory example. We will continue
using it throughout much of HowToLens.

Other searches can also be useful. Later in this chapter, we will use a fast optimiser for an initial fit before
running a more thorough Nautilus search.

You can explore these alternatives in the optional searches tutorial after completing HowToLens. For now, focus on
building appropriate models and assessing how well they explain the data.

__Wrap Up__

You have now fitted a lens model automatically, using the same image prediction and likelihood calculation that you
explored manually in Chapter 1.

The fit brings together four pieces:

- **Data:** the measurements, uncertainties and observational information.
- **Model:** the physical components, their free parameters and their priors.
- **Analysis:** the connection between model and data, predicting the observations and evaluating their likelihood.
- **Search:** the method that explores parameter values and returns the inferred results.

Keeping these pieces separate lets us change the model, dataset or search while retaining the same overall approach.
Later in the chapter, we will see how this applies to the other types of lensing data introduced in Chapter 1.

In the next tutorial, *Practicalities*, we include the foreground lens galaxy’s light and fit a more complex model. We
will also learn how to follow a search’s progress, navigate its output folder and inspect its saved results.
"""
