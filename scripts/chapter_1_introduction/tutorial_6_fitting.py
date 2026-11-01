"""
Tutorial 6: Fitting
===================

In the previous tutorial, we chose a lens and source, predicted their lensed
image and simulated a telescope observation. We will now work in the other
direction: starting with an image, we want to learn about the objects that
produced it.

A lens model is a mathematical description of the lens and source. Its
parameters specify properties such as the lens's mass distribution and the
source's position, size and brightness.

To assess a model, we calculate the image it predicts and compare that
prediction with the data. We account for the telescope's blurring and the
uncertainty in each pixel, so that differences are judged relative to the
noise.

We will first evaluate a model using the parameters that generated our
simulated data. We will then change those parameters and inspect how the
agreement changes. Finally, you will try adjusting parameters yourself.

This introduces model fitting: finding parameter values that describe the
data well. Creating a `FitImaging` object evaluates the parameters we supply;
it does not search for new values automatically.

__Contents__

- **Dataset:** Locate the imaging data saved in Tutorial 5.
- **Dataset Auto-Simulation:** Create the dataset if needed, then load it.
- **Mask:** Select the region of the image used in the comparison.
- **Masked Grid:** Inspect the coordinates of the selected pixels.
- **Fitting:** Compare a predicted image with the data and calculate fit statistics.
- **Incorrect Fit:** See how changing the source position affects the comparison.
- **Model Fitting:** Experiment with adjusting parameters yourself.
- **Likelihood Function:** Combine the calculation into a function of the model parameters.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We will use the simulated imaging dataset saved in Tutorial 5. It contains
the background-subtracted image, its noise map and the PSF.

The files are in `dataset/imaging/howtolens/`, relative to the repository root
from which this script is run.
"""
dataset_path = Path("dataset") / "imaging" / "howtolens"

"""
__Dataset Auto-Simulation__

If the dataset is missing, the following code runs Tutorial 5 to create it.
If you have already run that tutorial, we load the saved files.

The pixel scale is 0.1 arcseconds, matching the simulation.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/chapter_1_introduction/tutorial_5_data.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
Recall the three components of an imaging dataset:

- `data`: the measured brightness in each pixel.
- `noise_map`: the estimated noise standard deviation in each pixel.
- `psf`: the kernel describing how the telescope spreads light.

The data and noise map have the same units. The PSF kernel is normalised
to sum to one.

Below, we print the first value in each array and display the dataset.
The first image pixel and first noise-map pixel refer to the same sky
position; the first PSF value belongs to the separate, smaller kernel.
"""
print("Value of first pixel in imaging data:")
print(dataset.data.native[0, 0])
print("Value of first pixel in noise map:")
print(dataset.noise_map.native[0, 0])
print("Value of first pixel in PSF:")
print(dataset.psf.kernel.native[0, 0])

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

A mask selects which pixels we include when comparing the model with the data.
It can exclude unrelated objects, detector defects or regions outside the
area we want to analyse.

For this example, we keep a circular region with a radius of 3 arcseconds,
centred on the lens. This includes the lensed arcs and some surrounding sky.

We do not select individual pixels according to their measured
signal-to-noise ratio. Faint pixels still contain useful information: a model
that predicts bright emission where little is observed should disagree with
those data. The noise map determines how strongly each pixel contributes to
the comparison.

In a `Mask2D`, `True` means excluded and `False` means included.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,  # Arcseconds.
    centre=(0.0, 0.0),
)

print(mask)
print(mask[48:53, 48:53])

"""
Before applying the mask, let's display its boundary over the full image.
The boundary marks the edge of the region we will include in the comparison.

Check that the lensed arcs lie inside it and that some surrounding sky is
included. At this stage, we are only displaying the boundary; the dataset
has not yet been masked.
"""
aplt.plot_array(
    array=dataset.data,
    mask=mask.derive_grid.edge,
    title="Imaging Data With Mask Boundary",
)

"""
We now apply the mask to the dataset. Pixels outside the circular region
will be excluded when we compare the model with the data.
"""
dataset = dataset.apply_mask(mask=mask)

"""
The plotted image now shows the selected region, with the view zoomed to
the unmasked area. The plotting function obtains the mask boundary from
the array itself.
"""
aplt.plot_array(array=dataset.data, title="Masked Imaging Data")

"""
The mask is stored with the dataset, so later calculations can identify
which pixels to include.
"""
print("Mask2D:")
print(dataset.mask)

"""
The masked data retain the `native` and `slim` representations introduced
earlier:

- `native` arranges the values in the original 2D image shape. Masked pixels
  are filled with zeros.
- `slim` contains only the unmasked pixel values in a 1D array.

Those zeros in the native representation are placeholders. They do not mean
that zero light was measured outside the mask, and they do not contribute
to the comparison with the model.

Our original image has 101 x 101 pixels. The slim array is shorter because
we have selected only the circular region.
"""
print("Native image shape:")
print(dataset.data.native.shape)

print("Slim image shape:")
print(dataset.data.slim.shape)

"""
Despite its name, `pixels_in_mask` counts the unmasked pixels: those included
in the analysis. This number should match the length of the slim array.
"""
print("Number of unmasked pixels:")
print(dataset.mask.pixels_in_mask)

"""
The image and noise map share the same mask and pixel ordering. Their first
slim entries therefore describe the measurement and uncertainty at the same
sky position.
"""
print("First unmasked image value:")
print(dataset.data.slim[0])

print("First unmasked noise map value:")
print(dataset.noise_map.slim[0])

"""
In the native representation, the corner pixel is masked and therefore
contains a zero placeholder. The central pixel is unmasked and retains its
measured value.
"""
print("Masked corner pixel in the native image:")
print(dataset.data.native[0, 0])

centre = tuple(s // 2 for s in dataset.data.shape_native)

print("Unmasked central pixel in the native image:")
print(dataset.data.native[centre])

"""
__Masked Grid__

To predict the image, we need the sky coordinates corresponding to the
selected pixels. We obtain these from the mask as a grid of `(y, x)`
coordinates.

Each coordinate marks the centre of an unmasked pixel. Plotting the grid
shows the circular region that we selected.
"""
masked_grid = mask.derive_grid.unmasked

aplt.plot_grid(grid=masked_grid, title="Masked Grid2D")

"""
We can overlay these coordinates on the image to see where the selected
pixels lie. Each dot corresponds to one pixel whose measured brightness
will be compared with the model prediction.

The grid, image and noise map use matching pixel ordering, so their slim
entries refer to the same positions.
"""
aplt.plot_array(
    array=dataset.data,
    grid=masked_grid,
    title="Image Data With 2D Grid Overlaid",
)

"""
__Fitting__

Now that our data is masked, we are ready to proceed with the fitting process.

Fitting the data is done using the `Galaxy` and `Tracer objects that we introduced in previous tutorials. We will start by 
setting up a `Tracer`` object, using the same galaxy configuration that we previously used to simulate the 
imaging data. This setup will give us what is known as a 'perfect' fit, as the simulated and fitted models are identical.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Image")

"""
Next, let's plot the image of the tracer. This should look familiar, as it is the same image we saw in 
previous tutorials. The difference now is that we use the dataset's `grid`, which corresponds to the `masked_grid` 
we defined earlier. This means that the tracer image is only evaluated in the unmasked region, skipping calculations 
in masked regions.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=dataset.grid), title="Tracer Image To Be Fitted"
)

"""
Now, we proceed to fit the image by passing both the `Imaging` and `Tracer` objects to a `FitImaging` object. 
This object will compute key quantities that describe the fit’s quality:

`image`: Creates an image of the tracer using their image_2d_from() method.
`model_data`: Convolves the tracer image with the data's PSF to account for the effects of telescope optics.
`residual_map`: The difference between the model data and observed data.
`normalized_residual_map`: Residuals divided by noise values, giving units of noise.
`chi_squared_map`: Squares the normalized residuals.
`chi_squared` and `log_likelihood`: Sums the chi-squared values to compute chi_squared, and converts this into 
a log_likelihood, which measures how well the model fits the data (higher values indicate a better fit).

Let's create the fit and inspect each of these attributes:
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
The `model_data` represents the tracer's image after accounting for effects like PSF convolution. 

An important technical note is that when we mask data, we discussed above how the image of the tracer is not evaluated
outside the mask and is set to zero. This is a problem for PSF convolution, as the PSF blurs light from these regions
outside the mask but at its edge into the mask. They must be correctly evaluated to ensure the model image accurately
represents the image data.

The `FitImaging` object handles this internally, but evaluating the model image in the additional regions outside the mask
that are close enough to the mask edge to be blurred into the mask. 
"""
print("First model image pixel:")
print(fit.model_data.slim[0])
aplt.plot_array(array=fit.model_data, title="Model Image")

"""
Even before computing other fit quantities, we can normally assess if the fit is going to be good by visually comparing
the `data` and `model_data` and assessing if they look similar.

In this example, the tracer used to fit the data are the same as the tracer used to simulate it, so the two
look very similar (the only difference is the noise in the image).
"""
aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.model_data, title="Model Image")

r"""
The `residual_map` is the different between the observed image and model image, showing where in the image the fit is
good (e.g. low residuals) and where it is bad (e.g. high residuals).

The expression for the residual map is simply:

\[ \text{residual} = \text{data} - \text{model\_data} \]

The residual-map is plotted below, noting that all values are very close to zero because the fit is near perfect.
The only non-zero residuals are due to noise in the image.
"""
residual_map = dataset.data - fit.model_data
print("First residual-map pixel:")
print(residual_map.slim[0])

print("First residual-map pixel via fit:")
print(fit.residual_map.slim[0])

aplt.plot_array(array=fit.residual_map, title="Residual Map")

r"""
Are these residuals indicative of a good fit to the data? Without considering the noise in the data, it's difficult 
to ascertain. That is, its hard to ascenrtain if a residual value is large or small because this depends on the
amount of noise in that pixel.

The `normalized_residual_map` divides the residual-map by the noise-map, giving the residual in units of the noise.
Its expression is:

\[ \text{normalized\_residual} = \frac{\text{residual\_map}}{\text{noise\_map}} = \frac{\text{data} - \text{model\_data}}{\text{noise\_map}} \]

If you're familiar with the concept of standard deviations (sigma) in statistics, the normalized residual map represents 
how many standard deviations the residual is from zero. For instance, a normalized residual of 2.0 (corresponding 
to a 95% confidence interval) means that the probability of the model underestimating the data by that amount is only 5%.
"""
normalized_residual_map = residual_map / dataset.noise_map

print("First normalized residual-map pixel:")
print(normalized_residual_map.slim[0])

print("First normalized residual-map pixel via fit:")
print(fit.normalized_residual_map.slim[0])

aplt.plot_array(array=fit.normalized_residual_map, title="Normalized Residual Map")

r"""
Next, we define the `chi_squared_map`, which is obtained by squaring the `normalized_residual_map` and serves as a 
measure of goodness of fit.

The chi-squared map is calculated as:

\[ \chi^2 = \left(\frac{\text{data} - \text{model\_data}}{\text{noise\_map}}\right)^2 \]

Squaring the normalized residual map ensures all values are positive. For instance, both a normalized residual of -0.2 
and 0.2 would square to 0.04, indicating the same quality of fit in terms of `chi_squared`.

As seen from the normalized residual map, it's evident that the model provides a good fit to the data, in this
case because the chi-squared values are close to zero.
"""
chi_squared_map = (normalized_residual_map) ** 2
print("First chi-squared pixel:")
print(chi_squared_map.slim[0])

print("First chi-squared pixel via fit:")
print(fit.chi_squared_map.slim[0])

aplt.plot_array(array=fit.chi_squared_map, title="Chi Squared Map")

r"""
Now, we consolidate all the information in our `chi_squared_map` into a single measure of goodness-of-fit 
called `chi_squared`. 

It is defined as the sum of all values in the `chi_squared_map` and is computed as:

\[ \chi^2 = \sum \left(\frac{\text{data} - \text{model\_data}}{\text{noise\_map}}\right)^2 \]

This summing process highlights why ensuring all values in the chi-squared map are positive is crucial. If we 
didn't square the values (making them positive), positive and negative residuals would cancel each other out, 
leading to an inaccurate assessment of the model's fit to the data.

The lower the `chi_squared`, the fewer residuals exist between the model's fit and the data, indicating a better 
overall fit!
"""
chi_squared = np.sum(chi_squared_map)
print("Chi-squared = ", chi_squared)
print("Chi-squared via fit = ", fit.chi_squared)

"""
The reduced chi-squared is the `chi_squared` value divided by the number of data points (e.g., the number of pixels
in the mask). 

This quantity offers an intuitive measure of the goodness-of-fit, as it normalizes the `chi_squared` value by the
number of data points. That is, a reduced chi-squared of 1.0 indicates that the model provides a good fit to the data,
because every data point is fitted with a chi-squared value of 1.0.

A reduced chi-squared value significantly greater than 1.0 indicates that the model is not a good fit to the data,
whereas a value significantly less than 1.0 suggests that the model is overfitting the data.
"""
reduced_chi_squared = chi_squared / dataset.mask.pixels_in_mask
print("Reduced Chi-squared = ", reduced_chi_squared)

r"""
Another quantity that contributes to our final assessment of the goodness-of-fit is the `noise_normalization`.

The `noise_normalization` is computed by summing, over every pixel, the logarithm of 2 pi times the squared noise value:

\[
\text{{noise\_normalization}} = \sum \log(2 \pi \text{{noise\_map}}^2)
\]

This quantity is fixed because the noise-map remains constant throughout the fitting process. Despite this, 
including the `noise_normalization` is considered good practice due to its statistical significance.

Understanding the exact meaning of `noise_normalization` isn't critical for our primary goal of successfully 
fitting a model to a dataset. Essentially, it provides a measure of how well the noise properties of our data align 
with a Gaussian distribution.
"""
noise_normalization = np.sum(np.log(2 * np.pi * dataset.noise_map**2))
print("Noise Normalization = ", noise_normalization)
print("Noise Normalization via fit = ", fit.noise_normalization)

r"""
From the `chi_squared` and `noise_normalization`, we can define a final goodness-of-fit measure known as 
the `log_likelihood`. 

This measure is calculated by taking the sum of the `chi_squared` and `noise_normalization`, and then multiplying the 
result by -0.5:

\[ \text{log\_likelihood} = -0.5 \times \left( \chi^2 + \text{noise\_normalization} \right) \]

Don't worry about why we multiply by -0.5; it's a standard practice in statistics to ensure the log likelihood is
defined correctly.
"""
log_likelihood = -0.5 * (chi_squared + noise_normalization)
print("Log Likelihood = ", log_likelihood)
print("Log Likelihood via fit = ", fit.log_likelihood)

r"""
In the previous discussion, we noted that a lower \(\chi^2\) value indicates a better fit of the model to the 
observed data. 

When we calculate the log likelihood, we take the \(\chi^2\) value and multiply it by -0.5. This means that a 
higher log likelihood corresponds to a better model fit. Our goal when fitting models to data is to maximize the 
log likelihood.

The **reduced \(\chi^2\)** value provides an intuitive measure of goodness-of-fit. Values close to 1.0 suggest a 
good fit, while values below or above 1.0 indicate potential underfitting or overfitting of the data, respectively. 
In contrast, the log likelihood values can be less intuitive. For instance, a log likelihood value printed above 
might be around 5300.

However, log likelihoods become more meaningful when we compare them. For example, if we have two models, one with 
a log likelihood of 5300 and the other with 5310 we can conclude that the first model fits the data better 
because it has a higher log likelihood by 10.0. 

In fact, the difference in log likelihood between models can often be associated with a probability indicating how 
much better one model fits the data compared to another. This can be expressed in terms of standard deviations (sigma). 

As a rule of thumb:

- A difference in log likelihood of **2.5** suggests that one model is preferred at the **2.0 sigma** level.
- A difference in log likelihood of **5.0** indicates a preference at the **3.0 sigma** level.
- A difference in log likelihood of **10.0** suggests a preference at the **5.0 sigma** level.

All these metrics can be visualized together using the `aplt.subplot_fit_imaging` object, which offers a comprehensive 
overview of the fit quality. It also shows separate model images for the lens and source galaxies, and the appearance
of the source galaxy in the image and source planes.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

"""
If you're familiar with model-fitting, you've likely encountered terms like 'residuals', 'chi-squared', 
and 'log_likelihood' before. 

These metrics are standard ways to quantify the quality of a model fit. They are applicable not only to 1D data but 
also to more complex data structures like 2D images, 3D data cubes, or any other multidimensional datasets.

__Incorrect Fit__

In the previous section, we successfully created and fitted a lens model to the image data, resulting in an 
excellent fit. The residual map and chi-squared map showed no significant discrepancies, indicating that the 
strong lens's light was accurately captured by our model. This optimal solution translates to one of the highest log 
likelihood values possible, reflecting a good match between the model and the observed data.

Now, let's modify our lens model to create a fit that is close to the correct solution but slightly off. 
Specifically, we will slightly offset the center of the source galaxy by half a pixel (0.05") in both the x and y 
directions. This change will allow us to observe how even small deviations from the true parameters can impact the 
quality of the fit.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.15, 0.15),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

aplt.plot_array(array=tracer.image_2d_from(grid=dataset.grid), title="Image")

"""
After implementing this slight adjustment, we can now plot the fit. In doing so, we observe that residuals have 
emerged at the multiple images of the lensed source, which indicates a mismatch between our model and the data. 
Consequently, this discrepancy results in increased chi-squared values, which in turn affects our log likelihood.
"""
fit_bad = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit_bad)

"""
Next, we can compare the log likelihood of our current model to the log likelihood value we computed previously.
"""
print("Previous Likelihood:")
print(fit.log_likelihood)
print("New Likelihood:")
print(fit_bad.log_likelihood)

"""
As expected, we observe that the log likelihood has decreased! This decline confirms that our new model is indeed a 
worse fit to the data compared to the original model.

Now, let’s change our lens model once more, this time setting it to a position that is far from the true parameters. 
We will offset the source's center significantly to see how this extreme deviation affects the fit quality.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.5, 0.5),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit_very_bad = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit_very_bad)

"""
It is now evident that this model provides a terrible fit to the data. The tracer does not resemble a plausible
representation of our simulated strong lens dataset, which we already anticipated given that we generated the data ourselves!

As expected, the log likelihood has dropped dramatically with this poorly fitting model.
"""
print("Previous Likelihoods:")
print(fit.log_likelihood)
print(fit_bad.log_likelihood)
print("New Likelihood:")
print(fit_very_bad.log_likelihood)

"""
__Model Fitting__

In the previous sections, we used the true model to fit the data, which resulted in a high log likelihood and minimal 
residuals. We also demonstrated how even small deviations from the true parameters can significantly degrade the fit 
quality, reducing the log likelihood.

In practice, however, we don't know the "true" model. For example, we might have an image of a strong lens observed with 
the Hubble Space Telescope, but the values for parameters like its `einstein_radius` and others are 
unknown. The process of determining the best-fit model is called model fitting, and it is the main topic of
Chapter 2 of **HowToLens**.

To conclude this section, let's perform a basic, hands-on model fit to develop some intuition about how we can find 
the best-fit model. We'll start by loading a simple dataset that was simulated without any lens light, using 
an `IsothermalSph` lens  mass profile and `ExponentialCoreSph` source light profile, but the true parameters of these 
profiles are unknown.
"""
dataset_name = "simple__no_lens_light__mass_sis"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/no_lens_light__mass_sis.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Now, you'll try to determine the best-fit model for this image, corresponding to the parameters used to simulate the 
dataset.

We'll use the simplest possible approach: try different combinations of light and mass profile parameters and adjust 
them based on how well each model fits the data. You’ll quickly find that certain parameters produce a much better fit 
than others. For example, determining the correct values of the `centre` should not take too long.

Pay attention to the `log_likelihood` and the `residual_map` as you adjust the parameters. These will guide you in 
determining if your model is providing a good fit to the data. Aim to increase the log likelihood and reduce the 
residuals.

Keep experimenting with different values for a while, seeing how small you can make the residuals and how high you 
can push the log likelihood. Eventually, you’ll likely reach a point where further improvements become difficult, 
even after trying many different parameter values. This is a good point to stop and reflect on your first experience 
with model fitting.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.IsothermalSph(
        centre=(1.0, 1.0), einstein_radius=1.0
    ),  # These are the lens parameters you need to adjust
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(1.0, 1.0),
        intensity=1.0,
        effective_radius=1.0,
        radius_break=0.025,  # These are the source parameters you need to adjust
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)

print("Log Likelihood:")
print(fit.log_likelihood)


"""
Manually guessing model parameters repeatedly is a very inefficient and slow way to find the best fit. If the model 
were more complex—say, if the source galaxy had additional light profile components beyond just its `bulge` (like a 
second `Sersic` profile representing a `disk`)—the model would become so intricate that this manual approach 
would be practically impossible. This is definitely not how model fitting is done in practice.

However, this exercise has given you a basic intuition for how model fitting works. The statistical inference tools 
that are actually used for model fitting will be introduced in Chapter 2. Interestingly, these tools are not entirely 
different from the approach you just tried. Essentially, they also involve iteratively testing models until those 
with high log likelihoods are found. The key difference is that a computer can perform this process thousands of 
times, and it does so in a much more efficient and strategic way.

__Wrap Up__

In this tutorial, you have learned how to fit a lens model to imaging data, a fundamental process in astronomy
and statistical inference. 

Let's summarise what we have covered:

- **Dataset**: We loaded the imaging dataset that we previously simulated, consisting of the tracer image, noise map,
  and PSF.
  
- **Mask**: We applied a circular mask to the data, excluding regions with low signal-to-noise ratios from the analysis.

- **Masked Grid**: We created a masked grid, which contains only the coordinates of unmasked pixels, to evaluate the
  tracer's light profile.
  
- **Fitting**: We fitted the data with a lens model, computing key quantities like the model image, residuals,
  chi-squared, and log likelihood to assess the quality of the fit.
  
- **Bad Fits**: We demonstrated how even small deviations from the true parameters can significantly impact the fit
  quality, leading to decreased log likelihood values.
  
- **Model Fitting**: We performed a basic model fit on a simple dataset, adjusting the model parameters to improve the
  fit quality.  
"""
