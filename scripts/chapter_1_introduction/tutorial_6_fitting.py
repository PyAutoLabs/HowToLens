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

We will begin with the same lens and source parameters used to simulate the
data. This gives us a useful reference: we know the underlying model, so we
can see what agreement with a noisy observation should look like.

We should not expect the predicted image to match every observed pixel
exactly. The data contain random noise, whereas our prediction describes
the expected brightness.

As before, the lens has an Isothermal mass profile and the source has a
Sersic light profile. The lens galaxy's light was omitted from this simulation.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=(0.17647, 0.0),
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

"""
First, we evaluate the tracer on the dataset's grid. This calculates the
lensed source image at the selected image-plane coordinates.

The arcs should look familiar, but this image has not yet been blurred by
the telescope's PSF. We must include that blurring before comparing it with
the observed data.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=dataset.grid),
    title="Lensed Image Before PSF Blurring",
)

"""
The `FitImaging` object brings together our dataset and tracer. It calculates
the predicted image with PSF blurring and compares it with the data.

We supply the parameter values through the tracer. Creating this object
evaluates that choice of parameters; it does not adjust them.

We will inspect the predicted image first, then calculate the differences
between it and the data. These differences will form the basis of the
statistical quantities introduced below.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
The fit's `model_data` is the predicted image after PSF convolution. It has
the same units as the observed data and can be compared with them pixel
by pixel.

The mask selects which observed pixels enter the comparison, but light
outside its boundary can still be blurred into those pixels by the PSF.
`FitImaging` therefore also evaluates the surrounding light needed for
this convolution.

This is the same principle as the padding in Tutorial 5: we must account
for light beyond the edge of the region whose blurred image we want.

Below, compare the data and model image. Their arcs should have similar
positions, shapes and brightnesses. The model image is smooth, while the
data include noise. We do not add a new random noise realisation to the
model image when assessing the fit.
"""
print("First model image pixel:")
print(fit.model_data.slim[0])

aplt.plot_array(array=fit.data, title="Data")
aplt.plot_array(array=fit.model_data, title="Model Image")

r"""
Subtracting the model image from the data gives the `residual_map`:

\[
\text{residual} = \text{data} - \text{model\_data}
\]

A positive residual means that the observed pixel is brighter than predicted.
A negative residual means that it is fainter.

We calculate this difference explicitly below and compare its first value
with the value provided by `FitImaging`.
"""
residual_map = dataset.data - fit.model_data

print("First residual-map pixel:")
print(residual_map.slim[0])

print("First residual-map pixel via fit:")
print(fit.residual_map.slim[0])

aplt.plot_array(array=fit.residual_map, title="Residual Map")

"""
Because we used the parameters that generated the simulation, we expect
the residuals to be dominated by the noise in the data.

Look for positive and negative fluctuations rather than coherent arc-shaped
features. The fluctuations can be larger in bright regions, where the
Poisson noise is larger.

To judge whether a residual is large or small, we must compare it with
the noise expected in that pixel. This is our next step.

**🤖 Ask AI:** Why can a correct lens model leave larger absolute residuals
in bright arcs than in the faint background?
"""

r"""
The size of a residual only becomes informative when compared with the
uncertainty in that pixel. A difference of 0.1 electrons per second could
be small in a noisy pixel but substantial in a precisely measured one.

The `normalized_residual_map` divides each residual by its noise standard
deviation:

\[
r_i = \frac{d_i - m_i}{\sigma_i}
\]

Here, \(d_i\) is the observed value, \(m_i\) is the model prediction and
\(\sigma_i\) is the noise-map value for pixel \(i\).

A normalised residual of +2 means that the observation is two noise standard
deviations above the prediction. A value of -2 means that it is two standard
deviations below.

For a suitable model and well-estimated noise, we expect fluctuations around
zero, typically of order one. We do not expect every pixel to be close to zero.
"""
normalized_residual_map = residual_map / dataset.noise_map

print("First normalized residual-map pixel:")
print(normalized_residual_map.slim[0])

print("First normalized residual-map pixel via fit:")
print(fit.normalized_residual_map.slim[0])

aplt.plot_array(
    array=fit.normalized_residual_map,
    title="Normalized Residual Map",
)

r"""
Squaring the normalised residuals gives the `chi_squared_map`:

\[
\chi_i^2 = \left(\frac{d_i - m_i}{\sigma_i}\right)^2
\]

Each value measures one pixel's contribution to the overall disagreement.
The values are non-negative: residuals of +2 and -2 both contribute 4.

Look for pixels with large contributions and whether they form a coherent
pattern. A few large values can occur through noise, especially when an image
contains many pixels.

For our reference model, the contributions should average approximately one,
rather than all being close to zero.
"""
chi_squared_map = normalized_residual_map**2

print("First chi-squared pixel:")
print(chi_squared_map.slim[0])

print("First chi-squared pixel via fit:")
print(fit.chi_squared_map.slim[0])

aplt.plot_array(array=fit.chi_squared_map, title="Chi Squared Map")

r"""
We sum the contributions from all \(N\) unmasked pixels to obtain one number,
the total `chi_squared`:

\[
\chi^2 = \sum_{i=1}^{N}
\left(\frac{d_i - m_i}{\sigma_i}\right)^2
\]

Squaring before summing prevents positive and negative residuals from
cancelling. Large differences relative to the noise contribute strongly.

When comparing parameter choices using the same data, mask and noise map,
a lower chi-squared means smaller overall noise-weighted differences.
"""
chi_squared = np.sum(chi_squared_map.slim)

print("Chi-squared = ", chi_squared)
print("Chi-squared via fit = ", fit.chi_squared)

r"""
Dividing the total chi-squared by the number of unmasked pixels gives the
average contribution per pixel:

\[
\frac{\chi^2}{N}
\]

For the parameters that generated our simulation, we expect this to be
approximately one if the noise estimates describe the fluctuations well.
Individual pixels can contribute much more or much less than one.

A substantially larger value can indicate a mismatch between the model and
data, underestimated noise, or both. A substantially smaller value can
indicate overestimated noise; it does not automatically establish overfitting.

You may also encounter the term "reduced chi-squared". This divides chi-squared
by the number of degrees of freedom, often \(N-p\) when \(p\) independent
parameters have been estimated from the data. Here, we are simply inspecting
the average per pixel for a specified model.
"""
chi_squared_per_pixel = chi_squared / dataset.mask.pixels_in_mask

print("Chi-squared per pixel = ", chi_squared_per_pixel)

r"""
To turn this comparison into a likelihood, we need a statistical description
of the noise.

We will treat each pixel's measurement as an independent Gaussian random
variable, centred on the model prediction \(m_i\), with standard deviation
\(\sigma_i\). This is an approximation to the Poisson counts used in our
simulation, appropriate when the expected counts are sufficiently large.

The probability density for a pixel value is then:

\[
p(d_i \mid m_i, \sigma_i)
=
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left[-\frac{1}{2}
\left(\frac{d_i-m_i}{\sigma_i}\right)^2\right]
\]

The exponential penalises predictions far from the observed value. The factor
in front normalises the Gaussian so that its density integrates to one.

For independent pixels, we multiply these densities. Taking the natural
logarithm turns that product into a sum:

\[
\log \mathcal{L}
=
-\frac{1}{2}
\left[
\chi^2 + \sum_{i=1}^{N}\log(2\pi\sigma_i^2)
\right]
\]

The second term is called `noise_normalization` in PyAutoLens. It is part of
the probability density, not a test of whether the noise is Gaussian.

For our fixed dataset, mask and noise map, this term stays constant as we
change the lens and source parameters.
"""
noise_normalization = np.sum(
    np.log(2 * np.pi * dataset.noise_map.slim**2)
)

print("Noise Normalization = ", noise_normalization)
print("Noise Normalization via fit = ", fit.noise_normalization)

"""
We can now calculate the log likelihood and compare it with the value
returned by `FitImaging`.

The factor of -0.5 follows from the Gaussian expression above. With the
noise normalisation fixed, reducing chi-squared increases the log likelihood.
"""
log_likelihood = -0.5 * (chi_squared + noise_normalization)

print("Log Likelihood = ", log_likelihood)
print("Log Likelihood via fit = ", fit.log_likelihood)

"""
For the same dataset, mask and noise assumptions, a higher log likelihood
indicates better agreement between the predicted image and the data.

For example, a log likelihood of 5310 is higher than 5300, so the former
parameter choice provides the better fit under those assumptions.

The absolute value is not a probability that the model is correct. Nor does
a difference in log likelihood have a universal conversion into a number
of standard deviations.

We should also inspect where the model disagrees with the data. The subplot
below brings together the data, model image and residual diagnostics, along
with views of the source.

Look for residuals consistent with noise rather than coherent features
tracing the lensed arcs.
"""
aplt.subplot_fit_imaging(fit=fit)

"""
__Incorrect Fit__

We will now change the source position while keeping the lens mass and all
other source properties fixed.

The original source centre was `(0.1, 0.1)` arcseconds. We move it to
`(0.15, 0.15)`, a displacement of 0.05 arcseconds in each source-plane
coordinate.

Although 0.05 arcseconds is half our image pixel scale, lensing maps this
source-plane displacement into changes in the positions and shapes of the
observed arcs. It does not simply move every image by half a pixel.

The original `fit` remains our reference comparison.
"""
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

aplt.plot_array(
    array=tracer.image_2d_from(grid=dataset.grid),
    title="Lensed Image With Shifted Source",
)

"""
We evaluate this new tracer against the same dataset and inspect the fit.

Look for coherent positive and negative residuals around the arcs. These
indicate regions where the shifted source predicts too little or too much
light, rather than just random fluctuations about the observation.
"""
fit_bad = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit_bad)

"""
Compare the log likelihood with our reference value. We expect the shifted
source to give a lower value because its predicted image disagrees with
the data more strongly.
"""
print("Reference log likelihood:")
print(fit.log_likelihood)

print("Shifted-source log likelihood:")
print(fit_bad.log_likelihood)

print("Change from reference:")
print(fit_bad.log_likelihood - fit.log_likelihood)

"""
Next, move the source centre to `(0.5, 0.5)` arcseconds. All other parameters
remain unchanged.

This is a much larger displacement from the original `(0.1, 0.1)` position.
Compare the predicted arcs and residuals with the previous two examples.
"""
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
Print the three log likelihoods together and compare their ordering with
what you saw in the residual plots.

For this example, the larger source displacement should produce the strongest
disagreement and the lowest log likelihood.
"""
print("Reference source centre (0.1, 0.1):")
print(fit.log_likelihood)

print("Source centre (0.15, 0.15):")
print(fit_bad.log_likelihood)

print("Source centre (0.5, 0.5):")
print(fit_very_bad.log_likelihood)

"""
We have evaluated three choices of parameters, rather than searched for
the best-fitting values.

The parameters used to generate the simulation provide a useful reference,
but they need not give the highest possible likelihood for one noisy
observation. Small parameter changes can sometimes improve agreement with
that particular noise realisation.

To restore the original comparison, rerun the first tracer and `FitImaging`
cells in the Fitting section.

**🤖 Ask AI:** Why might the parameters that maximise the likelihood of a
noisy simulated image differ slightly from those used to generate it?
"""

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
