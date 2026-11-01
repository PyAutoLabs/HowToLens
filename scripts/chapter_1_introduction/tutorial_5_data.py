"""
Tutorial 5: Data
================

In the previous tutorials, we used tracers to create images of strong lenses.
We will now use one of these images to simulate an observation through a telescope.

A telescope spreads light over neighbouring pixels, making fine details less
sharp. Its detector collects light for a finite exposure time, and the number
of photons detected fluctuates from one observation to another. Light from the
background sky also contributes to the recorded image.

We will introduce these effects separately so that you can see what each one
does, then combine them using PyAutoLens's `SimulatorImaging` object.

Our example uses a charge-coupled device (CCD), a detector that records incoming
light as electrons collected in its pixels.

__Contents__

- **Initial Setup:** Create the lensed image that we will use for the simulation.
- **Optics Blurring:** Describe the telescope's blurring with a point spread function.
- **Poisson Noise:** Simulate fluctuations in the number of electrons collected.
- **Background Sky:** Include sky light and the additional noise it produces.
- **Simulator:** Combine these effects into an imaging dataset.
- **Output:** Save the data, noise map and point spread function to FITS files.
- **Point Source Data:** Introduce measured positions, fluxes and time delays.
- **Interferometer Data:** Introduce measurements made by radio interferometers.
- **Weak Lensing Data:** Introduce catalogues of background galaxy shapes.
- **Wrap Up:** Review what we have learned.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path
import autoarray as aa
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We begin with a grid of 101 x 101 pixels, with each pixel spanning 0.1 arcseconds
on the sky. As in the earlier tutorials, the grid supplies the coordinates at
which we calculate the lensed image.
"""
grid = al.Grid2D.uniform(
    shape_native=(101, 101),
    pixel_scales=0.1,
)

"""
Our lens galaxy has an `Isothermal` mass profile. We omit its light so that we
can focus on how the observation changes the appearance of the lensed source.
The source galaxy has a Sersic light profile.

For this simulation, image values represent electrons per second in each pixel.
Multiplying a pixel's value by the exposure time will therefore give the
expected number of electrons collected in that pixel.

We use the same units for the light profile's `intensity`, which sets its
brightness at the effective radius.
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
        intensity=1.0,  # Electrons per second per pixel.
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
First, let's look at the lensed image. The telescope's blurring, background sky
and random noise have not yet been included.

Look at the sharpness of the arcs and the faint light around them. This is our
starting image for seeing how each observational effect changes their appearance.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Lensed Image Before Simulating the Observation",
)

"""
__Optics Blurring__

A point source does not appear as a single perfectly sharp point in a telescope
image. The telescope spreads its light over a small region. The shape of this
response is called the point spread function, or PSF.

An extended image is blurred in the same way: light from every position is
spread according to the PSF. We calculate this using an operation called
convolution.

For our example, we will use a circular Gaussian PSF. Its `sigma` is
0.1 arcseconds, equal to one pixel on our grid. A larger sigma would spread the
light over a wider region.

The PSF is represented by an 11 x 11 array called a kernel. We normalise its
values to sum to one, so that it describes how light is redistributed.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.1,  # Arcseconds.
    pixel_scales=grid.pixel_scales,
    normalize=True,
)

"""
Plotting the kernel shows how the light from a point source is distributed:
the central pixels receive the most light, with progressively less farther
from the centre.
"""
aplt.plot_array(array=psf.kernel, title="PSF 2D Kernel")

"""
The PSF is often more informative when plotted on a log10 scale. This approach allows us to clearly observe values 
in its tail, which are much smaller than the central peak yet critical for many scientific analyses. The tail 
values may significantly affect the spread and detail captured in the data.
"""
aplt.plot_array(array=psf.kernel, title="PSF 2D Kernel (Log10)", use_log10=True)

"""
Next, we'll manually perform a 2D convolution of the PSF with the image of the galaxy. This convolution simulates the 
blurring that occurs when the telescope optics spread out the galaxy's light.

1. **Padding the Image**: Before convolution, we add padding (extra space with zero values) around the edges of the 
   image. This prevents unwanted edge effects when we perform the convolution, ensuring that the image's edges don't 
   become artificially altered by the process.

2. **Convolution**: Using the `Convolver` object's `convolve` method, we apply the 2D PSF convolution to the padded 
   image. This step combines the PSF with the galaxy's light, simulating how the telescope spreads out the light.

3. **Trimming the Image**: After convolution, we trim the padded areas back to their original size, obtaining a 
   convolved (blurred) image that matches the dimensions of the initial tracer image.
"""
image = tracer.image_2d_from(grid=grid)  # The original unblurred image of the galaxy.
padded_image = tracer.padded_image_2d_from(
    grid=grid,
    psf_shape_2d=psf.kernel.shape_native,  # Adding padding based on the PSF size.
)
convolved_image = psf.convolved_image_from(
    image=padded_image, blurring_image=None
)  # Applying the PSF convolution.
blurred_image = convolved_image.trimmed_after_convolution_from(
    kernel_shape=psf.kernel.shape_native
)  # Trimming back to the original size.

"""
We can now plot the original and the blurred images side by side. This allows us to clearly see how the PSF 
convolution affects the appearance of the galaxy, making the image appear softer and less sharp.
"""
aplt.plot_array(array=image, title="Tracer Image Before PSF")

aplt.plot_array(array=blurred_image, title="Tracer Image After PSF")


"""
__Poisson Noise__

In addition to the blurring caused by telescope optics, we also need to consider Poisson noise when simulating imaging 
data.

When a telescope captures an image of a galaxy, photons from the galaxy are collected by the telescope's mirror and 
directed onto a CCD (Charge-Coupled Device). The CCD is made up of a silicon lattice (or another material) that 
converts incoming photons into electrons. These electrons are then gathered into discrete squares, which form the 
pixels of the final image.

The process of converting photons into electrons is inherently random, following a Poisson distribution. This randomness 
means that the number of electrons in each pixel can vary, even if the same number of photons hits the CCD. Therefore, 
the electron count per pixel becomes a Poisson random variable. For our simulation, this means that the recorded 
number of photons in each pixel will differ slightly from the true number due to this randomness.

To replicate this effect in our simulation, we can add Poisson noise to the tracer image using NumPy’s random module, 
which generates values from a Poisson distribution.

It's important to note that the blurring caused by the telescope optics occurs before the photons reach the CCD. 
Therefore, we need to add the Poisson noise after blurring the tracer image.

We also need to consider the units of our image data. Let’s assume that the tracer image is measured in units of 
electrons per second ($e^- s^{-1}$), which is standard for CCD imaging data. To simulate the number of electrons 
actually detected in each pixel, we multiply the image by the observation’s exposure time. This conversion changes
the units to the total number of electrons collected per pixel over the entire exposure time.

Once the image is converted, we add Poisson noise, simulating the randomness in the photon-to-electron conversion 
process. After adding the noise, we convert the image back to units of electrons per second for analysis, as 
this is the preferred unit for astronomers when studying their data.
"""
exposure_time = 300.0  # Units of seconds
blurred_image_counts = (
    blurred_image * exposure_time
)  # Convert to total electrons detected over the exposure time.
blurred_image_with_poisson_noise = (
    np.random.poisson(blurred_image_counts, blurred_image_counts.shape) / exposure_time
)  # Add Poisson noise and convert back to electrons per second.

"""
Here is what the blurred image with Poisson noise looks like.
"""
aplt.plot_array(
    array=aa.Array2D(values=blurred_image_with_poisson_noise, mask=blurred_image.mask),
    title="Image With Poisson Noise",
)

"""
It is challenging to see the Poisson noise directly in the image above, as it is often subtle. To make the noise more 
visible, we can subtract the blurred image without Poisson noise from the one with noise.

This subtraction yields the "Poisson noise realization" which highlights the variation in each pixel due to the Poisson 
distribution of photons hitting the CCD. It represents the noise values that were added to each pixel. We call
it the realization because it is one possible outcome of the Poisson process, and the noise will be different each time
we simulate the image.
"""
poisson_noise_realization = blurred_image_with_poisson_noise - blurred_image

aplt.plot_array(
    array=aa.Array2D(values=poisson_noise_realization, mask=blurred_image.mask),
    title="Poisson Noise Realization",
)

"""
__Background Sky__

The final effect we will consider when simulating imaging data is the background sky.

In addition to light from the strong lens, the telescope also picks up light from the sky. This background sky light is 
primarily due to two sources: zodiacal light, which is light scattered by interplanetary dust in the solar system, 
and the unresolved emission from distant stars and galaxies.

For our simulation, we'll assume that the background sky has a uniform brightness across the image, measured at 
0.1 electrons per second per pixel. The background sky is added to the image before applying the PSF convolution 
and adding Poisson noise. This is important because it means that the background contributes additional noise to the 
image.

The background sky introduces noise throughout the entire image, including areas where the strong lens is not present. 
This is why CCD images often appear noisy, especially in regions far from where the strong lens signal is detected. 
The sky noise can make it more challenging to observe faint details of the lens and source galaxies.

To simulate this, we add a constant background sky to the tracer image and then apply Poisson noise to create the 
final simulated image as it would appear through a telescope.
"""
background_sky_level = 0.1

# Add background sky to the blurred tracer image.
blurred_image_with_sky = blurred_image + background_sky_level
blurred_image_with_sky_counts = blurred_image_with_sky * exposure_time

# Apply Poisson noise to the image with the background sky.
blurred_image_with_sky_poisson_noise = (
    np.random.poisson(
        blurred_image_with_sky_counts, blurred_image_with_sky_counts.shape
    )
    / exposure_time
)

# Visualize the image with background sky and Poisson noise.
aplt.plot_array(
    array=aa.Array2D(
        values=blurred_image_with_sky_poisson_noise, mask=blurred_image.mask
    ),
    title="Image With Background Sky",
)

# Create a noise map showing the differences between the blurred image with and without noise.
poisson_noise_realization = (
    blurred_image_with_sky_poisson_noise - blurred_image_with_sky
)

aplt.plot_array(
    array=aa.Array2D(values=poisson_noise_realization, mask=blurred_image.mask),
    title="Poisson Noise Realization",
)

"""
__Simulator__

The `SimulatorImaging` object lets us create simulated imaging data while including the effects of PSF blurring, 
Poisson noise, and background sky all at once:
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
By plotting the `data` from the dataset, we can see that it matches the image we simulated earlier. It includes 
the effects of PSF blurring, Poisson noise, and noise from the background sky. This image is a realistic 
approximation of what a telescope like the Hubble Space Telescope would capture.
"""
aplt.plot_array(array=dataset.data, title="Simulated Imaging Data")

"""
The dataset also includes the `psf` (Point Spread Function) used to blur the strong lens image.

For actual telescope data, the PSF is determined during data processing and is provided along with the observations. 
It's crucial for accurately deconvolving the PSF from the strong lens image, allowing us to recover the true properties 
of the strong lens. We'll explore this further in the next tutorial.
"""
aplt.plot_array(array=dataset.psf.kernel, title="Simulated PSF")

"""
The dataset includes a `noise_map`, which represents the Root Mean Square (RMS) standard deviation of the noise 
estimated for each pixel in the image. Higher noise values mean that the measurements in those pixels are 
less certain, so those pixels are given less weight when analyzing the data.

This `noise_map` is different from the Poisson noise arrays we plotted earlier. The Poisson noise arrays show the 
actual noise added to the image due to the random nature of photon-to-electron conversion on the CCD, as calculated 
using the numpy random module. These noise values are theoretical and cannot be directly measured in real telescope data.

In contrast, the `noise_map` is our best estimate of the noise present in the image, derived from the data itself 
and used in the fitting process.
"""
aplt.plot_array(array=dataset.noise_map, title="Simulated Noise Map")

"""
The `signal-to-noise_map` shows the ratio of the signal in each pixel to the noise level in that pixel. It is 
calculated by dividing the `data` by the `noise_map`.

This ratio helps us understand how much of the observed signal is reliable compared to the noise, allowing us to 
see where we can trust the detected signal from the strong lens and where the noise is more significant.

In general, a signal-to-noise ratio greater than 3 indicates that the signal is likely real and not overwhelmed by 
noise. For our datasets, the signal-to-noise ratio peaks at ~70, meaning we can trust the signal detected in the
image.
"""
aplt.plot_array(array=dataset.signal_to_noise_map, title="Signal-To-Noise Map")

"""
The `aplt.subplot_imaging_dataset` object can display all of these components together, making it a powerful tool for visualizing 
simulated imaging data.

It also shows the Data and PSF on a logarithmic (log10) scale, which helps highlight the faint details in these 
components.

The "Over Sampling" plots on the bottom of the figures display advanced features that can be ignored for now.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

We will now save these simulated data to `.fits` files, the standard format used by astronomers for storing images.
Most imaging data from telescopes like the Hubble Space Telescope (HST) are stored in this format.

The `dataset_path` specifies where the data will be saved, in this case the `dataset/imaging/howtolens/`
directory of the HowToLens repository (paths are relative to the repository root, from which scripts are run).

The files are named `data.fits`, `noise_map.fits`, and `psf.fits`, and will be used in the next tutorial.
"""
dataset_path = Path("dataset") / "imaging" / "howtolens"
print("Dataset Path: ", dataset_path)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    overwrite=True,
)

"""
__Point Source Data__

In Tutorial 4, we predicted the positions of a point source's multiple images. For real observations, we measure
these positions from the telescope image, accounting for its blurring and the foreground galaxy's light. Each
position also has an uncertainty.

A `PointDataset` collects these measurements for lens modelling. The illustrative values below are rounded from
Tutorial 4's predictions, rather than measured from observations. We include positions in arcseconds, fluxes in
arbitrary units and time delays in days relative to the first image, with assumed uncertainties of 0.01 arcseconds,
about 10 per cent in flux and one day in delay.
"""
point_dataset = al.PointDataset(
    name="point_0",
    positions=[
        (-1.039, -1.038),
        (0.442, 1.608),
        (1.609, 0.442),
        (1.179, 1.180),
    ],
    positions_noise_map=0.01,
    fluxes=[6.82, 55.20, 53.67, 100.71],
    fluxes_noise_map=[0.68, 5.52, 5.37, 10.07],
    time_delays=[0.0, -39.86, -40.03, -39.75],
    time_delays_noise_map=[1.0, 1.0, 1.0, 1.0],
)

print(point_dataset.info)

"""
The positions use `(y, x)` coordinates, and every list follows the same image order. Negative delays mean that
an image arrives before the first image. The name `point_0` matches the point-source component introduced in
Tutorial 4, connecting these measurements to that source when fitting a model.

Fluxes and time delays are optional: positions alone are enough to begin modelling. For more on loading and
fitting these datasets, continue to the workspace's
[point-source examples](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/point_source/start_here.py).

__Interferometer Data__

CCD imaging is not the only type of data used to study strong lenses. Radio and sub-mm interferometers, like the
Atacama Large Millimeter Array (ALMA) and the Jansky Very Large Array (JVLA), observe strong lenses at wavelengths
where a CCD cannot.

An interferometer does not observe an image of the lens. Each pair of antennas in the array measures a "visibility",
a Fourier component of the sky brightness, at a point in what is called the "uv-plane" set by the separation of the
two antennas. The dataset is therefore a set of complex visibilities in Fourier space, not a 2D image, and its
noise properties are very different from those of CCD data — there is no PSF convolution, Poisson noise or
background sky; instead each visibility has Gaussian noise.

One could Fourier transform the visibilities into an image (called a "dirty image") and fit that, but the transform
correlates the noise between pixels, making the fit statistically incorrect. **PyAutoLens** therefore fits lens
models directly in visibility space: the tracer's image is evaluated in real space on a grid (defined by a
real-space mask), Fourier transformed to the uv-plane and compared with the observed visibilities there.

Below, we load a simulated interferometer dataset (creating it first via the `scripts/simulator/interferometer.py`
script if it does not exist on your hard-disk, using the same auto-simulation idiom as later tutorials) and plot
its dirty images — the closest an interferometer dataset comes to the CCD images we simulated above.

HowToLens will not cover interferometry any further than this. The lecture series teaches lensing using CCD
imaging, and everything you learn transfers to visibility-space fitting. If you need to model interferometer data,
go to the `autolens_workspace/scripts/interferometer` package, which is the dedicated resource for uv-plane lens
modeling.
"""
dataset_path = Path("dataset") / "interferometer" / "simple"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/interferometer.py"],
        check=True,
    )

real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.1,
    radius=3.0,
)

dataset = al.Interferometer.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Weak Lensing Data__

There is one more type of lensing data to glimpse before we move on: weak lensing.

A weak lensing dataset looks nothing like the images above. Instead of the spectacular arcs and rings of strong
lensing, it is a *catalogue* of many background galaxies, each with a measured ellipticity — a shear — at its
(y, x) position on the sky. Each galaxy is only weakly sheared by the foreground mass, a distortion far too small
to see in any single galaxy, and the plot below shows this: a field of short line segments tracing the subtle,
coherent stretching of the background galaxy population, rather than any arc-like feature.

We load a simulated weak lensing shear catalogue (again auto-simulating it, via `scripts/simulator/weak_lensing.py`,
if it is not on your hard-disk) and plot it, just so you have seen what this data looks like.

That is deliberately all we will say for now — this glimpse is foreshadowing. Weak lensing gets a full treatment in
the final tutorial of chapter 4, which describes what these shear measurements are, why they are made far from the
lens centre and how they are fitted; until then, we defer all further description to that tutorial. For worked
modelling examples, see the workspace
[weak-lensing examples](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/weak/start_here.py).
"""
dataset_path = Path("dataset") / "weak_lensing" / "simple"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/weak_lensing.py"],
        check=True,
    )

dataset = al.from_json(file_path=dataset_path / "dataset.json")

aplt.subplot_weak_dataset(dataset=dataset)

"""
__Wrap Up__

In this tutorial, you learned how CCD imaging data of a lens is collected using real telescopes like the 
Hubble Space Telescope, and how to simulate this data using the `SimulatorImaging` object.

Let's summarise what we've covered:

- **Optics Blurring**: The optics of a telescope blur the light of the tracer, reducing the clarity and sharpness of
the images.

- **Poisson Noise**: The process of converting photons to electrons on a CCD introduces Poisson noise, which is random 
variability in the number of electrons collected in each pixel.

- **Background Sky**: Light from the sky is captured along with light from the lens, adding a layer of noise across 
the entire image.

- **Simulator**: The `SimulatorImaging` object enables us to simulate realistic imaging data by including all of 
these effects together and contains the `data`, `psf`, and `noise_map` components.

- **Output**: We saved the simulated data to `.fits` files, the standard format used by astronomers for storing images.

- **Point Source Data**: A `PointDataset` collects image positions and their uncertainties, with optional fluxes
and time delays, ready for point-source modelling.
"""
