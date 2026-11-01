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
The PSF becomes much fainter away from its centre. On the previous plot's linear
colour scale, these faint outer regions can be difficult to see.

A logarithmic colour scale makes them more visible. The kernel values are
unchanged; only the way we display them changes.
"""
aplt.plot_array(
    array=psf.kernel,
    title="PSF 2D Kernel (Log10)",
    use_log10=True,
)

"""
We will now convolve the lensed image with the PSF to calculate how it appears
after the telescope has blurred it.

Light just outside the image can be spread into pixels near its edges. To
include this light, we first evaluate the tracer on a larger grid, extending
beyond the region we want to observe. This produces a padded image containing
the model's predicted light in the surrounding pixels.

We then convolve this larger image with the PSF and trim away the extra border.
The result has the same dimensions as our original image.
"""
image = tracer.image_2d_from(grid=grid)

padded_image = tracer.padded_image_2d_from(
    grid=grid,
    psf_shape_2d=psf.kernel.shape_native,
)

convolved_image = psf.convolved_image_from(
    image=padded_image,
    blurring_image=None,
)

blurred_image = convolved_image.trimmed_after_convolution_from(
    kernel_shape=psf.kernel.shape_native
)

"""
Compare the lensed image before and after convolution. Look at the narrow
features in the arcs: the PSF spreads their light over neighbouring pixels,
making them less sharp. With our small PSF, the difference may be subtle.

To make the effect easier to see, change `sigma` in the PSF definition from
0.1 to 0.2 arcseconds. Rerun the PSF definition, convolution and plots below.
You should see stronger blurring in the second image.

Afterwards, restore `sigma=0.1` and rerun those cells before continuing.
"""
aplt.plot_array(array=image, title="Tracer Image Before PSF")

aplt.plot_array(array=blurred_image, title="Tracer Image After PSF")

"""
**🤖 Ask AI:** How does blurring by a telescope's PSF affect our ability to
distinguish two nearby features in a gravitationally lensed arc?
"""

"""
__Poisson Noise__

Even if a galaxy's brightness remains constant, repeated observations do not
collect exactly the same number of photons in each pixel. Photon arrivals
fluctuate randomly, and the detector records these arrivals as electrons.

For this simulation, we describe the number of electrons collected in each
pixel using a Poisson distribution. Its mean is the expected electron count.
Drawing a value from this distribution gives one possible observed count,
which can be above or below that expectation.

Our blurred image gives the expected electron rate in each pixel. Multiplying
it by an exposure time of 300 seconds gives the expected counts for that
exposure. We use these counts as the means of the Poisson distributions.

The telescope spreads the light before the detector collects it, so we draw
the counts from the blurred image.

Finally, we divide the randomly drawn counts by the exposure time to return
the image to electrons per second per pixel.
"""
exposure_time = 300.0  # Seconds.

blurred_image_counts = blurred_image * exposure_time

blurred_image_with_poisson_noise = (
    np.random.poisson(blurred_image_counts, blurred_image_counts.shape)
    / exposure_time
)

"""
Let's plot the resulting image. It contains the same lensed structure as
before, with random fluctuations around the expected brightness in each pixel.
These fluctuations may be difficult to distinguish at this display scale.
"""
aplt.plot_array(
    array=aa.Array2D(
        values=blurred_image_with_poisson_noise,
        mask=blurred_image.mask,
    ),
    title="Image With Poisson Noise",
)

"""
We can isolate the fluctuations by subtracting the expected blurred image from
the simulated observation.

This difference is called a noise realisation: one particular outcome of the
random process. Positive values indicate pixels that collected more electrons
than expected; negative values indicate pixels that collected fewer.

Rerunning the Poisson draw produces a different realisation. We can calculate
the exact difference here because we know the image used to generate the
simulation. For a real observation, that underlying image is unknown.
"""
poisson_noise_realization = blurred_image_with_poisson_noise - blurred_image

aplt.plot_array(
    array=aa.Array2D(values=poisson_noise_realization, mask=blurred_image.mask),
    title="Poisson Noise Realization",
)

"""
__Background Sky__

The telescope collects light from the surrounding sky as well as from the
strong lens. Sources of this background include sunlight scattered by dust
in the Solar System, known as zodiacal light, and unresolved distant objects.

We will represent the background as a uniform brightness of 0.1 electrons
per second per pixel. We add this to the blurred lensed image before drawing
the Poisson counts, so that both the source and sky contribute to the noise.

A uniform background remains uniform when convolved with a normalised PSF,
so we can add it after blurring the lensed image.

Even where the source contributes very little light, sky photons are still
collected. Their fluctuating counts make faint features harder to distinguish.
"""
background_sky_level = 0.1

blurred_image_with_sky = blurred_image + background_sky_level
blurred_image_with_sky_counts = blurred_image_with_sky * exposure_time

blurred_image_with_sky_poisson_noise = (
    np.random.poisson(
        blurred_image_with_sky_counts,
        blurred_image_with_sky_counts.shape,
    )
    / exposure_time
)

"""
This image includes the mean sky brightness as well as the random fluctuations
in the collected counts. Regions away from the lensed source therefore have
a background level around which their pixel values fluctuate.
"""
aplt.plot_array(
    array=aa.Array2D(
        values=blurred_image_with_sky_poisson_noise,
        mask=blurred_image.mask,
    ),
    title="Image With Background Sky",
)

"""
Subtracting the expected source-plus-sky image isolates the noise realisation.
Look for fluctuations even in regions where the lensed source is faint:
the background sky contributes noise there too.
"""
poisson_noise_realization = (
    blurred_image_with_sky_poisson_noise - blurred_image_with_sky
)

aplt.plot_array(
    array=aa.Array2D(
        values=poisson_noise_realization,
        mask=blurred_image.mask,
    ),
    title="Poisson Noise Realization",
)

"""
__Simulator__

We have now built up the observation one step at a time. The `SimulatorImaging`
object combines these steps: it blurs the lensed image with the PSF, adds the
background sky and draws the Poisson counts for the chosen exposure time.

By default, it also subtracts the mean background sky level from the resulting
image. This represents an observation whose background has been removed during
data processing.

Subtracting the mean sky level does not remove the random fluctuations caused
by sky photons. That noise remains in the image.

The simulator returns an `Imaging` dataset containing the simulated image,
the PSF and a noise map. We will inspect each of these below.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
The dataset's `data` contains the simulated observation, in electrons per
second per pixel.

It shows the same lensed structure as our manual simulation, but uses a new
noise realisation and has the mean sky level subtracted. Pixels far from the
source should therefore fluctuate around zero. Some can be negative: after
sky subtraction, a pixel that collected fewer sky photons than expected can
have a negative value.

These negative values do not represent negative light. They arise from
subtracting the estimated background from a noisy measurement.
"""
aplt.plot_array(array=dataset.data, title="Simulated Imaging Data")

"""
**🤖 Ask AI:** Why does subtracting the mean background sky from an astronomical
image leave the sky's Poisson noise behind?
"""

"""
The dataset also stores the PSF used to simulate the observation.

For real observations, we need an estimate of the PSF, which can come from
stars in the image or a model of the instrument.

When comparing a lens model with the data, we blur its predicted image using
this PSF. This lets us compare the model and observation at the same resolution.
The next tutorial will show this comparison in practice.
"""
aplt.plot_array(array=dataset.psf.kernel, title="Simulated PSF")

"""
The `noise_map` gives an estimated standard deviation for each pixel, in the
same units as the data. It describes the typical size of the random
fluctuations expected in that pixel.

This is different from the noise realisation we plotted earlier. A realisation
records the particular positive or negative fluctuation drawn for each pixel;
the noise map describes the scale of those fluctuations.

For Poisson counts, the standard deviation is the square root of the expected
number of counts. The simulator estimates this using the simulated counts
before sky subtraction, then divides by the exposure time to return to
electrons per second per pixel.

The noise map therefore includes contributions from both the source and sky,
even though the mean sky level has been removed from the data.

Look for larger noise values in the bright arcs, where more photons are
collected, and a background noise level elsewhere. More collected photons
produce larger absolute count fluctuations, but smaller fluctuations relative
to their mean.

In the next tutorial, the noise map will set the scale against which we judge
differences between the model image and the data.
"""
aplt.plot_array(array=dataset.noise_map, title="Simulated Noise Map")

"""
The `signal_to_noise_map` compares each pixel's measured brightness with its
estimated noise level. It is calculated by dividing `data` by `noise_map`,
giving a dimensionless ratio.

For example, a value of 5 means that the pixel's measured brightness is five
times its estimated noise standard deviation. Look for high values along the
bright arcs and lower values where the source is faint.

PyAutoLens sets negative ratios to zero in this map. The negative values in
the background-subtracted data are still present in `dataset.data`; this
operation only affects the signal-to-noise map.

A high value indicates that a pixel stands out relative to its estimated
noise. A single threshold does not, by itself, establish that a feature is
real: its spatial structure and the properties of the noise also matter.
"""
aplt.plot_array(
    array=dataset.signal_to_noise_map,
    title="Signal-To-Noise Map",
)

"""
We can display the dataset's main components together using
`aplt.subplot_imaging_dataset`.

Compare the data, noise map and signal-to-noise map. The bright arcs have
larger absolute noise than the faint background, but they also have a stronger
signal relative to that noise.

The figure includes logarithmic views of the data and PSF to help show faint
structure. These change how the values are displayed, not the underlying data.

Any panels labelled "Over Sample Size" describe how finely the model is
evaluated within pixels. We can leave these numerical settings aside here.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

We will save the simulated imaging dataset in FITS format, which is widely
used in astronomy for images and related data.

The files contain:

- `data.fits`: the simulated image after subtraction of the mean sky level.
- `noise_map.fits`: the estimated noise standard deviation in each pixel.
- `psf.fits`: the PSF kernel used to blur the image.

They are written to `dataset/imaging/howtolens/`, relative to the repository
root from which this script is run. The next tutorial will load these files
and compare the data with a lens model.

With `overwrite=True`, rerunning this cell replaces any existing files with
these names.
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

In Tutorial 4, we predicted the positions of a point source's multiple images.
For an observed lens, we can measure these positions from the telescope image,
accounting for the PSF and light from the foreground galaxy. Each measured
position has an uncertainty.

A `PointDataset` stores these measurements for analysis. It can also
include the images' fluxes and, for a variable source, the time delays between
them. Time delays are measured by monitoring how the source's variations
appear at different times in its multiple images.

The example below uses values rounded from Tutorial 4's predictions, with
assumed measurement uncertainties. These are illustrative data, not
measurements from a real observation.

Positions are in arcseconds, fluxes are in arbitrary units and time delays
are in days relative to the first image. We assume positional uncertainties
of 0.01 arcseconds, flux uncertainties of about 10 per cent and time-delay
uncertainties of one day.
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
Inspect the printed dataset summary. Positions use `(y, x)` coordinates, and
the position, flux and time-delay lists all follow the same image order.

The first image defines the zero of the time delays. A negative delay means
that the corresponding image arrives before this reference image. The zero
entry is a reference choice, not an independently measured delay.

The name `point_0` identifies the source associated with these measurements,
matching the point-source component introduced in Tutorial 4.

Fluxes and time delays are optional: a dataset can contain only positions
and their uncertainties. For further examples of working with these data,
see the workspace's
[point-source examples](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/point_source/start_here.py).
"""

"""
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
