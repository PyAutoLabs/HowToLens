"""
Simulator: Interferometer
=========================

This script simulates `Interferometer` data of a 'galaxy-scale' strong lens, as would be observed by a radio or
sub-mm interferometer like ALMA or the JVLA.

Unlike CCD imaging, an interferometer does not observe an image of the lens. It measures "visibilities", which are
the Fourier transform of the sky brightness sampled at a set of points in the "uv-plane", where each point
corresponds to the separation of a pair of antennas in the array.

It is used to illustrate interferometer data in the HowToLens lecture series. HowToLens does not cover
interferometry beyond this glimpse; the `autolens_workspace/scripts/interferometer` package is the dedicated
resource for uv-plane lens modeling.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
- **Simulate:** Simulate the image using a (y,x) real-space grid and a synthetic set of uv-plane baselines.
- **Ray Tracing:** Setup the lens galaxy's mass and source galaxy light for this simulated lens.
- **Output:** Output the simulated dataset to the dataset path as .fits files.
- **Visualize:** Output a subplot of the simulated dataset's dirty images and the tracer's quantities.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass.

__Model__

This script simulates `Interferometer` data of a 'galaxy-scale' strong lens where:

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a `SersicCore`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/*/interferometer/simulator.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.
"""
dataset_type = "interferometer"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Simulate__

For interferometer data, the strong lens's image is evaluated in real space on a (y,x) grid and then Fourier
transformed to the uv-plane, where it is compared to the observed visibilities.

Interferometers do not observe galaxies in a way where over sampling is necessary, therefore the grid uses no
over sampling.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

"""
To perform the Fourier transform we need the wavelengths of the baselines, which describe where in the uv-plane
each visibility samples the sky's Fourier transform.

For real data these are determined by the layout of the interferometer's antennas. The `autolens_workspace`
bundles baselines of real instruments (e.g. the Square Mile Array (SMA) and ALMA). For this tutorial dataset we
instead draw a small synthetic set of baselines from a Gaussian distribution in the uv-plane, with a scale
comparable to the SMA's (a few hundred visibilities), keeping the simulation fast and self-contained.
"""
rng = np.random.default_rng(1)

total_visibilities = 200

uv_wavelengths = rng.normal(loc=0.0, scale=1.0e5, size=(total_visibilities, 2))

"""
To simulate the interferometer dataset we first create a simulator, which defines the exposure time, noise levels
and Fourier transform method used in the simulation.

We use the `TransformerDFT`, an exact Discrete Fourier Transform which is fast for datasets with a low number of
visibilities like this one.
"""
simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=1000.0,
    transformer_class=al.TransformerDFT,
)

"""
__Ray Tracing__

Setup the lens galaxy's mass and source galaxy light for this simulated lens.

The lens galaxy has no light profile, because at the radio and sub-mm wavelengths an interferometer observes the
foreground lens galaxy typically emits negligibly.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=10.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
interferometer dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated interferometer dataset's dirty images before outputting it to fits.
"""
aplt.subplot_interferometer_dirty_images(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_interferometer(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset's dirty images and the tracer's quantities to the dataset path as
.png files.
"""
aplt.subplot_interferometer_dirty_images(
    dataset=dataset, output_path=dataset_path, output_format="png"
)

aplt.subplot_tracer(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=grid, output_path=dataset_path, output_format="png"
)

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
The dataset can be viewed in the folder `dataset/interferometer/simple`.
"""
