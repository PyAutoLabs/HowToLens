"""
Simulator: Lens With Extra Galaxy
=================================

This script simulates `Imaging` of a 'galaxy-scale' strong lens which is identical to the `lens_sersic` dataset
simulated for chapter 2 (lens light + mass + lensed source), but with one extra galaxy located a few arc-seconds
from the lens galaxy.

The extra galaxy has its own light profile, which blends into the outskirts of the image, and its own mass profile,
which contributes to the lensing of the source. Chapter 4's first tutorial uses this dataset to illustrate the two
approaches to dealing with such interloper galaxies: removing their light via noise scaling, or including them in
the lens model explicitly.

It is used to illustrate scaling up lens modeling in the HowToLens lecture series.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
- **Simulate:** Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
- **Ray Tracing:** Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.
- **Extra Galaxy:** Include one extra galaxy, whose light and mass must be masked or modeled in the tutorial.
- **Output:** Output the simulated dataset to the dataset path as .fits files.
- **Mask Extra Galaxy:** Build and save `mask_extra_galaxies.fits` so the tutorial can load it directly.
- **Visualize:** Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass.
- **Extra Galaxy Centre:** Output the centre of the extra galaxy to a .json file for use in the lens model.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light profile is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
 - There is one extra galaxy whose light is near the lens and whose mass perturbs the lensed source's emission.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/*/imaging/simulator.ipynb` notebook.
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
dataset_type = "imaging"
dataset_name = "lens_extra_galaxy"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.

This simulated lens has an extra galaxy offset from the main lens galaxy centre of (0.0", 0.0"). The adaptive over
sampling grid has both centres input to account for this.
"""
extra_galaxy_centre = (1.5, 2.5)

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0), extra_galaxy_centre],
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
Simulate a simple Gaussian PSF for the image.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
Create the simulator for the imaging data, which defines the exposure time, background sky, noise levels and psf.
"""
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Ray Tracing__

Setup the lens galaxy's light, mass and source galaxy light for this simulated lens.

These are identical to the `lens_sersic` dataset simulated for chapter 2, so that the only difference between the
two datasets is the extra galaxy included below.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
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
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
__Extra Galaxy__

Include one extra galaxy, which is offset a few arc-seconds from the lens galaxy.

Its light is an `ExponentialSph` profile, which blends into the image near the lensed source's emission, and its
mass is an `IsothermalSph` profile, which perturbs the ray-tracing of the source's light.

Note that its redshift is the same as the main lens galaxy, which is not necessarily the case in real observations.
If it were at a different redshift, the tools for masking or modeling the extra galaxy are equipped to handle this,
with mass modeling leading to multi-plane ray-tracing being performed.
"""
extra_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.ExponentialSph(
        centre=extra_galaxy_centre, intensity=2.0, effective_radius=0.5
    ),
    mass=al.mp.IsothermalSph(centre=extra_galaxy_centre, einstein_radius=0.15),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, extra_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.
"""
aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Mask Extra Galaxy__

Build and output a `mask_extra_galaxies.fits` covering the extra galaxy's region, so that the chapter 4 tutorial
which uses this dataset can load the mask directly without a separate data-preparation step.

The circle is sized to ~3x the galaxy's `effective_radius`, which comfortably covers the light extent of the
`ExponentialSph` profile used above. The geometry is derived from the same centre + radius defined for the extra
galaxy in this script, so it stays in sync with any future tweak to those values.

`Mask2D.circular` honours the `PYAUTO_SMALL_DATASETS=1` env var (caps to 16x16 at 0.6"/px), so the mask
automatically shrinks alongside the small-dataset image and never raises an out-of-bounds error.
"""
extra_galaxies_mask = np.zeros(dataset.shape_native, dtype=bool)

circle = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    centre=extra_galaxy_centre,
    radius=3.0 * 0.5,
    invert=True,  # True inside the circle (i.e. masked region)
)
extra_galaxies_mask = np.logical_or(extra_galaxies_mask, circle.native)

mask_extra_galaxies = al.Mask2D(
    mask=extra_galaxies_mask,
    pixel_scales=dataset.pixel_scales,
)

aplt.fits_array(
    array=mask_extra_galaxies,
    file_path=dataset_path / "mask_extra_galaxies.fits",
    overwrite=True,
)

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

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
__Extra Galaxy Centre__

Output the centre of the extra galaxy to a .json file, so that it can be used to set up the lens model in the
chapter 4 tutorial.
"""
al.output_to_json(
    obj=al.Grid2DIrregular(values=[extra_galaxy_centre]),
    file_path=dataset_path / "extra_galaxies_centres.json",
)

"""
The dataset can be viewed in the folder `autolens_workspace/dataset/imaging/lens_extra_galaxy`.
"""
