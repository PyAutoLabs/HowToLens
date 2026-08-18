"""
Simulator: Lens x3
==================

This script simulates `Imaging` of a 'galaxy-scale' lens where there are three lens galaxies, each with their own
light and mass profiles, which all contribute significantly to the lensing of a single background source.

Strong lenses with this complex mass distribution are more challenging to model than those with one or two lens
galaxies, because every additional deflector adds its own free light and mass parameters to the model.

A system with three co-dominant lens galaxies and no shared dark-matter halo is exactly what the
`autolens_workspace`'s `multi_galaxy` package is dedicated to — its `start_here` and `modeling` examples are
directly applicable to this lens. (Systems where a dominant group-scale halo joins the galaxies belong one rung
up the regime ladder, in the `group` package.)

This dataset is modeled in HowToLens chapter 4 and is used to illustrate how the parameter count of a lens model
grows with every co-dominant deflector, motivating the scaling relations introduced later in that chapter.

__Contents__

- **Model:** Compose the lens model fitted to the data.
- **Dataset Paths:** The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and.
- **Simulate:** Simulate the image using a (y,x) grid with the adaptive over sampling scheme.
- **Ray Tracing:** Setup the lens galaxies' light, mass and source galaxy light for this simulated lens.
- **Output:** Output the simulated dataset to the dataset path as .fits files.
- **Visualize:** Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass.

__Model__

This script simulates `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxies' light are three `Sersic`'s.
 - The lens galaxies' mass distributions are three `Isothermal`'s.
 - The source galaxy's light is an `ExponentialCoreSph`.

This dataset is used in chapter 4 of the **HowToLens** lectures.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/*/imaging/simulator.ipynb` notebook.
"""

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated (in this case, `Imaging` data) and `dataset_name`
gives it a descriptive name.
"""
dataset_type = "imaging"
dataset_name = "x3_lens_galaxies"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Simulate__

Simulate the image using a (y,x) grid with the adaptive over sampling scheme.

This simulated lens has three galaxies whose centres are offset from (0.0", 0.0"), forming a triangle around the
origin. The adaptive over sampling grid has all three centres input to account for this.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.1,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.9, 0.0), (-0.6, -0.9), (-0.6, 0.9)],
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

Setup the lens galaxies' light, mass and source galaxy light for this simulated lens.

The image plane is made of three separate lens galaxies, whose Einstein radii (0.9", 0.8" and 0.7") are
deliberately comparable — no single galaxy dominates the deflection of the source's light, which is what makes
every one of them a co-dominant deflector that must be modeled individually.

All three galaxies are at the same redshift, so ray tracing is single-plane and their deflection fields simply add.
"""
lens_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.9, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=30.0),
        intensity=0.1,
        effective_radius=0.7,
        sersic_index=3.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.9, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=30.0),
        einstein_radius=0.9,
    ),
)

lens_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.6, -0.9),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=120.0),
        intensity=0.1,
        effective_radius=0.6,
        sersic_index=2.5,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.6, -0.9),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=120.0),
        einstein_radius=0.8,
    ),
)

lens_galaxy_2 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(-0.6, 0.9),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        intensity=0.1,
        effective_radius=0.5,
        sersic_index=3.5,
    ),
    mass=al.mp.Isothermal(
        centre=(-0.6, 0.9),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=60.0),
        einstein_radius=0.7,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialCoreSph(
        centre=(0.05, 0.15), intensity=0.2, effective_radius=0.5, radius_break=0.025
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(
    galaxies=[lens_galaxy_0, lens_galaxy_1, lens_galaxy_2, source_galaxy]
)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

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
The dataset can be viewed in the folder `autolens_workspace/dataset/imaging/x3_lens_galaxies`.
"""
