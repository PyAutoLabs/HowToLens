"""
Simulator: Group Scale Lens
===========================

This script simulates `Imaging` of a 'group-scale' strong lens, which is used in chapter 4 of the **HowToLens**
lectures to illustrate lens modeling at the group scale.

The group consists of a single main lens galaxy -- the brightest group galaxy (BGG) -- and two smaller member
galaxies nearby, whose mass contributes significantly to the ray-tracing and is therefore included in the strong
lens model. A single source galaxy is lensed by the combined mass of all three galaxies.

__Model__

This script simulates `Imaging` of a 'group-scale' strong lens where:

 - The main lens galaxy's light is a `SersicSph` and its total mass distribution is an `IsothermalSph`.
 - The two member galaxies have `SersicSph` light profiles and tidally truncated `dPIEMassSph` mass profiles.
 - The source galaxy's light is a `SersicCore`.

The member galaxies are simulated to follow the light-mass scaling relation used in the group-scale modeling
tutorial: `sigma = sigma_ref * (L / L_ref) ** 0.25` and `r_cut = r_cut_ref * (L / L_ref) ** 0.7`, where `L` is
each member's luminosity. The second member is 2.07 times more luminous than the first, so its velocity
dispersion is 1.2 times higher (240 km/s vs 200 km/s) and its truncation radius 1.67 times larger.

__Contents__

- **Dataset Paths:** The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a.
- **Grid:** Define the 2D grid of (y,x) coordinates the simulated image is evaluated on.
- **Galaxy Centres:** Define the centres of the main lens galaxy and member galaxies.
- **Over Sampling:** Set up the adaptive over-sampling scheme, centred on every galaxy in the group.
- **PSF / Simulator:** The Point Spread Function and simulator defining the observational properties of the data.
- **Main Lens Galaxy:** The brightest group galaxy (BGG) at the origin (0.0", 0.0").
- **Member Galaxies:** The two smaller group members, with tidally truncated `dPIEMassSph` mass profiles.
- **Source Galaxy:** The source galaxy whose lensed image we simulate.
- **Ray Tracing:** Use all galaxies to set up a tracer, which generates the image that is simulated.
- **Output:** Output the simulated dataset to the dataset path as .fits files.
- **Visualize:** Output a subplot of the simulated dataset and the tracer's quantities to the dataset path.
- **Tracer json:** Save the `Tracer` in the dataset folder as a .json file.
- **Centre JSON Files:** Save the centres of the main lens galaxy and member galaxies as .json files.
- **Positions:** Solve for the multiple-image positions of the lensed source and save them as a .json file.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/*/group/simulator.ipynb` notebook.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a descriptive name.

The dataset is output to `dataset/group/simple`.
"""
dataset_type = "group"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Grid__

Define the 2D grid of (y,x) coordinates the simulated image is evaluated on.

A group-scale lens spans a wider area of sky than a galaxy-scale lens, because it contains multiple lens
galaxies which are separated by many arc-seconds. The grid is therefore larger (250 x 250 pixels, or 25.0")
than those used in previous chapters.
"""
grid = al.Grid2D.uniform(
    shape_native=(250, 250),
    pixel_scales=0.1,
)

"""
__Galaxy Centres__

Define the centres of the main lens galaxy and member galaxies. These are used for over-sampling and are also
output to .json files so that the modeling tutorial can load them.
"""
main_lens_centres = [(0.0, 0.0)]
member_centres = [(3.5, 2.5), (-4.4, -5.0)]

"""
__Over Sampling__

Over sampling evaluates the light profiles on a higher resolution grid in their bright central regions, to
ensure the calculation is accurate.

The adaptive over-sampling scheme below is applied at the centre of every galaxy in the group, not just the
main lens galaxy.
"""
over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + member_centres,
)

grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

"""
__PSF / Simulator__

Simulate a simple Gaussian PSF for the image, and create the simulator for the imaging data, which defines the
exposure time, background sky, noise levels and psf.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

"""
__Main Lens Galaxy__

The main lens galaxy is the brightest group galaxy (BGG), located at the origin (0.0", 0.0"). It has a spherical
Sersic light profile and an isothermal mass profile with a large Einstein radius of 4.0", reflecting that it
dominates the mass of the group.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=0.7, effective_radius=2.0, sersic_index=4.0
    ),
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=4.0),
)

"""
__Member Galaxies__

The two member galaxies are smaller galaxies orbiting within the group. They have spherical Sersic light
profiles and tidally truncated `dPIEMassSph` mass profiles (vanishing core `r_core = 0.0` and a finite
truncation radius `r_cut`), reflecting that their outer dark matter has been stripped by the tides of the
group's potential.

The members follow the light-mass scaling relation used in the modeling tutorial: the second member is 2.07
times more luminous than the first (`intensity` of 1.866 vs 0.9, with identical `effective_radius` and
`sersic_index`), so its `sigma` is a factor 2.07 ** 0.25 = 1.2 higher (240 km/s vs 200 km/s) and its `r_cut`
a factor 2.07 ** 0.7 = 1.67 larger (13.3" vs 8.0").

The `dPIEMassSph` profile is parameterized in Lenstool's native convention -- `sigma` (fiducial velocity
dispersion, km/s), `r_core` and `r_cut` (arcsec) -- and converts these to a lensing strength internally using
the object/source redshifts and cosmology (`H0` / `Om0`, which default to Planck-like values).
"""
member_galaxy_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(3.5, 2.5), intensity=0.9, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(3.5, 2.5),
        sigma=200.0,
        r_core=0.0,
        r_cut=8.0,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

member_galaxy_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(-4.4, -5.0), intensity=1.866, effective_radius=0.8, sersic_index=3.0
    ),
    mass=al.mp.dPIEMassSph(
        centre=(-4.4, -5.0),
        sigma=240.0,
        r_core=0.0,
        r_cut=13.3,
        redshift_object=0.5,
        redshift_source=1.0,
    ),
)

member_galaxies = [member_galaxy_0, member_galaxy_1]

"""
__Source Galaxy__

The source galaxy whose lensed image we simulate. It uses a cored Sersic profile, whose light changes gradually
in its central regions and can therefore be evaluated accurately without adaptive over-sampling.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=3.0,
        effective_radius=0.4,
        sersic_index=1.0,
    ),
)

"""
__Ray Tracing__

Use all galaxies to set up a tracer, which will generate the image for the simulated `Imaging` dataset.

The tracer combines the main lens galaxy, the member galaxies and the source galaxy.
"""
tracer = al.Tracer(galaxies=[lens_galaxy] + member_galaxies + [source_galaxy])

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

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and
galaxies are safely stored and available to check how the dataset was simulated in the future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Centre JSON Files__

Save the centres of the main lens galaxy and member galaxies as .json files. These are loaded by the group-scale
modeling tutorial to set up the lens model (e.g. fixing the centres of the member galaxies).
"""
al.output_to_json(
    obj=al.Grid2DIrregular(main_lens_centres),
    file_path=Path(dataset_path, "main_lens_centres.json"),
)

al.output_to_json(
    obj=al.Grid2DIrregular(member_centres),
    file_path=Path(dataset_path, "extra_galaxies_centres.json"),
)

"""
__Positions__

Solve for the multiple-image positions of the lensed source galaxy, which can be used as input to group-scale
modeling to help the non-linear search converge (and are the primary observable of cluster-scale modeling,
introduced in the next chapter 4 tutorial).
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1),
    pixel_scale_precision=0.001,
    magnification_threshold=0.01,
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.bulge.centre
)

al.output_to_json(
    obj=positions,
    file_path=dataset_path / "positions.json",
)

"""
The dataset can be viewed in the folder `dataset/group/simple`.
"""
