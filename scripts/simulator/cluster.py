"""
Simulator: Cluster
==================

This script simulates a strong lens on the 'cluster' scale: 2 main lens galaxies (a brightest cluster
galaxy and a satellite), 10 lower-mass cluster member galaxies on a luminosity-mass scaling relation,
a cluster-scale dark matter halo not tied to any individual galaxy, and 2 multiply-imaged background
source galaxies at *different* redshifts (z = 1.0 and z = 2.0) — a genuine multi-plane lens.

It is used to illustrate cluster-scale lens modeling in the HowToLens lecture series.

Cluster modeling uses the *point source* API: rather than fitting the extended arc light of each
lensed source, only the image-plane positions of its multiple images are fitted. This script therefore
simulates point-source datasets (one per source) alongside CCD imaging, which in real datasets is used
to measure those positions and to visually confirm the lens configuration.

__Contents__

- **Dataset Paths:** The `dataset_path` describes where the simulated cluster dataset is output to.
- **Redshifts:** The lens redshift and the two distinct source redshifts of the multi-plane system.
- **Galaxy Centres:** The centres of the main lens galaxies, scaling members, halo and sources.
- **Grids:** The imaging grid (with adaptive over sampling) and a coarse visualization grid.
- **Main Lens Galaxies:** The 2 individually-modelled cluster members, each a `SersicSph` light and `dPIEMassSph` mass.
- **Scaling Member Galaxies:** 10 lower-mass members whose masses derive from a luminosity-mass scaling relation.
- **Host Dark Matter Halo:** A standalone `NFWMCRLudlowSph` halo with `mass_at_200 = 10^15.3`.
- **Source Galaxies:** The 2 multi-plane background sources, each a `SersicCore` light + a `Point` component.
- **Ray Tracing:** Combine all galaxies into a single multi-plane `Tracer`.
- **Point Solver:** Solve the lens equation for the image-plane multiple-image positions of each source.
- **Point Datasets:** Collect the per-source positions (with noise) into `PointDataset` objects.
- **Combined CSV:** Write all point datasets to a single hand-editable CSV.
- **Scaling Galaxies CSV:** Write the scaling-member centres and luminosities to `scaling_galaxies.csv`.
- **Model CSVs:** Write the truth model to `mass.csv` + `light.csv` + `point.csv` (the named-galaxy CSV API).
- **Tracer json:** Save the true `Tracer` in the dataset folder as a .json file.
- **Imaging:** Simulate CCD imaging of the cluster and output it to .fits files.
- **Visualize:** Output subplots of the point datasets, tracer and imaging to the dataset path.

__Model__

This script simulates a 'cluster-scale' strong lens where:

 - The 2 main lens galaxies have `SersicSph` light profiles and `dPIEMassSph` mass profiles.
 - The 10 scaling-tier member galaxies have `dPIEMassSph` masses set by a luminosity-mass scaling relation.
 - The cluster's dark matter halo is a standalone `NFWMCRLudlowSph` profile.
 - The 2 source galaxies have `SersicCore` light profiles and `Point` components at z = 1.0 and z = 2.0.

__Start Here Notebook__

If any code in this script is unclear, refer to the `autolens_workspace/*/cluster/simulator.ipynb` notebook.
"""

import importlib.util
import sys

if importlib.util.find_spec("jax") is None:
    print(
        "Skipping this simulator: it requires the `jax` package, which is not "
        "installed (install with `pip install autolens[optional]`)."
    )
    sys.exit(0)

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Dataset Paths__

The `dataset_type` describes the type of data being simulated and `dataset_name` gives it a
descriptive name. The cluster dataset is output to `dataset/cluster/simple`, which the
tutorial on cluster-scale lensing loads via the auto-simulation guard.
"""
dataset_type = "cluster"
dataset_name = "simple"
dataset_path = Path("dataset", dataset_type, dataset_name)

"""
__Redshifts__

All lens galaxies and the host dark matter halo sit at the lens redshift z = 0.5. The two sources sit
at *different* redshifts (z = 1.0 and z = 2.0), making this a genuine multi-plane lens: light from the
z = 2.0 source passes through the z = 1.0 plane on its way to us, so the `Tracer` must ray-trace
through every plane in redshift order.
"""
redshift_lens = 0.5
source_redshifts = [1.0, 2.0]

"""
__Galaxy Centres__

The host halo is anchored at the cluster centre (the origin); the two main galaxies are the central
brightest cluster galaxy (BCG) and a satellite offset to the upper-right. The 10 scaling-member
centres sit at radii of 5-15" from the centre — inside the strongly-lensed region of the host halo but
clear of the cores of the two main galaxies. The source centres are chosen so both sources land in the
strongly-lensed region and are multiply imaged.
"""
main_lens_centres = [
    (0.0, 0.0),  # BCG at cluster centre
    (10.0, 8.0),  # satellite member
]

scaling_galaxies_centres = [
    (5.5, -6.5),
    (-7.5, 3.0),
    (12.0, -5.0),
    (-4.0, -9.0),
    (3.0, 13.0),
    (-14.0, 4.0),
    (15.0, 9.0),
    (-9.0, -12.0),
    (8.5, 5.5),
    (-6.5, 11.0),
]

scaling_galaxies_luminosities = [
    0.40,
    0.32,
    0.25,
    0.20,
    0.16,
    0.13,
    0.10,
    0.08,
    0.06,
    0.05,
]

host_halo_centre = (0.0, 0.0)

source_centres = [
    (0.3, 0.5),
    (-0.8, 1.2),
]

"""
__Grids__

The typical Einstein radius of a 10^15 solar-mass halo is ~20-30" and the member galaxies span ~30"
across, so the field must be large (100" x 100") to capture the multiple images and arcs — far bigger
than the galaxy-scale fields of earlier simulators.

Two grids are used: a high-resolution `imaging_grid` for rendering the CCD image, with adaptive over
sampling around the centre of every cluster member, and a coarse `viz_grid` used only for the
visualization plots at the end of the script (which do not need full resolution).
"""
imaging_grid = al.Grid2D.uniform(
    shape_native=(500, 500),
    pixel_scales=0.2,
)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=imaging_grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres + scaling_galaxies_centres,
)

imaging_grid = imaging_grid.apply_over_sampling(over_sample_size=over_sample_size)

viz_grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.5)

"""
__Main Lens Galaxies__

The 2 individually-modelled cluster members. Each has a `SersicSph` light profile (used for the CCD
imaging — the light plays no role in point-source modeling) and a `dPIEMassSph` mass profile, the
standard truncated mass profile for cluster members. Its parameters are Lenstool's native ones:

 - `sigma` (km/s): the fiducial velocity dispersion, setting the lens strength.
 - `r_core` (arcsec): the core radius, fixed to 0 here (the standard vanishing-core convention).
 - `r_cut` (arcsec): the truncation radius, beyond which the density falls off rapidly — the tidal
   stripping physics of galaxies orbiting inside a cluster's shared potential.
"""
main_lens_dpie_params = [
    # (r_core, r_cut, sigma)  per galaxy — arcsec, arcsec, km/s
    (0.0, 20.0, 330.0),  # BCG — strongest
    (0.0, 12.0, 210.0),  # satellite
]

main_lens_sersic_params = [
    # (intensity, effective_radius, sersic_index)
    (1.5, 3.0, 4.0),  # BCG — bright and extended
    (0.8, 1.5, 3.5),  # satellite
]

main_lens_galaxies = []
for centre, (r_core, r_cut, sigma), (intensity, effective_radius, sersic_index) in zip(
    main_lens_centres, main_lens_dpie_params, main_lens_sersic_params
):
    bulge = al.lp.SersicSph(
        centre=centre,
        intensity=intensity,
        effective_radius=effective_radius,
        sersic_index=sersic_index,
    )
    mass = al.mp.dPIEMassSph(
        centre=centre,
        sigma=sigma,
        r_core=r_core,
        r_cut=r_cut,
        redshift_object=redshift_lens,
        redshift_source=max(source_redshifts),
    )
    main_lens_galaxies.append(al.Galaxy(redshift=redshift_lens, bulge=bulge, mass=mass))

"""
__Scaling Member Galaxies__

The 10 lower-mass cluster members are modelled collectively via a luminosity-mass scaling relation —
the convention used by essentially every published cluster strong-lensing analysis:

    sigma_i  = sigma_ref * (L_i / L_ref) ** 0.25
    r_cut_i  = r_cut_ref * (L_i / L_ref) ** 0.7
    r_core_i = 0

where `L_ref` is a fixed reference luminosity and `sigma_ref` is the fiducial velocity dispersion of a
galaxy at that reference luminosity. The truth value simulated here is `sigma_ref = 85.0` km/s — the
value the tutorial's model-fit seeks to recover with a single free parameter for the whole tier.
"""
scaling_sigma_ref_truth = 85.0
scaling_sigma_exponent = 0.25  # alpha (Faber-Jackson)
scaling_gamma = 0.2  # mass-to-light tilt, fixed
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7
reference_luminosity = 1.0
scaling_r_core = 0.0  # vanishing core — fixed, never scaled
scaling_r_cut_ref = 5.0

scaling_galaxies = []
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    bulge = al.lp.SersicSph(
        centre=centre,
        intensity=luminosity,
        effective_radius=0.8,
        sersic_index=3.0,
    )
    luminosity_ratio = luminosity / reference_luminosity
    mass = al.mp.dPIEMassSph(
        centre=centre,
        sigma=scaling_sigma_ref_truth * luminosity_ratio**scaling_sigma_exponent,
        r_core=scaling_r_core,
        r_cut=scaling_r_cut_ref * luminosity_ratio**scaling_rcut_exponent,
        redshift_object=redshift_lens,
        redshift_source=max(source_redshifts),
    )
    scaling_galaxies.append(al.Galaxy(redshift=redshift_lens, bulge=bulge, mass=mass))

"""
__Host Dark Matter Halo__

A standalone galaxy holding the cluster's dark matter halo. It has no light profile — it sits in the
tracer solely to contribute mass. `NFWMCRLudlowSph` is parameterised by the physical halo mass within
r_200 (`mass_at_200 = 10^15.3`, ~2 x 10^15 solar masses) and the redshifts; its concentration follows
from a concentration-mass relation. The `redshift_source` is anchored to the *furthest* source
(z = 2.0), the multi-plane convention used throughout (deflections are normalized to the final plane).
"""
host_halo = al.mp.NFWMCRLudlowSph(
    centre=host_halo_centre,
    mass_at_200=10**15.3,
    redshift_object=redshift_lens,
    redshift_source=max(source_redshifts),
)

host_halo_galaxy = al.Galaxy(redshift=redshift_lens, dark=host_halo)

"""
__Source Galaxies__

The 2 background sources at *different* redshifts. Each carries a `SersicCore` light profile (so the
lensed arcs appear in the CCD imaging) and a `Point` component whose multiple-image positions we solve
for below — those positions are the data the tutorial's model-fit uses.

The attribute name of each `Point` (`point_0`, `point_1`) is important: it is the name that pairs each
model component to its `PointDataset` during modeling.
"""
source_galaxies = []
for i, (centre, src_z) in enumerate(zip(source_centres, source_redshifts)):
    bulge = al.lp.SersicCore(
        centre=centre,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0 + 30.0 * i),
        intensity=2.0,
        effective_radius=0.3,
        sersic_index=1.0,
    )
    point = al.ps.Point(centre=centre)
    source_galaxies.append(
        al.Galaxy(redshift=src_z, bulge=bulge, **{f"point_{i}": point})
    )

"""
__Ray Tracing__

Combine the main lens galaxies, scaling members, host halo and sources into a single tracer. With
sources at distinct redshifts, the tracer automatically performs multi-plane ray tracing.
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies
    + scaling_galaxies
    + [host_halo_galaxy]
    + source_galaxies
)

"""
__Point Solver__

The `PointSolver` solves the lens equation for the image-plane positions of each point source, by
ray-tracing triangles from the image plane to the source plane and iteratively refining those that
contain the source centre. At cluster scale (13 lens galaxies, multi-plane ray tracing) this solve is
expensive, so we accelerate it with JAX: `use_jax=True` plus a `jax.jit` wrapper around the solve call
compiles the triangle-refinement kernel once and reuses it for both sources.

The solver's grid must span the full cluster field, because the multiple images sit at radii of
~20-30" from the cluster centre — far outside the arc-second scale grids of galaxy-scale lensing.

`plane_redshift` is per-source and must be passed: it tells the solver which source plane to solve
for. Without it, the solver defaults to the tracer's final plane, which would silently place the
z = 1.0 source's images as if it sat at z = 2.0.
"""
from autolens.jax import register_tracer_classes

register_tracer_classes(tracer)

solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(400, 400), pixel_scales=0.25),
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
    use_jax=True,
)


def jitted_solve_for(plane_redshift):
    @jax.jit
    def jitted_solve(tracer, source_plane_coordinate):
        return solver.solve(
            tracer=tracer,
            source_plane_coordinate=source_plane_coordinate,
            plane_redshift=plane_redshift,
        ).array

    return jitted_solve


positions_list = []
for i, (src_centre, src_z) in enumerate(zip(source_centres, source_redshifts)):
    coord = jnp.asarray(src_centre)
    jitted_solve = jitted_solve_for(float(src_z))
    raw = np.asarray(jitted_solve(tracer, coord))
    finite = ~(np.isinf(raw).any(axis=1) | np.isnan(raw).any(axis=1))
    positions_list.append(al.Grid2DIrregular(raw[finite]))

    print(f"point_{i} (z={src_z}): {int(finite.sum())} multiple images solved")

"""
__Point Datasets__

One `PointDataset` per source. The `name` (`point_0`, `point_1`) pairs each dataset with the matching
`Point` component in the lens model during modeling, and the `redshift` records each source's
redshift so the multi-plane geometry can be rebuilt when the dataset is loaded.

The position uncertainty is 0.005" (5 mas), the centroid precision achievable by PSF fitting on
space-telescope imaging — not the imaging pixel scale.
"""
position_noise = 0.005

dataset_list = []
for i, positions in enumerate(positions_list):
    dataset = al.PointDataset(
        name=f"point_{i}",
        positions=positions,
        positions_noise_map=position_noise,
        redshift=source_redshifts[i],
    )
    dataset_list.append(dataset)

for i, dataset in enumerate(dataset_list):
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / f"point_dataset_{i}.json",
    )

"""
__Combined CSV__

For cluster workflows with many sources, a single CSV with one row per observed multiple image —
grouped by source `name`, with `y`, `x`, `positions_noise` and `redshift` columns — is far easier to
edit in a spreadsheet than many per-source files. `al.output_to_csv` writes every dataset into one
file, which the tutorial loads back with `al.list_from_csv`.
"""
al.output_to_csv(
    datasets=dataset_list,
    file_path=dataset_path / "point_datasets.csv",
)

"""
__Scaling Galaxies CSV__

The scaling-tier members are written to `scaling_galaxies.csv` — one row per member carrying its
centre and luminosity (`y, x, luminosity`). This is the catalogue the tutorial's scaling relation
consumes: scaling a real cluster up to hundreds of members is a CSV-level edit, with the model's
free-parameter count unchanged.
"""
al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)

"""
__Model CSVs__

Write the truth model out as three family-level CSVs — `mass.csv`, `light.csv`, `point.csv` — keyed
by galaxy name. The tutorial loads these with `al.galaxy_models_from_csv` and composes them into
`af.Model` galaxies ready for the non-linear search. See `autolens_workspace/*/cluster/csv_api.py`
for the full schema walkthrough.
"""
mass_profiles = {
    **{f"lens_{i}": {"mass": g.mass} for i, g in enumerate(main_lens_galaxies)},
    "host_halo": {"dark": host_halo_galaxy.dark},
}

light_profiles = {
    **{f"lens_{i}": {"bulge": g.bulge} for i, g in enumerate(main_lens_galaxies)},
    **{f"source_{i}": {"bulge": g.bulge} for i, g in enumerate(source_galaxies)},
}

point_profiles = {
    f"source_{i}": {f"point_{i}": getattr(g, f"point_{i}")}
    for i, g in enumerate(source_galaxies)
}

redshifts_by_galaxy = {
    **{f"lens_{i}": redshift_lens for i in range(len(main_lens_galaxies))},
    "host_halo": redshift_lens,
    **{f"source_{i}": z for i, z in enumerate(source_redshifts)},
}

al.galaxy_models_to_csv(
    profiles_by_galaxy=mass_profiles,
    file_path=dataset_path / "mass.csv",
    family="mass",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=light_profiles,
    file_path=dataset_path / "light.csv",
    family="light",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=point_profiles,
    file_path=dataset_path / "point.csv",
    family="point",
    redshifts=redshifts_by_galaxy,
)

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass
profiles and galaxies are safely stored and available to check how the dataset was simulated in the
future.

This can be loaded via the method `tracer = al.from_json()`.
"""
al.output_to_json(
    obj=tracer,
    file_path=Path(dataset_path, "tracer.json"),
)

"""
__Imaging__

Strong lens clusters come with imaging data, used to *measure* the point positions and to visually
confirm the lens configuration. Although modeling is point-source only, we simulate CCD imaging so
the dataset looks like a realistic cluster observation.
"""
psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.2, pixel_scales=imaging_grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=imaging_grid)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)

"""
__Visualize__

Output .png plots of the per-source point datasets, the tracer, and the simulated imaging to the
dataset path.
"""
for pd in dataset_list:
    aplt.subplot_point_dataset(
        dataset=pd, output_path=dataset_path, output_format="png"
    )

aplt.subplot_imaging_dataset(dataset=dataset)
aplt.plot_array(array=dataset.data, title="Data")

aplt.subplot_tracer(
    tracer=tracer, grid=viz_grid, output_path=dataset_path, output_format="png"
)
aplt.subplot_galaxies_images(
    tracer=tracer, grid=viz_grid, output_path=dataset_path, output_format="png"
)

"""
The dataset can be viewed in the folder `dataset/cluster/simple`.
"""
