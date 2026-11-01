"""
Tutorial 8: Adaptive Pixelization
=================================

In this tutorial we will introduce a new `Pixelization` object, which uses an `Overlay` image-mesh and a `Delaunay`
mesh.

This pixelization does not use a rectangular mesh, but instead uses a `Delaunay` triangulation.

So, why would we want to do that? Lets take another quick look at the rectangular grid.

__Contents__

- **Initial Setup:** We'll use the same strong lensing data as the previous tutorial, where.
- **Advantages and Disadvantages:** Lets think about the rectangular pixelization.
- **Image Mesh:** The Delaunay mesh is an irregular grid of pixels (or triangles) in the source-plane.
- **Regularization:** Penalize differences between neighbouring reconstructed values.
- **Source Compactness:** Reconstruct three sources with the same lens, mesh and noise.
- **Wrap Up:** Summary of the script and next steps.

"""

import importlib.util
import sys

if importlib.util.find_spec("jax") is None:
    print(
        "Skipping this tutorial: it requires the `jax` package, which is not "
        "installed (install with `pip install autolens[optional]`)."
    )
    sys.exit(0)

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import numpy as np
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

We'll use the same strong lensing data as the previous tutorial, where:

 - The lens galaxy's light is omitted.
 - The lens system uses an `Isothermal` galaxy mass profile and a separate `ExternalShear` field.
 - The source galaxy's light is an `Sersic`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding simulator
script. This ensures that all example scripts can be run without manually simulating data first.
"""
al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/no_lens_light.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
The lines of code below do everything we're used to, that is, setup an image, mask it, trace it via a tracer, setup
the rectangular mapper, etc.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)
field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
)

dataset = dataset.apply_mask(mask=mask)

mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)


pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=0.5),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
__Advantages and Disadvantages__

The rectangular mesh is a useful baseline, but its finite resolution can limit a reconstruction.
`RectangularBilinearAdaptDensity` adjusts rectangular sampling to the traced data density; it is not simply a
uniform source-plane grid. Inspect where its mesh points lie relative to the bright source.
"""
mapper = fit.inversion.cls_list_from(al.Mapper)[0]
rectangular_pixels = mapper.pixels
print(f"Rectangular source pixels: {rectangular_pixels}")
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
A mesh point in a faint region is not necessarily wasted: those measurements can constrain the source's extent. The
evidence balances fit quality, regularization and complexity, rather than simply rewarding fewer pixels. An
image-plane overlay offers another allocation: ray-tracing its points concentrates source-plane sampling according
to lens magnification. Whether this improves a fit must be checked for the data and regularization used.

__Image Mesh__

The Delaunay mesh is an irregular triangulation in the source-plane. First determine a set of (y,x) source-plane
coordinates defining this grid, specifically where each triangle vertex is located.

We do this using an `image_mesh`, which defines a method to determine a set of coordinates in the image-plane which
are ray-traced to the source-plane. These traced coordinates are the vertices of our source-pixel mesh.

Below, we use the `Overlay` image-mesh to do this, which overlays a grid of (y,x) coordinates over the image-plane
mask and retains all (Y,x) coordinates which fall within this mask.
"""
dataset = dataset.apply_sparse_operator()

image_mesh = al.image_mesh.Overlay(shape=(20, 20))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

"""
We can plot this grid over the image, to see that it is a coarse grid of (y,x) coordinates laid over the image.
"""

aplt.plot_array(
    array=dataset.data, grid=image_plane_mesh_grid, title="Image-plane mesh overlay"
)

"""
Give the source galaxy a `Pixelization` containing a `Delaunay` mesh. The fit receives the image-plane mesh
coordinates separately through `AdaptImages` below, and ray-traces them to construct the source-plane mesh.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(
        pixels=image_plane_mesh_grid.shape[0],
    ),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

"""
By using this tracer in a fit, we see that our source-plane no longer uses rectangular pixels, but a Delaunay mesh!
"""
adapt_images = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid},
)

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

mapper = fit.inversion.cls_list_from(al.Mapper)[0]


aplt.subplot_fit_imaging(fit=fit)

"""
Compare the residuals and reconstructed source with the rectangular example. The overlay retains only points inside
the mask, so its actual number of source pixels is smaller than the full 20 by 20 overlay. The two fits also use
different regularization coefficients; this is not a controlled ranking of mesh quality.
"""
print(f"Delaunay source pixels: {mapper.pixels}; rectangular: {rectangular_pixels}")
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
__Regularization__

Constant regularization penalizes differences between neighbouring reconstructed source values. Rectangular meshes
use their grid connectivity; Delaunay meshes use the connected vertices of the triangulation. A vertex need not
have exactly three neighbours. One coefficient controls the strength of smoothing across the mesh.

__Source Compactness__

Magnification depends on the lens and source position, not on the brightness distribution we place there. Keep this
lens, image-plane overlay and regularization fixed while making three source profiles progressively more compact.
This restores a direct test of a limitation of magnification-based sampling.

Use identical PSF, mask, Gaussian noise map and noise realization for each simulated image. The profiles retain the
same intensity at their effective radius, so their total fluxes and signal-to-noise ratios also differ; this is a
diagnostic experiment, not an isolated measurement of the effect of radius. The coefficient of 1.0 is an explicit
illustrative choice, not an evidence optimum for any of these datasets.
"""

simulation_grid = al.Grid2D.uniform(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)
simulation_grid = simulation_grid.apply_over_sampling(over_sample_size=8)
simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=dataset.psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
)
noise_sigma = 0.05
noise_values = np.random.default_rng(1).normal(
    loc=0.0, scale=noise_sigma, size=simulation_grid.shape_native
)
common_noise_map = al.Array2D.no_mask(
    values=np.full(simulation_grid.shape_native, noise_sigma),
    pixel_scales=dataset.pixel_scales,
)

compactness_fits = {}
for label, effective_radius, sersic_index in [
    ("Extended", 0.5, 1.0),
    ("Compact", 0.2, 2.5),
    ("Very compact", 0.1, 4.0),
]:
    true_source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.15),
            intensity=0.2,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        ),
    )
    true_tracer = al.Tracer(galaxies=[lens_galaxy, true_source], fields=[field])
    noiseless = simulator.via_tracer_from(tracer=true_tracer, grid=simulation_grid)
    simulated_dataset = al.Imaging(
        data=al.Array2D.no_mask(
            values=np.asarray(noiseless.data.native) + noise_values,
            pixel_scales=dataset.pixel_scales,
        ),
        noise_map=common_noise_map,
        psf=dataset.psf,
    )
    simulated_dataset = simulated_dataset.apply_mask(mask=mask)
    simulated_dataset = simulated_dataset.apply_sparse_operator()
    compactness_fit = al.FitImaging(
        dataset=simulated_dataset, tracer=tracer, adapt_images=adapt_images
    )
    compactness_fits[label] = compactness_fit
    compactness_mapper = compactness_fit.inversion.cls_list_from(al.Mapper)[0]
    mesh_radii = np.sqrt(
        np.sum(np.asarray(compactness_mapper.source_plane_mesh_grid) ** 2, axis=1)
    )
    vertices_within_radius = np.count_nonzero(mesh_radii <= effective_radius)
    print(
        f"{label}: radius={effective_radius:.2f}, index={sersic_index:.1f}, "
        f"coefficient=1.0, mesh pixels={compactness_mapper.pixels}, "
        f"vertices within circular effective-radius aperture={vertices_within_radius}, "
        f"chi_squared / data pixels={compactness_fit.chi_squared / simulated_dataset.data.shape[0]:.3f}, "
        f"max |normalized residual|={np.max(np.abs(compactness_fit.normalized_residual_map)):.3f}"
    )
    aplt.subplot_fit_imaging(fit=compactness_fit)
    aplt.subplot_fit_imaging_of_planes(fit=compactness_fit)

"""
The printed aperture counts measure available mesh vertices, not the number of independently constrained source
values. Inspect the source reconstructions alongside the residual maps: can the same mesh resolve each bright
centre without coherent residuals? Changing the coefficient may trade smoothing of the centre against noise in the
outskirts, but cannot add mesh vertices. We have not searched for an optimal coefficient, and these separate
simulated datasets must not be ranked against one another using their absolute evidences.

__Wrap Up__

The overlay responds to magnification rather than source morphology. The compactness experiment lets us inspect
where that distinction matters; neither fewer pixels nor a more adaptive mesh guarantees a better reconstruction.
Tutorials 11 and 12 explore adapting the mesh and regularization to a fitted brightness distribution. They test
those choices on the same data so that changes in fit quality and evidence can be assessed meaningfully.

You can also experiment with rectangular and Delaunay meshes by changing the pixelization in a pipeline.
"""
