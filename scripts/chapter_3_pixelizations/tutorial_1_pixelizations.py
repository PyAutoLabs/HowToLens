"""
Tutorial 1: Pixelizations, Mappers and Inversions
=================================================

__Contents__

- Opening — Pixelized source reconstruction
- A complex source
- Why a simple source model falls short
- A grid for the source light
- Following the same image pixels into the source plane
- Where a traced image pixel meets the source mesh
- One source region, several image regions
- Reconstructing the source
- A first look at regularization
- From source indexes to reconstructed brightnesses
- Turning the reconstructed source back into an image
- Inspecting the reconstruction
- Changing the source resolution
- Wrap up
"""

"""
__Opening — Pixelized source reconstruction__

Look at a resolved lensed arc. Its broad shape may be smooth, but the light along it can contain bright knots, faint extensions and irregular structure. A model might reproduce the overall arc while leaving compact, coherent features in the residual image.

In Chapter 2, we introduced linear light profiles and multi-Gaussian expansions (MGEs). By combining Gaussians of different widths, an MGE can describe a much more flexible radial brightness distribution than a single Sérsic profile. Gaussian groups with different ellipticities and orientations can also describe changes in flattening and isophotal twists. Fixed widths and linearly solved intensities keep the number of non-linear parameters manageable.

That flexibility makes MGE a useful modelling tool, but consider the geometry of our worked example. Within each galaxy’s Gaussian group, the components shared a common centre and elliptical shape. Adding more Gaussians gives us finer control over how brightness changes around that centre. It does not give us separate centres for star-forming knots, off-centre clumps or several distinct source components.

We could add offset Gaussian groups. MGEs are not fundamentally restricted to one centre. The practical difficulty is deciding how many groups to use, where to place them and how to fit their geometry as the source becomes more complicated.

This is where the image residuals become useful. If an MGE captures the broad light distribution but leaves compact structure along the arcs, its source description may need more spatial freedom. Those residuals are not proof of source clumpiness: errors in the mass model, foreground light subtraction, telescope blurring or other aspects of the data treatment can also produce structure. We need to investigate their origin.

In this chapter, we introduce pixelized source reconstruction. Instead of prescribing a light-profile centre and shape for every feature, we describe the source brightness on a mesh. Different regions of that mesh can take different brightness values, allowing the reconstruction to represent more complicated source structure.

The connection to Chapter 2 is direct. There, we solved for Gaussian intensities linearly. Here, for a specified mass model and mesh, we solve for source brightness values using the same underlying advantage: they enter the predicted image linearly.

This additional freedom also allows the reconstruction to fit noise. We therefore introduce regularization—a prior that controls how freely neighbouring source brightnesses can vary. Pixelization is not assumption-free: the mass model, mesh geometry, resolution and regularization all affect the inferred source.

MGE remains valuable, especially for foreground lens light and initial source fits. Pixelization is the next tool when the source requires more spatial freedom. We will learn how to use these tools together, inspect their reconstructions and recognise when their assumptions need revisiting.

__A complex source__

We will use a simulated source with several light components at different positions, giving it an asymmetric appearance. The simulation provides both the lensed image and the original source, so we can see what the reconstruction is trying to recover.

The next figures show these two views. Look at the separate bright regions in the source and how lensing stretches and repeats their structure in the image.

For observed data, we would only have the lensed image. The source reconstruction would be an inference from that image and our lens model, not an independent observation of the unlensed galaxy.
"""

from autolens import jax_wrapper

from pathlib import Path
import subprocess
import sys

import numpy as np
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "source_complex"

if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/simulator/source_complex.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.1,
    outer_radius=3.2,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=1)

# Match mesh size to capped smoke data; normal tutorial runs retain 25 x 25.
mesh_shape = (8, 8) if dataset.shape_native == al.util.dataset.SMALL_DATASETS_SHAPE_NATIVE else (25, 25)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=(0.17647, 0.0),
    ),
)

aplt.subplot_imaging_dataset(dataset=dataset)

input_tracer = al.from_json(file_path=dataset_path / "tracer.json")
source_plot_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.03)
aplt.plot_array(
    array=input_tracer.galaxies[-1].image_2d_from(grid=source_plot_grid),
    title="Input Source: Separate Bright Components",
)

"""
__Why a simple source model falls short__

An MGE can describe flexible radial light distributions, as we saw in Chapter 2. Here we fix a Gaussian group to one common centre and elliptical shape to isolate the missing spatial freedom: adding widths at that centre cannot independently place all the bright regions in this source.

We will compare the simulated image with a model that uses one common-centre MGE group, keeping the lens mass fixed to the values used in the simulation. This is an illustrative fixed-geometry comparison, not an optimized MGE search. The fixed mass cannot compensate for missing source structure.

Inspect the residual map for coherent structure along the arcs. Such structure would show where this source model leaves light unexplained, rather than just the random differences expected from noise.

**🤖 Ask AI:** If I allowed the lens mass parameters to vary while using this overly simple source model, how could the missing source structure affect the inferred mass distribution?
"""

gaussians = [
    al.lp_linear.Gaussian(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), sigma=sigma)
    for sigma in np.logspace(-2, np.log10(2.0), 20)
]
analytic_source = al.Galaxy(
    redshift=1.0, bulge=al.lp_basis.Basis(profile_list=gaussians),
)
analytic_fit = al.FitImaging(
    dataset=dataset, tracer=al.Tracer(galaxies=[lens_galaxy, analytic_source]),
)
aplt.subplot_fit_imaging(fit=analytic_fit)

"""
Offset Gaussian groups could supply this spatial freedom too, at the cost of choosing and fitting their geometry. MGE remains useful for initial source fits and foreground light. Here we explore a mesh as the next tool for more complicated source structure.


__A grid for the source light__

A pixelization describes the source using brightness values distributed across a mesh. We infer these values from the image instead of requiring the entire source to follow a small set of analytic profiles.

We begin with `RectangularUniform`, which places mesh points at regular intervals in the source plane. Its `shape` specifies the number of points along each direction. The next plot shows where these points lie relative to the ray-traced image coordinates.

The mesh also defines how brightness is interpolated between its points. We will examine this more closely in the next tutorial. For now, the important distinction is that the mesh specifies where we represent the source; it does not yet tell us how bright the source is.



__Following the same image pixels into the source plane__

Every unmasked image pixel has an index and a position. Ray tracing changes its position, but preserves its index: image point 100 and traced source point 100 describe the same light path.

We will highlight two small groups of image pixels and follow them into the source plane. Matching colours identify matching indexes in the two plots. The source-plane points here are ray-traced image coordinates; they are not the points of our source mesh.

For this demonstration, we use one sample at each image-pixel centre, making the correspondence easy to follow. More accurate image integration can use several samples within each pixel.
"""

import numpy as np
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset = dataset.apply_over_sampling(over_sample_size_pixelization=1)

# Match the mass model used by the source_complex simulator.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=(0.17647, 0.0),
    ),
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)]
)

image_grid = dataset.grids.pixelization
source_grid = tracer.traced_grid_2d_list_from(grid=image_grid)[1]

# Select small groups near two opposite sides of the lens.
image_coordinates = np.asarray(image_grid)
indexes = [
    np.argsort(
        np.sum((image_coordinates - position) ** 2, axis=1)
    )[:5].tolist()
    for position in [(0.0, 1.6), (0.0, -1.6)]
]

aaplt.plot_grid(
    grid=image_grid,
    indexes=indexes,
    title="Selected Image-Pixel Coordinates",
)
aaplt.plot_grid(
    grid=source_grid,
    indexes=indexes,
    title="The Same Coordinates After Ray Tracing",
)

for group in indexes:
    for index in group:
        print(
            f"Image index {index}: {np.asarray(image_grid[index])} "
            f"-> source position {np.asarray(source_grid[index])}"
        )

"""
Compare the separation and shape of each group before and after ray tracing. Their indexes remain unchanged even when the lens strongly changes their arrangement.



__Where a traced image pixel meets the source mesh__

We now place a separate rectangular mesh in the source plane. Its indexes identify the brightness values we will reconstruct. An image-pixel index and a source-mesh index therefore refer to different things.

A traced image coordinate generally falls between mesh points. With bilinear interpolation, its brightness is calculated from the four surrounding mesh values, weighted according to its position. Falling within one rectangular cell does not mean taking its brightness from just one source point.

The next cell selects one image pixel and highlights the source mesh points contributing to it. We print their indexes and weights to make this relationship explicit.
"""

mesh = al.mesh.RectangularUniform(shape=mesh_shape)

interpolator = mesh.interpolator_from(
    source_plane_data_grid=source_grid,
    source_plane_mesh_grid=None,
)

mapper = al.Mapper(
    interpolator=interpolator,
    regularization=al.reg.Constant(coefficient=1.0),
)

image_index = indexes[0][0]

# One row describes how source mesh values contribute to this image pixel.
weights = np.asarray(mapper.mapping_matrix[image_index])
source_indexes = np.flatnonzero(np.abs(weights) > 1.0e-12)

aaplt.plot_grid(
    grid=mapper.source_plane_mesh_grid,
    indexes=[source_indexes.tolist()],
    title=f"Source Mesh Points Contributing to Image Pixel {image_index}",
)

print("Ray-traced position:", np.asarray(source_grid[image_index]))
for source_index in source_indexes:
    print(
        f"Source mesh index {source_index}: "
        f"position={np.asarray(mapper.source_plane_mesh_grid[source_index])}, "
        f"weight={weights[source_index]:.4f}"
    )

"""
The mapper stores these relationships for all image pixels. We do not need to construct or solve the mapping matrix by hand; inspecting one row simply exposes what the mapping means.



__One source region, several image regions__

We can also follow the mapping in the opposite direction. Select a source mesh point and ask which image pixels receive a contribution from it.

The figure below uses the same colour for a selected source region and its associated image regions. A small source region can contribute to widely separated parts of the image: this is how multiple images constrain a shared source reconstruction.

The coloured image regions show interpolation support. They are not a strict statement that every traced coordinate falls inside the coloured source cell.
"""

source_coordinates = np.asarray(mapper.source_plane_mesh_grid)

# Find the mesh point nearest the source-plane origin.
central_index = int(
    np.argmin(np.sum(source_coordinates**2, axis=1))
)

pix_indexes = [[central_index]]
mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    region_labels=[str(central_index)],
    mesh_grid=mapper.source_plane_mesh_grid,
)

# Now compare two nearby source points with a more distant point.
nearby_indexes = np.argsort(
    np.sum(
        (source_coordinates - source_coordinates[central_index]) ** 2,
        axis=1,
    )
)[:2].tolist()

offset_index = int(
    np.argmin(
        np.sum(
            (source_coordinates - np.array([0.3, 0.3])) ** 2,
            axis=1,
        )
    )
)

pix_indexes = [nearby_indexes, [offset_index]]
mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    region_labels=[str(group) for group in pix_indexes],
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
Try changing `(0.3, 0.3)` to another source position and follow how its image regions change. The number of coloured patches is not necessarily the physical image multiplicity: masking and pixel sampling can split or hide regions.



__Reconstructing the source__

We now have a mesh and a description of how its brightness values contribute to the observed image. The next step is to infer those values from the data. This is called an inversion.

For a fixed lens mass model, the source brightnesses enter the predicted image linearly. PyAutoLens uses this property to solve for them together, rather than asking a non-linear search to explore each brightness value individually.

The next cell evaluates a `FitImaging` using a pixelized source. We keep the lens mass fixed to the simulation values, so this calculation reconstructs the source without searching for a new mass model.



__A first look at regularization__

Giving the source many adjustable brightness values lets it represent complex structure, but also gives it freedom to fit noise. We therefore include regularization: a prior that favours smoother source reconstructions.

For this first reconstruction, we use `Constant` regularization with a fixed coefficient. It discourages large differences between neighbouring mesh values. The data can still support structure, but adding structure comes with a penalty.

We will keep this setting fixed for the rest of this tutorial. In tutorial 2, we will vary it and see how too little smoothing fits noise, while too much smoothing erases source detail.
"""

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)
source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)
fit = al.FitImaging(
    dataset=dataset, tracer=al.Tracer(galaxies=[lens_galaxy, source_galaxy]),
)
aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
__From source indexes to reconstructed brightnesses__

So far, the mapper tells us how source brightness would contribute to the image, but we have not determined those brightnesses.

The inversion combines these mappings with the observed image, its noise and telescope blurring. It solves for the source brightness values together, including the smoothness prior supplied by regularization.

In this example there is one mapper. Entry `j` of its reconstruction therefore gives the brightness associated with source mesh index `j`. We can print selected values and show their locations on the reconstructed source.
"""

inversion = al.Inversion(
    dataset=dataset,
    linear_obj_list=[mapper],
)

reconstruction = np.asarray(inversion.reconstruction)

for group in pix_indexes:
    for source_index in group:
        print(
            f"Source mesh index {source_index}: "
            f"reconstructed brightness={reconstruction[source_index]:.6g}"
        )

aaplt.plot_mapper(
    mapper=mapper,
    solution_vector=reconstruction,
    mesh_grid=mapper.source_plane_mesh_grid,
    regions=[mapping.source_contours for mapping in mappings],
    region_labels=[str(group) for group in pix_indexes],
    title="Reconstructed Source and Selected Mesh Indexes",
)

"""
These brightnesses are inferred jointly. We do not independently fit each coloured image region: shared source values must explain their contributions across the image, while regularization connects neighbouring source values.



__Turning the reconstructed source back into an image__

The source reconstruction and the reconstructed image are different objects. The source lives on our source-plane mesh. Mapping its brightness values through the lens produces an image-plane prediction.

The telescope then blurs that prediction. This blurred image is what we compare with the observed data. The inversion already accounts for that blurring when solving for the source; it is not an extra correction applied after an otherwise completed fit.

The following plots separate these stages.
"""

image_before_blurring = inversion.mapped_reconstructed_data
model_image = inversion.mapped_reconstructed_operated_data

residual_map = dataset.data - model_image
normalized_residual_map = residual_map / dataset.noise_map

aplt.plot_array(
    array=image_before_blurring,
    title="Reconstructed Lensed Image Before Telescope Blurring",
)
aplt.plot_array(
    array=model_image,
    title="Model Image Including Telescope Blurring",
)
aplt.plot_array(
    array=residual_map,
    title="Residuals: Data Minus Model",
)
aplt.plot_array(
    array=normalized_residual_map,
    title="Residuals Divided by the Noise",
)

"""
Compare the source reconstruction with these image-plane plots. A feature in the source can appear in several places in the lensed image, and telescope blurring can make those image features less distinct.

Inspect the residuals for coherent unexplained structure. A plausible-looking source alone does not establish that the model explains the data.

**🤖 Ask AI:** Why must the same reconstructed source brightnesses explain all of their associated image regions, rather than allowing each arc to have an independent source reconstruction?

__Inspecting the reconstruction__

We will plot the inferred source alongside the model image and residuals. These answer different questions.

The source reconstruction shows the brightness distribution inferred under our chosen lens model, mesh and regularization. The model image shows how that reconstruction appears after lensing and telescope blurring. The residuals show what remains unexplained in the observed image.

Compare these residuals with those from the simple source model earlier in the tutorial. Look particularly at the image regions associated with the source’s separate bright components. The comparison will show whether the extra source freedom helps explain those features.

Because this is a simulation, we can also compare the reconstruction with the input source. We should not expect an exact copy: finite resolution, noise and regularization limit the detail we can recover.
"""

aplt.subplot_fit_imaging(fit=analytic_fit)
aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
__Changing the source resolution__

The mesh resolution controls how finely we can represent the source. A coarse mesh may merge distinct features, while a finer mesh gives the reconstruction more freedom.

Repeat the reconstruction with mesh shapes of `(15, 15)`, `(25, 25)` and `(35, 35)`, keeping the lens mass and regularization coefficient fixed. Compare the source structure and image residuals in each case, then restore the starting shape of `(25, 25)`.

More mesh points do not necessarily reveal more reliable detail. The data must constrain that detail, and changing the mesh also changes how the regularization acts. This experiment explores the effect of resolution; choosing an appropriate reconstruction requires considering resolution and smoothing together.

**🤖 Ask AI:** When I increase the source mesh resolution, how can I distinguish newly resolved source structure from extra freedom being used to fit noise?
"""

for shape in ([(6, 6), (8, 8), (10, 10)] if mesh_shape == (8, 8) else [(15, 15), (25, 25), (35, 35)]):
    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=shape),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )
    resolution_fit = al.FitImaging(
        dataset=dataset, tracer=al.Tracer(galaxies=[lens_galaxy, source_galaxy]),
    )
    aplt.subplot_fit_imaging_of_planes(fit=resolution_fit)
    aplt.subplot_fit_imaging(fit=resolution_fit)

"""
__Wrap up__

We have reconstructed a complex source without prescribing the positions and shapes of its individual light components.

The mesh specifies where we represent the source. The mapper connects its brightness values to the observed image through the lens model. The inversion solves for those values, with regularization controlling the freedom to introduce structure.

Throughout this tutorial, the lens mass was fixed. Later in the chapter, we will vary the mass model and reconstruct the source for each proposed model, allowing the image to constrain both.

First, we need to understand how interpolation and regularization shape the reconstruction. Tutorial 2 explores these choices and introduces the Bayesian evidence used to compare different levels of smoothing.
"""
