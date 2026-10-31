"""
Tutorial 3: Borders
===================

__Contents__

- Why the extent of the source mesh matters
- What we mean by a border
- Starting without border relocation
- Including the central image pixels
- Following the central pixels into the source plane
- Tracing the outer border
- Switching border relocation on
- What relocation changes—and what it does not
- The mask still needs to cover the lensed source
- Wrap up
"""

"""
__Why the extent of the source mesh matters__

In the previous tutorials, we chose how many points our rectangular source mesh contained. However, the number of points is only part of what determines its resolution. We also need to consider how much of the source plane those points cover.

A `(25, 25)` mesh spread across a small source region can describe finer structure than the same mesh spread across a much larger region. If a few ray-traced coordinates extend far beyond the source, the mesh can expand to accommodate them, leaving fewer points where we actually need to reconstruct the galaxy.

This tutorial examines how that happens and introduces the border relocator, which provides a practical way to control these extreme coordinates.

We return to a simpler simulated source so that its morphology does not distract from the geometry. The image contains lensed source light but no foreground lens light.
"""

from autolens import jax_wrapper

from pathlib import Path
import subprocess
import sys

import numpy as np
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "simple__no_lens_light"

if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/simulator/no_lens_light.py"],
        check=True,
    )

dataset_unmasked = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

dataset_unmasked = dataset_unmasked.apply_over_sampling(
    over_sample_size_pixelization=1
)

aplt.subplot_imaging_dataset(dataset=dataset_unmasked)

"""
We will compare two masks with the same outer radius. One includes the centre of the image; the other excludes it. This lets us see what happens when the central image pixels enter the reconstruction.

__What we mean by a border__

The border is the outermost boundary of the unmasked image region.

For a circular mask, this is straightforward: the border follows the outside of the circle. An annular mask has an inner and an outer edge, but the border used here follows the outer edge. The hole in the middle does not become a second outer border.

Let’s plot both masks and mark their border coordinates.
"""

mask_circular = al.Mask2D.circular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    radius=2.5,
)

mask_annular = al.Mask2D.circular_annular(
    shape_native=dataset_unmasked.shape_native,
    pixel_scales=dataset_unmasked.pixel_scales,
    inner_radius=0.8,
    outer_radius=2.5,
)

dataset_circular = dataset_unmasked.apply_mask(mask=mask_circular)
dataset_annular = dataset_unmasked.apply_mask(mask=mask_annular)

aplt.plot_array(
    array=dataset_circular.data,
    positions=mask_circular.derive_grid.border,
    title="Circular Mask: Outer Border",
)

aplt.plot_array(
    array=dataset_annular.data,
    positions=mask_annular.derive_grid.border,
    title="Annular Mask: Outer Border",
)

"""
The outer borders should coincide because the masks share the same outer radius. The difference is inside: the circular mask includes image pixels near the lens centre that the annular mask removes.

We will keep the masks fixed while comparing the source-plane grids. Choosing the mask and relocating a border are different operations: the mask selects which image pixels enter the fit, while relocation changes how certain coordinates are treated when constructing the source reconstruction.

__Starting without border relocation__

First, let’s reconstruct the source using the annular mask with border relocation explicitly switched off.

We use the mass model that generated the simulation and a uniform rectangular source mesh. The helper below lets us change the mask and relocation setting while keeping the lens model, mesh shape and regularization coefficient unchanged.
"""

# Match mesh size to capped smoke data; normal tutorial runs retain 25 x 25.
mesh_shape = (8, 8) if dataset_unmasked.shape_native == al.util.dataset.SMALL_DATASETS_SHAPE_NATIVE else (25, 25)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(
            axis_ratio=0.9,
            angle=45.0,
        ),
    ),
)

field = al.MassField(
    redshift=0.5,
    shear=al.mp.ExternalShear(
        gamma_1=0.05,
        gamma_2=0.05,
    ),
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

tracer = al.Tracer(
    galaxies=[lens_galaxy, source_galaxy],
    fields=[field],
)


def fit_with_mask_and_border(mask, use_border_relocator):
    dataset = dataset_unmasked.apply_mask(mask=mask)

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(
            use_border_relocator=use_border_relocator,
        ),
    )


fit_annular = fit_with_mask_and_border(
    mask=mask_annular,
    use_border_relocator=False,
)

mapper_annular = fit_annular.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_grid(
    grid=mapper_annular.source_plane_data_grid,
    title="Annular Mask: Ray-Traced Image Coordinates",
)

aaplt.plot_mapper(
    mapper=mapper_annular,
    solution_vector=fit_annular.inversion.reconstruction,
    mesh_grid=mapper_annular.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Annular Mask: Source Reconstruction",
)

"""
The first plot shows where the image coordinates land after ray tracing. The second shows the source mesh and the brightnesses inferred on it.

Look at the coordinate ranges as well as the source itself. These tell us how much source-plane area the mesh is covering. We will use this reconstruction as a reference when we add the central image pixels.

__Including the central image pixels__

Now repeat the calculation with the circular mask, still without border relocation.

The outer image boundary is unchanged. We are adding the pixels within the central hole of the annulus, rather than extending the image to larger radii.
"""

fit_circular = fit_with_mask_and_border(
    mask=mask_circular,
    use_border_relocator=False,
)

mapper_circular = fit_circular.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_grid(
    grid=mapper_circular.source_plane_data_grid,
    title="Circular Mask: Ray-Traced Image Coordinates",
)

aaplt.plot_mapper(
    mapper=mapper_circular,
    solution_vector=fit_circular.inversion.reconstruction,
    mesh_grid=mapper_circular.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Circular Mask: Source Reconstruction",
)

for label, mapper in [
    ("Annular", mapper_annular),
    ("Circular", mapper_circular),
]:
    coordinates = np.asarray(mapper.source_plane_mesh_grid)
    span_y, span_x = np.ptp(coordinates, axis=0)

    print(
        f"{label} mesh coordinate spans: "
        f"y={span_y:.3f} arcsec, x={span_x:.3f} arcsec"
    )

"""
Compare the source-plane extent with the annular case. The added image pixels can trace outside the compact region occupied by the other coordinates, causing the rectangular mesh to cover a larger area.

Both reconstructions still use the same number of mesh points. If the mesh expands, their spacing increases, reducing the resolution available to describe the central source.

This illustrates why increasing the number of image pixels does not automatically improve a reconstruction with a fixed mesh. We must also consider how those pixels affect the source-plane coverage.

The two masks contain different data, so this is a geometric comparison—not a direct evidence comparison between otherwise identical fits.

__Following the central pixels into the source plane__

Let’s confirm where the extra source-plane coordinates come from.

We select two small groups of image pixels: one close to the lens centre and another near the lensed arc. As in tutorial 1, we use the same index groups in both planes, so matching colours identify the same light paths.
"""

image_coordinates = np.asarray(
    mapper_circular.image_plane_data_grid
)

central_indexes = np.argsort(
    np.sum(image_coordinates**2, axis=1)
)[:12].tolist()

arc_indexes = np.argsort(
    np.sum(
        (image_coordinates - np.array([0.0, 1.6])) ** 2,
        axis=1,
    )
)[:12].tolist()

index_groups = [central_indexes, arc_indexes]

aaplt.plot_grid(
    grid=mapper_circular.image_plane_data_grid,
    indexes=index_groups,
    title="Image Coordinates: Central Pixels and Arc Pixels",
)

aaplt.plot_grid(
    grid=mapper_circular.source_plane_data_grid,
    indexes=index_groups,
    title="The Same Pixels Traced into the Source Plane",
)

for index in central_indexes[:4]:
    print(
        f"Image index {index}: "
        f"{image_coordinates[index]} -> "
        f"{np.asarray(mapper_circular.source_plane_data_grid[index])}"
    )

"""
The central image coordinates need not map close to the source-plane origin. Their source positions depend on the difference between their image positions and the lens deflections.

Near the centre of a cuspy mass distribution, a small image region can correspond to a much larger source region. This is demagnification: the image occupies less area than the corresponding source region. It does not require the deflection angle itself to become arbitrarily large.

For our reconstruction, the practical consequence is that central image pixels can map well outside the region traced by the arcs. The mesh then spends some of its resolution covering those distant positions.

There is another concern. Source structure reconstructed at those positions may predict other images outside our chosen mask, where we are not comparing the model with the data. The reconstruction is then missing some of the observational constraints that would otherwise act on that structure.

In the next blocks, we will trace the outer border into the source plane and examine how border relocation handles coordinates beyond it. We will compare the same circularly masked dataset with relocation switched off and on, so that the effect of relocation is isolated.

**🤖 Ask AI:** Why can a small group of central image pixels change the resolution of a rectangular source mesh, even when most of the observed source light lies in the arcs?

__Tracing the outer border__

We have seen that some central image pixels trace beyond the compact source region. To decide how to handle those coordinates, we first need a reference boundary in the source plane.

We obtain this from the outer border of the image mask. These are ordinary image coordinates, so we can ray-trace them through the same lens model as every other pixel. Their traced positions describe where the edge of our fitted image region maps into the source plane.

Let’s highlight the border pixels in the image and follow the same indexes into the source plane. Border relocation is still switched off here, so we are looking at the original ray-traced coordinates.
"""

border_indexes = mask_circular.derive_indexes.border_slim.tolist()

aaplt.plot_grid(
    grid=mapper_circular.image_plane_data_grid,
    indexes=[border_indexes],
    title="Image-Plane Coordinates: Outer Border Highlighted",
)

aaplt.plot_grid(
    grid=mapper_circular.source_plane_data_grid,
    indexes=[border_indexes],
    title="Source-Plane Coordinates: Traced Outer Border Highlighted",
)

"""
Compare the highlighted border with the central image pixels traced in the previous block. Some image coordinates that were well inside the image mask can land beyond the traced outer border in the source plane.

This is the important geometric distinction: being inside the image-plane border does not guarantee that a coordinate remains inside its traced counterpart.

The border relocator uses the traced outer-border coordinates to estimate a source-plane boundary. In the current implementation, this is an ellipse. Coordinates beyond that boundary are moved inward for the source reconstruction, preventing them from extending its coverage unchecked.

__Switching border relocation on__

Let’s repeat the circular-mask fit with border relocation enabled.

The data, mask, lens mass, mesh shape and regularization coefficient all remain unchanged. This time, the source mesh is constructed using the relocated coordinates.
"""

fit_relocated = fit_with_mask_and_border(
    mask=mask_circular,
    use_border_relocator=True,
)

mapper_relocated = fit_relocated.inversion.cls_list_from(al.Mapper)[0]

for label, mapper in [
    ("Without Relocation", mapper_circular),
    ("With Relocation", mapper_relocated),
]:
    aaplt.plot_grid(
        grid=mapper.source_plane_data_grid,
        indexes=index_groups,
        title=f"{label}: Selected Source-Plane Coordinates",
    )

    mesh_coordinates = np.asarray(mapper.source_plane_mesh_grid)
    span_y, span_x = np.ptp(mesh_coordinates, axis=0)

    print(
        f"{label}: mesh spans "
        f"y={span_y:.3f} arcsec, x={span_x:.3f} arcsec"
    )

"""
Follow the same coloured coordinate groups through the two plots. Coordinates outside the estimated boundary are relocated towards it; those inside remain unchanged.

If relocation reduces the mesh extent, the same number of source points now covers a smaller area. This can restore resolution in the region where the source is being reconstructed, without increasing the mesh dimensions.

We can also identify which coordinates moved and display their positions before and after relocation.
"""

coordinates_before = np.asarray(
    mapper_circular.source_plane_data_grid
)
coordinates_after = np.asarray(
    mapper_relocated.source_plane_data_grid
)

displacements = np.linalg.norm(
    coordinates_after - coordinates_before,
    axis=1,
)
moved_indexes = np.flatnonzero(displacements > 1.0e-8).tolist()

print("Number of relocated image-pixel coordinates:", len(moved_indexes))

if moved_indexes:
    aaplt.plot_grid(
        grid=mapper_circular.source_plane_data_grid,
        indexes=[moved_indexes],
        title="Relocated Coordinates: Original Positions",
    )

    aaplt.plot_grid(
        grid=mapper_relocated.source_plane_data_grid,
        indexes=[moved_indexes],
        title="Relocated Coordinates: Positions Used by the Mapper",
    )

"""
These are still the same image pixels with the same measured brightnesses. Relocation changes their source-plane mapping coordinates; it does not remove them from the fit or move their positions in the observed image.

__What relocation changes—and what it does not__

Border relocation is a numerical treatment of the source reconstruction. It does not change the lens mass profile or the physical deflection calculation.

That distinction matters. If we ray-trace the image grid directly through the tracer, we recover the original source coordinates. The mapper used by the relocated inversion instead contains the coordinates after the border treatment.

Let’s compare these explicitly.
"""

physical_source_grid = tracer.traced_grid_2d_list_from(
    grid=dataset_circular.grids.pixelization
)[1]

physical_coordinates = np.asarray(physical_source_grid)

print(
    "Largest difference between direct ray tracing and the "
    "mapper without relocation:",
    np.max(
        np.linalg.norm(
            physical_coordinates - coordinates_before,
            axis=1,
        )
    ),
)

if moved_indexes:
    index = moved_indexes[0]

    print("Image-pixel index:", index)
    print("Directly ray-traced position:", physical_coordinates[index])
    print("Position used after relocation:", coordinates_after[index])

"""
We should therefore understand relocation as a practical boundary treatment, not an alternative physical lens equation. It controls how extreme coordinates participate in a reconstruction whose image coverage is finite.

The next step is to inspect its effect on the inferred source and the model image. We use a common source colour scale so that changes in brightness are not hidden by separately scaled figures.
"""

source_vmax = max(
    float(np.max(np.asarray(current_fit.inversion.reconstruction)))
    for current_fit in [fit_circular, fit_relocated]
)

for label, current_fit in [
    ("Without Relocation", fit_circular),
    ("With Relocation", fit_relocated),
]:
    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=f"{label}: Source Reconstruction",
    )

    aplt.plot_array(
        array=(
            dataset_circular.data - current_fit.model_data
        ) / dataset_circular.noise_map,
        title=f"{label}: Normalized Residuals",
    )

"""
Look for changes in the mesh coverage, the source structure and the residuals. A more compact mesh alone does not prove that the model is accurate, and relocation does not guarantee freedom from systematic errors.

We have also kept the regularization coefficient fixed to isolate the change in configuration. As tutorial 2 showed, changing mesh geometry can alter how regularization acts, so this is a diagnostic comparison rather than a complete optimization of both reconstructions.

__The mask still needs to cover the lensed source__

Border relocation does not replace choosing an appropriate image mask.

If the mask excludes part of an arc or another image of the source, those pixels do not constrain the reconstruction. Relocating source coordinates cannot recover information that we have excluded from the fit.

The mask also determines the border from which the source-plane boundary is estimated. Changing its size therefore changes both the image pixels being fitted and the boundary treatment.

Let’s inspect several circular masks before fitting them. We plot their borders on the full image so that any source light outside the selected region remains visible.
"""

mask_radii = [2.0, 2.5, 3.0]
masks_by_radius = {}

for radius in mask_radii:
    mask = al.Mask2D.circular(
        shape_native=dataset_unmasked.shape_native,
        pixel_scales=dataset_unmasked.pixel_scales,
        radius=radius,
    )
    masks_by_radius[radius] = mask

    aplt.plot_array(
        array=dataset_unmasked.data,
        positions=mask.derive_grid.border,
        title=f"Full Image with Mask Border at {radius:.1f} Arcsec",
    )

"""
Check whether each border encloses the visible lensed emission with some surrounding background. A mask that follows only the brightest parts can exclude faint extensions that still matter to the reconstruction.

Now compare the resulting fits with relocation enabled.
"""

fits_by_radius = {}

for radius, mask in masks_by_radius.items():
    current_fit = fit_with_mask_and_border(
        mask=mask,
        use_border_relocator=True,
    )
    fits_by_radius[radius] = current_fit

    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        zoom_to_brightest=False,
        title=f"Source Reconstruction: Mask Radius {radius:.1f} Arcsec",
    )

    aplt.subplot_fit_imaging(fit=current_fit)

"""
Compare the source structure and residuals, especially where a smaller mask approaches the arcs. If the reconstruction changes substantially as clearly relevant source light enters the mask, the smaller-mask result deserves closer inspection.

Do not rank these masks by their raw evidence values: they contain different sets of image pixels. The purpose here is to examine coverage and stability, not to select whichever mask produces the largest number.

**🤖 Ask AI:** Why can a reconstruction appear satisfactory inside a small mask while failing to explain other visible parts of the same lensed source?

__Wrap up__

We began with a rectangular source mesh whose resolution depended on both its number of points and the area they covered. A small number of extreme ray-traced coordinates could expand that area and reduce the resolution available for the source.

The border relocator uses the traced outer image border to estimate a source-plane boundary, then moves coordinates beyond that boundary inward for the reconstruction. The observed image pixels remain in the fit, and the physical lens model is unchanged.

In routine modelling, we normally leave this treatment enabled. Knowing what it does is useful when a source mesh extends unexpectedly far from the reconstructed galaxy, or when a reconstruction changes strongly with the mask.

These issues can also occur in systems with several lens galaxies. Extreme coordinates need not originate only near the centre of the image; they can be associated with other mass concentrations. The same diagnostic applies: follow the image coordinates into the source plane and inspect the boundary and mesh coverage.

For subsequent inspection, return to our circular mask with radius `2.5` arcsec and relocation enabled.
"""

dataset = dataset_circular
fit = fit_relocated
mapper = mapper_relocated

aplt.subplot_fit_imaging(fit=fit)

aaplt.plot_mapper(
    mapper=mapper,
    solution_vector=fit.inversion.reconstruction,
    mesh_grid=mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Source Reconstruction with Border Relocation Enabled",
)

"""
We have now examined the source mesh, interpolation, regularization and border treatment while holding the lens mass fixed.

In the next tutorial, we let the mass model vary. We will see how a pixelized source participates in lens modelling, why its flexibility can admit misleading solutions, and how a sensible starting model helps the search find the relevant region of parameter space.
"""
