"""
Tutorial 3: Borders
===================

Explore how masks and border relocation affect the extent and resolution of a source mesh.

__Contents__

- Why the extent of the source mesh matters
- What we mean by a border
- Starting without border relocation
- Including the central image pixels
- Following the central pixels into the source plane
- Tracing the outer border
- Switching border relocation on
- What relocation changes
- The mask still needs to cover the lensed source
- Wrap Up

__Why the extent of the source mesh matters__

Imagine spreading a `(25, 25)` rectangular mesh across a source-plane region one arcsecond wide. Now spread the same mesh across a region ten arcseconds wide. We still have the same number of brightness values, but the spacing between them is ten times larger. Small source components that the first mesh could resolve may be poorly represented by the second.

What determines this extent?

The uniform rectangular mesh we are using is constructed to cover the ray-traced image coordinates. If some coordinates land far away from the source galaxy, the mesh expands to include them. Much of its resolution can then be spent covering regions where there is little or no source light.

This can happen when image pixels close to the centre of the lens are strongly demagnified and trace far out into the source plane. Adding these central image pixels to a fit can therefore have a surprisingly large effect on the source mesh.

In this tutorial, we will follow those pixels through the calculation and introduce **border relocation**, which prevents these distant coordinates from unnecessarily expanding the reconstruction.

We return to a simpler simulated source, with no foreground lens light, so that we can focus on the geometry.
"""

from autolens import jax_wrapper

from pathlib import Path

import numpy as np
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "simple__no_lens_light"

al.util.dataset.simulate_if_needed(
    dataset_path,
    simulator="scripts/simulator/no_lens_light.py",
)

dataset_unmasked = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

dataset_unmasked = dataset_unmasked.apply_over_sampling(
    over_sample_size_pixelization=1,
)

aplt.subplot_imaging_dataset(dataset=dataset_unmasked)

"""
We will compare two masks with the same outer radius. One includes the central image pixels; the other removes them. This lets us isolate how including the centre changes the area covered by the source mesh.

---

__What we mean by a border__

Here, the **border** means the outermost boundary of the unmasked image region: the ring of image pixels just inside the mask’s outer edge.

For a circular mask, this follows the circumference of the circle. An annular mask also has an inner edge around its central hole, but that inner edge is not the border used for relocation.

Let’s construct both masks and mark their outer borders on the image.
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
Both masks have the same outer border because their outer radii are identical. Their difference is in the centre: the circular mask includes pixels that the annular mask excludes.

We can ray-trace the marked border pixels just as we ray-trace any other image pixels. Their traced positions define a border in the **source plane**, which we will use to identify coordinates that land beyond it.

Keep these two operations separate. **Masking** determines which image pixels we fit. **Border relocation** changes how coordinates beyond the traced border enter the source reconstruction; it does not remove those image pixels from the fit.

__Starting without border relocation__

Let’s first reconstruct the source using the annular mask, which excludes the central image pixels. We explicitly switch border relocation off so that we can inspect the original ray-traced coordinates.

We use the same mass model that generated the simulated image: an isothermal galaxy with external shear. Keeping the mass model fixed lets us focus on how the mask affects the source reconstruction.
"""

mesh_shape = (25, 25)

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

"""
The helper below lets us change the mask and switch relocation on or off. We keep the mesh dimensions and regularization coefficient fixed throughout these comparisons. We are investigating the mesh geometry here, rather than finding the preferred coefficient for each setup.
"""


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
The first plot shows where the unmasked image pixels trace in the source plane. The second shows the mesh covering those coordinates and the source brightness reconstructed on it.

Pay attention to the axis ranges and the spacing of the mesh points. These provide our reference for the next comparison.

---

__Including the central image pixels__

Now repeat the reconstruction with the circular mask.

Its outer radius is unchanged. We are simply adding the image pixels inside the central hole of the annulus. Before looking at the result, consider what you expect: will adding pixels near the **image centre** necessarily add coordinates near the **source centre**?
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

"""
Compare the coordinate ranges with the annular reconstruction. The added central image pixels can trace well outside the region occupied by the other source-plane coordinates. The rectangular mesh expands to cover them.

Let’s measure the extent of each mesh.
"""

for label, mapper in [
    ("Annular", mapper_annular),
    ("Circular", mapper_circular),
]:
    coordinates = np.asarray(mapper.source_plane_mesh_grid)
    span_y, span_x = np.ptp(coordinates, axis=0)

    print(f"{label} mesh spans: " f"y={span_y:.3f} arcsec, x={span_x:.3f} arcsec")

"""
Both meshes still contain `25 × 25` points. When their extent increases, the points become more widely spaced. We therefore have fewer points across the source galaxy itself, reducing our ability to resolve its structure.

Adding more image pixels has changed the resolution available to reconstruct the source, even though we did not change the mesh dimensions!

The masks contain different sets of image pixels, so we should not rank these two fits by their absolute log evidences. For now, compare their geometry and source reconstructions.

---

__Following the central pixels into the source plane__

Let’s identify exactly where those distant source-plane coordinates came from.

We select two groups of image pixels: twelve close to the lens centre and twelve near the arc. We highlight the same index groups in both planes, so that each colour follows the same pixels through the ray tracing.
"""

image_coordinates = np.asarray(mapper_circular.image_plane_data_grid)

central_indexes = np.argsort(np.sum(image_coordinates**2, axis=1))[:12].tolist()

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
    title="Image Plane: Central Pixels and Arc Pixels",
)

aaplt.plot_grid(
    grid=mapper_circular.source_plane_data_grid,
    indexes=index_groups,
    title="Source Plane: The Same Ray-Traced Pixels",
)

r"""
The pixels near the arc trace towards the source. The central image pixels can land much further away.

Recall the lens equation:

\[
\boldsymbol{\beta}
=
\boldsymbol{\theta}
-
\boldsymbol{\alpha}(\boldsymbol{\theta}).
\]

A small image-plane position, \(\boldsymbol{\theta}\), does not necessarily give a small source-plane position, \(\boldsymbol{\beta}\). We must subtract the deflection at that position.

Near the centre of our singular isothermal lens, a small image region can correspond to a much larger source-plane region. This is **demagnification**: the image occupies less area than the corresponding region of the source plane. It does not mean that the deflection angle becomes infinite.

We can inspect a few of these coordinate pairs directly.
"""

source_coordinates = np.asarray(mapper_circular.source_plane_data_grid)

for index in central_indexes[:4]:
    print(
        f"Image pixel {index}: "
        f"{image_coordinates[index]} -> "
        f"{source_coordinates[index]}"
    )

"""
These central pixels contain no bright source emission in our simulated image, yet their traced positions still help determine the rectangular mesh’s extent. The mesh consequently spends resolution covering distant regions instead of concentrating it around the source.

There is another potential problem: assigning source brightness at these distant positions may predict additional images outside our mask. Those predictions would not be checked against the excluded image data.

Next, we will trace the outer image border into the source plane and use it to control these distant coordinates.

__Tracing the outer border__

We have seen that pixels near the image centre can trace far beyond the source galaxy. To decide which coordinates to relocate, we first need a boundary in the source plane.

We obtain this by ray-tracing the **outer border of the image mask**. These border pixels pass through the same lens equation as every other image pixel.

Let’s highlight them in the image plane, then follow the same indexes into the source plane. Relocation is still switched off, so we are looking at their original ray-traced positions.
"""

border_indexes = mask_circular.derive_indexes.border_slim.tolist()

aaplt.plot_grid(
    grid=mapper_circular.image_plane_data_grid,
    indexes=[border_indexes],
    title="Image Plane: Outer Border",
)

aaplt.plot_grid(
    grid=mapper_circular.source_plane_data_grid,
    indexes=[border_indexes],
    title="Source Plane: Ray-Traced Outer Border",
)

"""
Compare the traced border with the central pixels highlighted previously. Although those central pixels lie well inside the image mask, some land **outside the traced border** in the source plane.

This is possible because the lens mapping does not preserve the ordering of distances from the centre. Being inside the image border does not guarantee that a pixel traces inside its source-plane counterpart.

The border relocator uses the traced border to estimate a source-plane boundary. Coordinates beyond that boundary are moved inward for the reconstruction, preventing them from forcing the mesh to extend far beyond the region we want to reconstruct.

---

__Switching border relocation on__

Let’s repeat the circular-mask fit with border relocation enabled.

We use exactly the same image pixels, lens model, mesh dimensions and regularization coefficient. The change is how the source reconstruction handles coordinates beyond the traced border.
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
        title=f"{label}: Selected Source Coordinates",
    )

    mesh_coordinates = np.asarray(mapper.source_plane_mesh_grid)
    span_y, span_x = np.ptp(mesh_coordinates, axis=0)

    print(f"{label}: mesh spans " f"y={span_y:.3f} arcsec, x={span_x:.3f} arcsec")

"""
Follow the coloured groups between the plots. The distant coordinates move inward, while coordinates inside the relocation boundary retain their positions.

When this reduces the mesh’s extent, our `25 × 25` points cover a smaller area. Their spacing decreases, giving us finer resolution across the source without increasing the number of source brightness values.

Let’s identify every coordinate that moved.
"""

coordinates_before = np.asarray(mapper_circular.source_plane_data_grid)
coordinates_after = np.asarray(mapper_relocated.source_plane_data_grid)

displacements = np.linalg.norm(
    coordinates_after - coordinates_before,
    axis=1,
)

moved_indexes = np.flatnonzero(displacements > 1.0e-8).tolist()

print("Number of relocated coordinates:", len(moved_indexes))

if moved_indexes:
    aaplt.plot_grid(
        grid=mapper_circular.source_plane_data_grid,
        indexes=[moved_indexes],
        title="Relocated Coordinates: Before",
    )

    aaplt.plot_grid(
        grid=mapper_relocated.source_plane_data_grid,
        indexes=[moved_indexes],
        title="Relocated Coordinates: After",
    )

"""
These coordinates still correspond to the same image pixels and measured brightnesses. We have changed where they map within the source reconstruction. We have not moved any observed image pixels or excluded them from the fit.

---

__What relocation changes__

Does relocating these coordinates mean we have changed the lens model?

No. The mass profile and its deflection angles are unchanged. Border relocation is applied when constructing the source reconstruction, after ray tracing.

We can demonstrate this by ray-tracing the image coordinates directly through the tracer. This returns their original source-plane positions.
"""

physical_source_grid = tracer.traced_grid_2d_list_from(
    grid=dataset_circular.grids.pixelization
)[1]

physical_coordinates = np.asarray(physical_source_grid)

print(
    "Largest difference between direct ray tracing "
    "and the mapper without relocation:",
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
    print("Ray-traced position:", physical_coordinates[index])
    print("Position used after relocation:", coordinates_after[index])

"""
The first difference should be zero, apart from numerical precision. For a relocated pixel, the second comparison shows the distinction between its ray-traced position and the position used by the inversion.

Relocation therefore changes the mapping used to reconstruct the source, and can change the predicted image even though the lens mass is fixed.

Let’s inspect that effect. We use the same brightness scale for both source plots so that their colours remain comparable.
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
        array=(dataset_circular.data - current_fit.model_data)
        / dataset_circular.noise_map,
        title=f"{label}: Normalized Residuals",
    )

"""
Compare the spacing of the mesh points across the source. Does the smaller extent allow its brightness distribution to be represented more accurately? Are corresponding changes visible in the image residuals?

We have kept the regularization coefficient fixed to show the effect of switching relocation on. As we learned in tutorial 2, changing the mesh geometry can also change the preferred coefficient. A full comparison would therefore repeat the regularization search for each setup.

Next, we will examine why border relocation still requires a mask that covers the lensed source.

__The mask still needs to cover the lensed source__

Border relocation helps us control the source mesh, but it cannot compensate for an image mask that excludes part of the lensed source.

Suppose our mask cuts through an arc. The reconstruction only has to explain the pixels inside the mask; the excluded part of the arc contributes nothing to the fit. A reconstruction could therefore fit the selected pixels well while failing to explain the rest of the observed source.

The mask also determines which border coordinates we trace into the source plane. Changing its radius changes both the data being fitted and the boundary used for relocation.

Let’s inspect three masks. We draw their borders on the **full image**, so that any source emission outside each mask remains visible.
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
        title=f"Full Image: Mask Radius {radius:.1f} Arcsec",
    )

"""
Check whether each border encloses the lensed emission, including its faint extensions, with some surrounding background. We should not choose a mask simply because it closely follows the brightest pixels.

Now reconstruct the source with each mask, keeping border relocation enabled.
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
        title=f"Source: Mask Radius {radius:.1f} Arcsec",
    )

    aplt.subplot_fit_imaging(fit=current_fit)

"""
Compare the source reconstructions and residuals. Does the source change substantially as additional lensed emission enters the fit? Does a smaller mask hide a region where the model struggles?

These masks contain different image pixels, so their absolute log evidences should not be used to select the mask. Our aim is to include the relevant emission and understand how the reconstruction responds to the chosen boundary.

A larger mask can also spread the fixed number of mesh points over a larger source-plane region. If the reconstruction changes, inspect the mesh spacing as well as the newly included data.

---

__Wrap Up__

The resolution of a rectangular source mesh depends on both its number of points and the area they cover.

We saw that central image pixels can trace far from the source galaxy. Without border relocation, these coordinates can expand the mesh, leaving fewer points across the source itself.

Border relocation uses the traced outer image border to estimate a source-plane boundary. Coordinates beyond that boundary are moved inward when constructing the source reconstruction. The observed image pixels remain in the fit, and the lens mass model is unchanged.

We will keep border relocation enabled in the fits that follow. If a mesh extends unexpectedly far from the reconstructed source, we now know how to investigate: follow the image pixels into the source plane and inspect which coordinates determine its extent.

This can also matter for lenses containing several mass concentrations. The pixels responsible for distant source-plane coordinates need not all lie near the centre of the image.

Let’s finish by returning to our circular mask with radius `2.5` arcsec.
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
    title="Source Reconstruction with Border Relocation",
)

"""
So far, we have investigated the source mesh, interpolation, regularization and borders while holding the lens mass fixed.

Next, we will let the mass model vary. This introduces another challenge: a flexible source reconstruction can sometimes produce a close image fit even when the lens mass model is wrong.

---

**🤖 Ask AI:** Why can adding central image pixels reduce the useful resolution of a rectangular source mesh, and what does border relocation change?
"""
