"""
Tutorial 2: Mappers
===================

In the previous tutorial, we used a pixelization to create a `Mapper`. However, it was not clear what a `Mapper`
does, why it was called a mapper and whether it was mapping anything at all!

Therefore, in this tutorial, we'll cover mappers in more detail.

__Contents__

- **Initial Setup:** we'll use new strong lensing data, where.
- **Mappers:** We now setup a `Pixelization` and use it to create a `Mapper` via the tracer`s source-plane grid.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Wrap Up:** Summary of the script and next steps.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

"""
__Initial Setup__

we'll use new strong lensing data, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/no_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
    over_sample_size_pixelization=1,
)

"""
Now, lets set up our `Grid2D` (using the image above).
"""
grid = al.Grid2D.uniform(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    over_sample_size=1,
)

"""
Our `Tracer` will use the same lens galaxy and source galaxy that we used to simulate the imaging data (although,
because we're modeling the source with a pixel-grid, we do not pass the source any light profiles).
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

tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(grid=grid)[1]

"""
__Mappers__

We now setup a `Pixelization` and use it to create a `Mapper` via the tracer`s source-plane grid, just like we did in
the previous tutorial.
"""
mesh = al.mesh.RectangularBilinearAdaptDensity(shape=(25, 25))

pixelization = al.Pixelization(mesh=mesh)

interpolator = mesh.interpolator_from(
    source_plane_data_grid=source_plane_grid, source_plane_mesh_grid=None
)

mapper = al.Mapper(interpolator=interpolator)


"""
Lets look at the two things a mapper holds. In the image-plane it has the (y,x) coordinate of every image-pixel it
uses, and in the source-plane it has the rectangular mesh those coordinates land on after being ray-traced by the
tracer.
"""
aplt.plot_array(
    array=dataset.data, title="Image", positions=mapper.image_plane_data_grid
)
aplt.plot_grid(grid=mapper.source_plane_mesh_grid, title="Source-Plane Mesh Grid")

"""
The mapper's job is to pair those two things together. Before we look at how it pairs whole source pixels, it helps
to see the simpler, point-level version: take a handful of image-pixel coordinates and follow the *same* coordinates
into the source-plane.

We do this by giving the `indexes=` input a list of index groups, where each group is drawn in its own colour. The
same colour therefore marks the same coordinates in both planes.
"""
total_points = len(mapper.image_plane_data_grid)

side = int(round(total_points**0.5))

centre = side * (side // 2) + side // 2

indexes = [
    [index for index in range(0, 10) if index < total_points],
    [
        centre + step * side
        for step in range(-5, 5)
        if 0 <= centre + step * side < total_points
    ],
]

aaplt.plot_grid(
    grid=mapper.image_plane_data_grid,
    indexes=indexes,
    title="Image-Plane Data Grid",
)
aaplt.plot_grid(
    grid=mapper.source_plane_data_grid,
    indexes=indexes,
    title="Source-Plane Data Grid (Traced)",
)

"""
The red points, the first ten image-pixels at the very corner of the image, barely move: they are far from the lens
galaxy, so they are hardly deflected. The green points run vertically through the centre of the image, straight
through the lens galaxy, and are dragged a long way -- in the source-plane they no longer form a straight line, and
they pile up near the centre.

This is only half the story though, because it tells us where individual coordinates go, not which *source pixel*
they end up inside. That is what a mapper actually stores, and it is what we look at next.

We can now make the mapping appear the other way round. That is, we can input a source-pixel index (of our
rectangular mesh) and ask the mapper which image-pixels land inside it.

`mapper.mappings_from` answers exactly this. It takes a list of source-pixel index *groups* and returns one `Mapping`
per group. Each `Mapping` carries:

 - `source_contours`: the outline of the source-plane cell(s) in the group.
 - `image_contours`: the outlines of the connected regions of image-pixels which map into that group.

Both are polygons in arc-seconds, so we can draw them straight onto the image and onto the source-plane, in the same
colour. That colour is the whole point: it says "this source region produces these image regions".

Lets map source pixel 312, the central source-pixel, and draw it in both planes. We clamp the index with
`min(index, mapper.pixels - 1)` so the tutorial still runs if you shrink the mesh.
"""
pix_indexes = [[min(312, mapper.pixels - 1)]]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
There we have it, multiple imaging in all its glory.

On the right, a single small red rectangle: one cell of the source-plane mesh. On the left, that same red appears in
two or more *separate* regions of the image, sitting on the lensed source's ring. The lens galaxy's mass has taken
one patch of source and produced several images of it, and the mapper knows which image-pixels belong to which.

Notice also that the red regions in the image-plane are wider than a strict "four image pixels per source pixel"
footprint would be. This is because the pairing of image-pixels to source-pixels is not one-to-one: a bilinear
interpolation scheme is used, so an image-pixel which lands near the edge of a source-pixel, but outside it, is still
paired with that source-pixel with a weight.

Try changing the source-pixel indexes below. This will give you a feel for how different regions of the source-plane
map to the image.
"""
pix_indexes = [
    [min(index, mapper.pixels - 1) for index in group] for group in [[312, 313], [412]]
]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
Two groups, two colours. The first group is two neighbouring source pixels, so red covers twice the area of the
source-plane and correspondingly wider arcs in the image. The second group is a single pixel further from the centre
of the source-plane, drawn in green, and its images land somewhere else entirely on the ring.

This is the key insight of the whole chapter: *every* source pixel has a set of image regions like this, and the
collection of all of them is the "mapping matrix" that an inversion solves. Tutorial 3 puts that matrix to work.

Okay, so I think we can agree, mapper's map things! More specifically, they map source-plane pixels to multiple pixels
in the observed image of a strong lens.

__Mask__

Finally, lets repeat the steps that we performed above, but now using a masked image. By applying a `Mask2D`, the
mapper only maps image-pixels that are not removed by the mask. This removes the (many) image pixels at the edge of the
image, where the source is not present. These pixels also pad-out the source-plane, thus by removing them our
source-plane reduces in size.

Lets first see why they are worth removing, by drawing source pixels near the edge of the mesh:
"""
pix_indexes = [
    [min(index, mapper.pixels - 1) for index in group]
    for group in [[600, 601, 602, 603, 604], [620, 621, 622, 623, 624]]
]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
Both groups sit along the bottom edge of the source-plane mesh, and both map to a single blob in a corner of the
image, far from the lensed source, where there is no signal to reconstruct. They are wasted source pixels: the
inversion has to solve for them, but the data has nothing to say about them.

Lets use an annular `Mask2D`, which will capture the ring-like shape of the lensed source galaxy.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=1.0,
    outer_radius=2.2,
)

dataset = dataset.apply_mask(mask=mask)
aplt.plot_array(array=dataset.data, title="Data")

"""
To create the mapper, we need to trace the masked imaging's grid to the source-plane via the tracer.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization)[1]

"""
We can now use the masked source-plane grid to create a new `Mapper` (using the same rectangular 25 x 25 pixelization 
as before).
"""
interpolator = mesh.interpolator_from(
    source_plane_data_grid=source_plane_grid, source_plane_mesh_grid=None
)
mapper = al.Mapper(interpolator=interpolator)

"""
Lets plot it, including the image-plane grid and source-plane grid, which are now much smaller than before because
the mask has removed the many image pixels at the edge of the image.
"""


aplt.plot_array(array=dataset.data, title="Image")


aplt.plot_grid(grid=mapper.source_plane_mesh_grid, title="Source-Plane Mesh Grid")

"""
First, look how much closer we are to the source-plane (The axis sizes have decreased from ~ -2.5" -> 2.5" to 
~ -0.6" to 0.6"). 

We can more clearly see the diamond of points in the centre of the source-plane (for those who have been reading up, 
this diamond is called the `caustic`).

The source-plane is now only ~1" across instead of ~7", so every source pixel covers a much larger fraction of the
figure and the polygons we drew above are far easier to read. Lets draw four source pixels at once, each in its own
colour.
"""
pix_indexes = [
    [min(index, mapper.pixels - 1)] for index in [312, 314, 316, 318]
]

mappings = mapper.mappings_from(pix_indexes=pix_indexes)

aaplt.subplot_image_and_mapper(
    mapper=mapper,
    image=dataset.data,
    regions=mappings,
    mesh_grid=mapper.source_plane_mesh_grid,
)

"""
Four source pixels, four colours, and in the image-plane each colour picks out its own short segments of the ring,
on opposite sides of it. Source pixels which are neighbours in the source-plane produce image regions which are
neighbours along each arc -- the lensing is smooth, and the mapper preserves that.

The number of image-plane regions each source pixel maps to is the number of multiple images it produces (the mask
can split one arc into more than one region, so treat these counts as a lower bound on the physical multiplicity):
"""
for mapping in mappings:
    print(
        f"Source pixel {mapping.pix_indexes} maps to {len(mapping.image_regions)} image-plane regions."
    )

"""
__Wrap Up__

In this tutorial, we learnt about mappers, and we used them to understand how the image and source plane map to one 
another. Your exercises are:

 1) Change the einstein radius of the lens galaxy in small increments (e.g. einstein radius 1.6" -> 1.55"). As the 
 radius deviates from 1.6" (the input value of the simulated lens), what do you notice about where the points map 
 from the centre of the source-plane (where the source-galaxy is simulated, e.g. (0.0", 0.0"))?
        
 2) Think about how this could help us actually model lenses. We have said we're going to reconstruct our source
 galaxies on the pixel-grid. So, how does knowing how each pixel maps to the image actually help us? If you've not got
 any bright ideas, then worry not, that's exactly what we're going to cover in the next tutorial.
"""
