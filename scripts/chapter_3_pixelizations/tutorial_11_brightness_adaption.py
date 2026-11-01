"""
Tutorial 11: Brightness Adaption
================================

In the previous tutorial we motivated our need to adapt the pixelization to the source's morphology, such that
source pixels congregates in the source's brightest regions regardless of where the source is located in the
source-plane.

This poses a challenge; how do we adapt our pixelization to the reconstructed source's light, before we've actually
reconstructed the source and therefore know what to adapt it too?

To do this, we define 'adapt_images' of the lensed source galaxy, which are model images of the source computed
using a previous lens model that has been fit to the image (e.g. in the earlier search of a pipeline). This image
tells us where in the image our source is located, thus informing us of where we need to adapt our source
pixelization!

Tutorial 8's compactness experiment motivated adapting to the source's morphology. Here we use the standalone
`simple__no_lens_light` dataset, rather than one of those progressively compact simulated sources. We begin with a
magnification-based pixelization and use its model image as an adapt image.

__Contents__

- **Initial Setup:** we'll use the same strong lensing data as the previous tutorial, where.
- **Adapt Image:** We can use this fit to set up our adapt image.
- **Adaption:** Now lets take a look at brightness based adaption in action.
- **Hilbert:** So how does the `adapt_image` adapt the pixelization to the source's brightness?
- **Weight Map:** We now have a sense of how our `Hilbert` image-mesh is computed, so lets look at how we create
  the.
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
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

"""
__Initial Setup__

we'll use the same strong lensing data as the previous tutorial, where:

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
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_sparse_operator()

"""
Next, we're going to fit the image using the RectangularBilinearAdaptDensity magnification based grid.

The code below does all the usual steps required to do this.
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

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Constant(coefficient=3.3),
)

source_galaxy_magnification = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_magnification], fields=[field])

magnification_fit = al.FitImaging(dataset=dataset, tracer=tracer)
fit = magnification_fit

"""
Inspect the source reconstruction and residuals before using its model image to guide the next mesh.
"""
mapper = fit.inversion.cls_list_from(al.Mapper)[0]


aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
__Adapt Image__

We can use this fit to set up our adapt image.

The adapt image is conditional on this first fit. Check its residuals: an inaccurate starting model can guide the
next mesh to the wrong parts of the image.
"""
adapt_image = fit.model_data.slim

"""
__Adaption__

Now lets take a look at brightness based adaption in action.

Below, we define a source galaxy using the `Hilbert` image-mesh and a `Delaunay` mesh. We discuss below how Hilbert
adapts to the source light. Unlike the density-adaptive rectangular mesh, Delaunay uses the supplied Hilbert points
as its source-mesh vertices after ray tracing.
"""
image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=10.0)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)

pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0]),
    regularization=al.reg.Constant(coefficient=1.0),
)

galaxy_adapt = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

"""
The adapt image is paired to the galaxy that it represents and that it is used to adapt the mesh too.

This uses the `AdaptImages` object, which receives a dictionary mapping every galaxy to its adapt image.
"""
adapt_images = al.AdaptImages(
    galaxy_image_dict={galaxy_adapt: adapt_image},
    galaxy_image_plane_mesh_grid_dict={galaxy_adapt: image_plane_mesh_grid},
)

"""
We now fit using this adapt image and mesh using the normal API.

Note however that the `FitImaging` object receives the `adapt_images` as an input and they are used when setting up
the image-mesh and mesh.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, galaxy_adapt], fields=[field])

brightness_fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)
fit = brightness_fit

aplt.plot_array(
    array=dataset.data, positions=image_plane_mesh_grid, title="Hilbert Image-Mesh Points",
)
mapper = fit.inversion.cls_list_from(al.Mapper)[0]
aaplt.plot_mapper(
    mapper=mapper, solution_vector=fit.inversion.reconstruction,
    mesh_grid=mapper.source_plane_mesh_grid, title="Brightness-Adaptive Source Mesh",
)
aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)


"""
Compare the source mesh and residuals with the initial density-adaptive reconstruction. We can now compare measured
evidence values instead of carrying a number over from another simulation. This first comparison changes both mesh
family and regularization coefficient; it does not isolate brightness weighting alone.
"""
print("Evidence using magnification based pixelization:", magnification_fit.log_evidence)
print("Evidence using brightness based pixelization:", brightness_fit.log_evidence)
print("Difference in log evidence:", brightness_fit.log_evidence - magnification_fit.log_evidence)
print("Source mesh values:", mapper.pixels)

"""
Brightness adaptation gives us another way to distribute source resolution. It does not guarantee an improved fit
or higher evidence for every dataset. Inspect both the calculated comparison and the residuals.

__Hilbert__

So how does the `adapt_image` adapt the pixelization to the source's brightness? It uses a standard algorithm for
partitioning data in statistics called a Hilbert curve:

https://en.wikipedia.org/wiki/Hilbert_curve

In simple terms, this algorithm works as follows:

 1) Input an image of weight values to the Hilbert algorithm which determines the "hilbert space filling curve" 
 (e.g. this `weight_map` is determined from the adapt-image). The Hilbert space-filling curve fills more space
 there
 the weight values are highest, and less space where they are lowest. It therefore closely traces the brightest
 regions of the image.
    
 2) Probabilistically draw N $(y,x)$ points from this Hilbert curve, where the majority of points will therefore be
 drawn from high weighted regions. 
    
 3) These N $(y,x)$ are our source-pixel centres, albeit we have drawn them in the image-plane so we first map them
 to the source-plane, via the mass model, in order to set up the source pixel centres of the mesh. Because points
 are
 drawn from high weighted regions (e.g. the brightest image pixels in the lensed source adapt image), we will trace
 more source-pixels to the brightest regions of where the source is actually reconstructed.
 
__Weight Map__

We now have a sense of how our `Hilbert` image-mesh is computed, so lets look at how we create the weighted data
the Hilbert space filling curve uses.

This image, called the `weight_map` is generated using the `weight_floor` and `weight_power` parameters of the
`Hilbert` object. The weight map is generated following 4 steps:

 1) Take an absolute value of all pixels in the adapt image, because negative values break the Hilbert algorithm.
    
 2) Divide all values of this image by its maximum value, such that the adapt-image now only contains values
 between
 0.0 and 1.0 (where the values of 1.0 are the maximum values of the adapt-image).
    
 3) Raise all values to the power of `weight_power`. A power of 1.0 leaves them unchanged; a power of 0.0
 makes all weights equal to 1.0.

 4) Set any value below `weight_floor` to that floor. This is a minimum weight, not a value added to every pixel.
 
The idea is that using high values of `weight_power` will make the highest weight values much higher than the
lowest values, such that the Hilbert curve will trace these values much more than the lower values. The
weight_floor gives the algorithm some balance, by introducing a floor to the weight map that prevents the lowest
values from being weighted to near zero.

Lets look at this in action. we'll inspect 3 weight_maps, using a weight_power of 0.0, 5.0 and 10.0 and setting the
`weight_floor` to 0.0 for now.
"""
image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=0.0)

weight_map = image_mesh.weight_map_from(adapt_data=adapt_image)
weight_map = al.Array2D(values=weight_map, mask=mask)

aplt.plot_array(array=weight_map, title="Hilbert Weight Power 0: Weight Map")


image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=5.0)

weight_map = image_mesh.weight_map_from(adapt_data=adapt_image)
weight_map = al.Array2D(values=weight_map, mask=mask)

aplt.plot_array(array=weight_map, title="Hilbert Weight Power 5: Weight Map")


image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=10.0)

weight_map = image_mesh.weight_map_from(adapt_data=adapt_image)
weight_map = al.Array2D(values=weight_map, mask=mask)

aplt.plot_array(array=weight_map, title="Hilbert Weight Power 10: Weight Map")

"""
When we increase the weight-power the brightest regions of the adapt-image become weighted higher relative to the
fainter regions.

This means that the Hilbert algorithm will adapt more pixels to the brightest regions of the source.

Let us compare powers 0, 5 and 10 using the same number of Hilbert points, Delaunay mesh and regularization
coefficient. We reuse the power-10 reconstruction above. Each new power changes the mesh points, so we must
reconstruct the source again before comparing evidence.
"""
fits_by_power = {10.0: brightness_fit}

for weight_power in (0.0, 5.0, 10.0):
    if weight_power not in fits_by_power:
        image_mesh = al.image_mesh.Hilbert(
            pixels=500, weight_floor=0.0, weight_power=weight_power,
        )
        image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
            mask=dataset.mask, adapt_data=adapt_image,
        )
        source = al.Galaxy(
            redshift=1.0,
            pixelization=al.Pixelization(
                mesh=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0]),
                regularization=al.reg.Constant(coefficient=1.0),
            ),
        )
        adapt_images = al.AdaptImages(
            galaxy_image_dict={source: adapt_image},
            galaxy_image_plane_mesh_grid_dict={source: image_plane_mesh_grid},
        )
        tracer = al.Tracer(galaxies=[lens_galaxy, source], fields=[field])
        fits_by_power[weight_power] = al.FitImaging(
            dataset=dataset, tracer=tracer, adapt_images=adapt_images,
        )

    current_fit = fits_by_power[weight_power]
    current_mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]
    aaplt.plot_mapper(
        mapper=current_mapper, solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=current_mapper.source_plane_mesh_grid,
        title=f"Hilbert Weight Power {weight_power:g}",
    )
    aplt.subplot_fit_imaging(fit=current_fit)
    print(f"Weight power {weight_power:g}: log evidence={float(current_fit.log_evidence):.3f}")

print(
    "Log evidence difference, power 10 minus power 0:",
    fits_by_power[10.0].log_evidence - fits_by_power[0.0].log_evidence,
)

"""
So, what does the `weight_floor` do? Increasing the weight-power congregates pixels around the source. However,
there is a risk that by congregating too many source pixels in its brightest regions we lose resolution further
out, where the source is bright, but not its brightest!

The `weight_floor` allows these regions to maintain a higher weighting whilst the `weight_power` increases. This
means that the mesh can retain more coverage away from the brightest regions. Check whether this improves the fit.

Let us compare a floor of 0.5 with the power-10, floor-zero fit, keeping the other settings fixed:
"""
image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.5, weight_power=10.0)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=adapt_image
)


pixelization = al.Pixelization(
    mesh=al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0]),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_weight_floor = al.Galaxy(
    redshift=1.0, pixelization=pixelization
)

weight_map = image_mesh.weight_map_from(adapt_data=adapt_image)
weight_map = al.Array2D(values=weight_map, mask=mask)

aplt.plot_array(array=weight_map, title="Hilbert Weight Floor 0.5: Weight Map")

tracer = al.Tracer(galaxies=[lens_galaxy, source_weight_floor], fields=[field])

adapt_images = al.AdaptImages(
    galaxy_image_dict={source_weight_floor: adapt_image},
    galaxy_image_plane_mesh_grid_dict={source_weight_floor: image_plane_mesh_grid},
)

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

floor_fit = fit
floor_mapper = fit.inversion.cls_list_from(al.Mapper)[0]
aaplt.plot_mapper(
    mapper=floor_mapper, solution_vector=fit.inversion.reconstruction,
    mesh_grid=floor_mapper.source_plane_mesh_grid, title="Hilbert Weight Floor 0.5",
)
aplt.subplot_fit_imaging(fit=fit)
print("Evidence with weight floor 0.0:", fits_by_power[10.0].log_evidence)
print("Evidence with weight floor 0.5:", floor_fit.log_evidence)
print("Difference in log evidence:", floor_fit.log_evidence - fits_by_power[10.0].log_evidence)

"""
__Wrap Up__

The measured evidence comparisons show how these choices affect this dataset. Revisit tutorial 2 for the
interpretation of evidence: image agreement and the source prior must be considered together.

Brightness weighting can concentrate mesh points where source light is expected. Excessive concentration can leave
faint regions poorly resolved, and a more detailed mesh does not establish that every reconstructed feature is
real. Inspect the mesh, source and residuals together before choosing settings.

Here we held the coefficient fixed while comparing powers and floors. A full analysis would also explore
regularization, since its preferred strength depends on mesh geometry. Fewer source values can reduce the cost of
the inversion, but neither faster fits nor higher evidence are guaranteed by brightness adaptation alone.

Tutorial 11 done, next up, adaptive regularization!
"""
