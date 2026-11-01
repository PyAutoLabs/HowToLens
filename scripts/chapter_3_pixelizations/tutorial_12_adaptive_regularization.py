"""
Tutorial 12: Adaptive Regularization
====================================

Tutorial 2 introduced `Constant` regularization. Different regions of the source can favour different levels of
regularization, motivating a scheme which adapts to the reconstructed source's surface brightness.

This raises the same question as before, how do we adapt our regularization scheme to the source before we've
reconstructed it? Just like in the last tutorial, we'll use a model image of a strongly lensed source from a
previous model that we've begun calling the `adapt-image`.

__Contents__

- **Initial Setup:** we'll use the same strong lensing data as the previous tutorial, where.
- **Convenience Function:** We are going to fit the image using a magnification based grid.
- **Adaptive Regularization:** Lets now look at adaptive regularization in action, by setting up a adapt-image and
  using the.
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

Tutorial 8's compactness experiment motivates the need for resolution and smoothing checks. As in tutorial 11, this
standalone example uses `simple__no_lens_light`, rather than the progressively compact simulations, where:

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

"""
__Convenience Function__

We are going to fit the image using a magnification based grid.

To perform the fits, we'll use a convenience function to fit the lens data we simulated above.
"""


def fit_via_source_galaxy_from(dataset, source_galaxy, adapt_images=None):
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

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

    return al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)


"""
Use the magnification based source to fit this data.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Constant(coefficient=3.3),
)

source_magnification = al.Galaxy(redshift=1.0, pixelization=pixelization)

constant_fit = fit_via_source_galaxy_from(dataset=dataset, source_galaxy=source_magnification)
fit = constant_fit

mapper = fit.inversion.cls_list_from(al.Mapper)[0]


aplt.subplot_fit_imaging(fit=fit)
aplt.plot_array(array=fit.model_data, title="Plane 1 Image")

"""
Inspect the residuals before changing the prior. We can display the actual regularization weights on the source
mesh: Constant gives every source value the input coefficient of 3.3. These are weights, rather than source
brightnesses or an image-plane reconstruction.
"""
constant_weights = mapper.regularization.regularization_weights_from(linear_obj=mapper)
aaplt.plot_mapper(
    mapper=mapper, solution_vector=constant_weights,
    mesh_grid=mapper.source_plane_mesh_grid, zoom_to_brightest=False,
    vmin=0.0, vmax=4.0, title="Constant Regularization Weights",
)
print("Constant weight range:", float(min(constant_weights)), float(max(constant_weights)))

"""
__Adaptive Regularization__

Lets now look at adaptive regularization in action, by setting up a adapt-image and using the `Adapt`
regularization scheme.

This introduces additional parameters, that are explained below.
"""
adapt_image = fit.model_data.slim

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Adapt(
        inner_coefficient=0.005, outer_coefficient=1.9, signal_scale=3.0
    ),
)

source_adapt_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={source_adapt_regularization: adapt_image}
)

fit = fit_via_source_galaxy_from(
    dataset=dataset,
    source_galaxy=source_adapt_regularization,
    adapt_images=adapt_images,
)

adaptive_fit = fit
mapper = fit.inversion.cls_list_from(al.Mapper)[0]
adaptive_weights = mapper.regularization.regularization_weights_from(linear_obj=mapper)
aaplt.plot_mapper(
    mapper=mapper, solution_vector=adaptive_weights,
    mesh_grid=mapper.source_plane_mesh_grid, zoom_to_brightest=False,
    vmin=0.0, vmax=4.0, title="Adaptive Regularization Weights",
)
aaplt.plot_mapper(
    mapper=mapper, solution_vector=fit.inversion.reconstruction,
    mesh_grid=mapper.source_plane_mesh_grid, title="Adaptively Regularized Source",
)
print("Adaptive weight range:", float(min(adaptive_weights)), float(max(adaptive_weights)))

"""
The weights now vary across the source mesh. Compare their locations with the reconstructed source, using the same
weight colour scale as the Constant example. These weights enter the regularization matrix; they are not
uncertainties on the reconstructed brightnesses.

Let us compare the evidence measured from these two fits:
"""
print("Evidence using constant regularization:", constant_fit.log_evidence)
print("Evidence using adaptive regularization:", adaptive_fit.log_evidence)
print("Difference in log evidence:", adaptive_fit.log_evidence - constant_fit.log_evidence)

aplt.subplot_fit_imaging(fit=fit)

"""
An evidence increase is something to measure, rather than assume. The mesh is unchanged here, so this comparison
examines the regularization choice. These hand-chosen settings are not optimized, and varying the smoothing cannot
supply resolution that the source mesh lacks. Inspect the source and residuals alongside the evidence.

__How does adaptive regularization work?__

For every source-pixel, we have a mapping between that pixel and a set of pixels in the adapt-image. Therefore, for
every source-pixel, if we sum the values of all adapt-image pixels that map to it we get an estimate of how much of
the lensed source's signal we expect will be reconstructed. We call this each pixel's `pixel signal`.

If a source-pixel has a higher pixel-signal, we anticipate that it`ll reconstruct more flux and we use this
information to regularize it less. Conversely, if the pixel-signal is close to zero, the source pixel will
reconstruct near-zero flux and regularization will smooth over these pixels by using a high regularization
coefficient.

This works as follows:

 1) For every source pixel, compute its pixel-signal, the summed flux of all corresponding image-pixels in the 
 adapt-galaxy-image.
    
 2) Divide every pixel-signal by the number of image-pixels that map directly to that source-pixel. In doing so,
 all
 pixel-signals are 'relative'. This means that source-pixels which by chance map to more image-pixels than their 
 neighbors will not have a higher pixel-signal, and visa versa. This ensures the specific pixelization
 does not by itself determine the adaptive regularization pattern.
    
 3) Divide the pixel-signals by the maximum pixel signal so that they range between 0.0 and 1.0.
    
 4) Raise these values to the power of the adapt-parameter `signal_scale`. For a `signal_scale` of 0.0, all 
 pixels will therefore have the same final pixel-scale. As the `signal_scale` increases, a sharper transition of 
 pixel-signal values arises between regions with high and low pixel-signals.
    
 5) Compute every source pixel's effective regularization coefficient as:
    
 (inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)) ** 2.0
    
 This uses two regularization coefficient parameters, one which is applied to pixels with high pixel-signals and
 one to
 pixels with low pixel-signals. Thus, pixels in the inner regions of the source may be given a lower level of 
 regularization than pixels further away, as desired.

Thus, we now adapt our regularization scheme to the source's surface brightness. Where its brighter (and therefore
has a steeper flux gradient) we apply a lower level of regularization than further out. Furthermore, in the edges
of the source-plane where no source-flux is present we will assume a high regularization coefficient that smooths
over all of the source-pixels.

Try looking at a couple of extra solutions which use with different inner and outer regularization coefficients or
signal scales. I doubt you'll notice a lot change visually, but the log evidence certainly has a lot of room for
manuveur with different values.

You may find solutions that raise an `InversionException`. These solutions mean that the matrix used during the
linear algebra calculation was ill-defined, and could not be inverted. These solutions are removed during lens
modeling.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(24, 24)),
    regularization=al.reg.Adapt(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
)

source_adapt_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={source_adapt_regularization: adapt_image}
)

fit = fit_via_source_galaxy_from(
    dataset=dataset,
    source_galaxy=source_adapt_regularization,
    adapt_images=adapt_images,
)

mapper = fit.inversion.cls_list_from(al.Mapper)[0]


aplt.subplot_fit_imaging(fit=fit)

aplt.plot_array(
    array=fit.inversion.mapped_reconstructed_operated_data,
    title="Inversion Reconstruction",
)

alternative_weights = mapper.regularization.regularization_weights_from(linear_obj=mapper)
aaplt.plot_mapper(
    mapper=mapper, solution_vector=alternative_weights,
    mesh_grid=mapper.source_plane_mesh_grid, zoom_to_brightest=False,
    vmin=0.0, vmax=4.0, title="Alternative Adaptive Regularization Weights",
)
print("Evidence using alternative adaptive regularization:", fit.log_evidence)
print("Difference from constant regularization:", fit.log_evidence - constant_fit.log_evidence)
print("Difference from first adaptive example:", fit.log_evidence - adaptive_fit.log_evidence)

"""
__Wrap Up__

Adaptive regularization lets bright and faint parts of the source receive different smoothing weights. Lower
weights can preserve sharp structure where the data support it; higher weights can correlate values in faint
regions that would otherwise fit noise. The adapt image guides that distinction and must itself be checked.

The evidence comparisons quantify the outcomes for our chosen settings. They do not guarantee an increase for every
dataset, a unique maximum, or recovery of arbitrary source detail. Mesh resolution, the adapt image and the lens
model still limit the reconstruction. In a scientific analysis, explore the regularization parameters and inspect
the source and residuals together.

That completes this chapter on pixelized source reconstruction. You can now investigate flexible source
reconstructions while checking the assumptions that shape them. In chapter 4, `Scaling Up Lensing`, we take these
tools to bigger systems: lenses with extra galaxies along the line of sight, multi-galaxy lenses, scaling
relations, group and cluster scale lenses and weak lensing.
"""
