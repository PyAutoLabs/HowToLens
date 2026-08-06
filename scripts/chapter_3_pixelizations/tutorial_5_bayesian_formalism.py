"""
Tutorial 5: Bayesian Formalism
==============================

In tutorials 1 to 4, we built an intuition for how pixelized source reconstruction works: pixelizations place a
pixel-grid in the source-plane, mappers pair source-pixels with image-pixels, inversions solve for the source-pixel
fluxes that best fit the data, and regularization smooths the solution within a Bayesian framework.

This tutorial collects the linear algebra behind all of that in one place. It is the counterpart of chapter 1's
tutorial 5 on the lensing formalism: the hands-on tutorials built intuition, and now we write down the equations.
We will construct, step-by-step and in code, every matrix and vector the inversion uses, solve for the source
reconstruction ourselves and compute the Bayesian evidence by hand, comparing our answer at the end to the
`FitImaging` object which performs this calculation internally.

None of this is required to *use* **PyAutoLens** for lens modeling. However, if you publish results which use a
pixelization, this is the calculation your paper's likelihood function section will describe, and understanding it
removes any sense that the source reconstruction is a "black box".

The formalism follows Warren & Dye 2003 (https://arxiv.org/abs/astro-ph/0302587), hereafter WD03, with the data
vector notation of Nightingale & Dye 2015 (https://arxiv.org/abs/1412.7436), hereafter N15. The Bayesian evidence
was derived by Suyu et al. 2006 (https://arxiv.org/abs/astro-ph/0601493) and translated to **PyAutoLens** notation
in Dye et al. 2008 (https://arxiv.org/abs/0804.4002).

__Contents__

- **Initial Setup:** Load the dataset, mask it and disable over sampling so the algebra stays simple.
- **Mesh Shape:** Fix the rectangular mesh's shape and note why edge pixels are zeroed.
- **Ray Tracing:** Trace the masked image-pixel coordinates to the source-plane via the lens equation.
- **Border Relocation:** Relocate demagnified coordinates that trace far outside the source-plane border.
- **Source Pixel Centres:** Overlay the rectangular mesh over the traced coordinates.
- **Interpolation:** Pair every traced image-pixel with source pixels via bilinear interpolation.
- **Mapper:** Package the interpolation into a `Mapper` describing all image-to-source mappings.
- **Mapping Matrix:** Express the mappings as the 2D matrix $f$.
- **Blurred Mapping Matrix:** Convolve every column of $f$ with the imaging PSF.
- **Data Vector:** Compute the data vector $D$ from the blurred mapping matrix, data and noise-map.
- **Curvature Matrix:** Compute the curvature matrix $F$.
- **Unregularized Solve:** Solve $s = F^{-1} D$ and see the over-fitted mess this produces.
- **Regularization Matrix:** Compute the regularization matrix $H$ encoding the smoothness prior.
- **Source Reconstruction:** Solve the regularized system $s = [F + H]^{-1} D$.
- **Image Reconstruction:** Map the reconstruction back to the image-plane via the blurred mapping matrix.
- **Likelihood Function:** The five terms which combine into the log evidence.
- **Chi Squared:** The goodness-of-fit of the reconstructed image to the data.
- **Regularization Term:** The penalty $s^{T} H s$ applied by the smoothness prior.
- **Complexity Terms:** The log determinant terms which penalize complex source reconstructions.
- **Noise Normalization Term:** The Gaussian noise normalization.
- **Log Evidence:** Combine all five terms into the log evidence.
- **Fit:** Compare our by-hand log evidence to the `FitImaging` object's internal calculation.
- **Wrap Up:** Summary and next steps.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use the same strong lensing data as the previous tutorials, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.

Because the lens galaxy's light is omitted, the data is already "lens subtracted". WD03 make the same assumption,
so in the equations below the lens light model $b_{j}$ is zero everywhere. For a lens with light, one simply computes
the lens galaxy's PSF-convolved image first and subtracts it from the data before the steps below (the workspace
guide referenced at the end shows this in full).
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
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

The likelihood is only evaluated within a mask, which we define as a 3.0" circle containing all of the lensed
source's light.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

masked_dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=masked_dataset)

"""
__Over Sampling__

Over sampling splits every image pixel into a sub-grid of sub-pixels, each of which is traced to the source-plane
and paired with source pixels individually. It improves accuracy, but it also multiplies the number of rows in every
matrix below by the number of sub-pixels.

To keep the algebra as easy to follow as possible, we disable over sampling by setting both sub-grid sizes to 1, so
each image pixel is represented by the single coordinate at its centre.
"""
masked_dataset = masked_dataset.apply_over_sampling(
    over_sample_size_lp=1,
    over_sample_size_pixelization=1,
)

"""
__Mesh Shape__

The `mesh_shape` defines the number of pixels in the rectangular mesh used to reconstruct the source, set below
to 20 x 20 = 400 source pixels.

We use the `RectangularUniform` mesh, where all rectangular source pixels have the same size, rather than the
`RectangularAdaptDensity` mesh used in the previous tutorials. The uniform mesh keeps the geometry simple, and every
equation below applies unchanged to the adaptive meshes -- only the source pixel centres move.

By default, source pixels at the edge of the mesh are forced to solutions of zero flux by the linear algebra solver.
This prevents unphysical solutions where the mesh edge lights up to fit residuals, and does not change any of the
formalism below.
"""
mesh_pixels_yx = 20
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Tracer__

We use the same lens galaxy mass model as the previous tutorials (an `Isothermal` plus `ExternalShear`, the true
model of the simulated data) and a source galaxy whose `Pixelization` pairs the `RectangularUniform` mesh with
`Constant` regularization (whose role appears later, when we reach the matrix $H$).
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

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
__Ray Tracing__

Every 2D (y,x) image-plane coordinate $\theta$ is ray-traced to its source-plane coordinate $\beta$ by subtracting
the deflection angles $\alpha$ of the mass profiles, via the lens equation we met in chapter 1:

 $\beta = \theta - \alpha(\theta)$

The pixelization uses its own grid of coordinates, `masked_dataset.grids.pixelization`, with one coordinate at the
centre of every masked image pixel (because we disabled over sampling above). We trace this grid to the source-plane,
taking the final entry of the traced grid list (the source-plane).
"""
traced_grid_pixelization = tracer.traced_grid_2d_list_from(
    grid=masked_dataset.grids.pixelization
)[-1]

aplt.plot_grid(grid=traced_grid_pixelization, title="Traced Source-Plane Grid")

"""
__Border Relocation__

Coordinates which pass close to the mass profile's centre are heavily demagnified and can trace to the far outskirts
of the source-plane.

We relocate these coordinates to the edge of the source-plane border (defined via the border of the image-plane
mask), exactly as tutorial 6 explains in detail (see also figure 2 of https://arxiv.org/abs/1708.07377). This stops
a handful of demagnified coordinates stretching the mesh over a huge, empty region of the source-plane.
"""
from autoarray.inversion.mesh.border_relocator import BorderRelocator

border_relocator = BorderRelocator(mask=masked_dataset.mask, sub_size=1)

relocated_grid = border_relocator.relocated_grid_from(grid=traced_grid_pixelization)

aplt.plot_grid(grid=relocated_grid, title="Relocated Source-Plane Grid")

"""
__Source Pixel Centres__

To reconstruct the source on a `RectangularUniform` mesh, we need the centres of its rectangular source pixels.

We compute these by overlaying a uniform rectangular grid over the relocated traced grid, sized so the mesh fully
contains the region of the source-plane the traced image-pixels cover, without wasted edge pixels beyond it.
"""
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import overlay_grid_from

mesh_grid = overlay_grid_from(
    shape_native=mesh_shape, grid=al.Grid2DIrregular(relocated_grid)
)

"""
__Interpolation__

We now combine the two grids to create an `Interpolator`, which describes how every traced image-pixel coordinate
maps to the rectangular mesh pixels.

For a rectangular mesh the scheme is bilinear interpolation: every image pixel maps to the rectangular pixel it
lands in *and* its three nearest neighbours, with weights set by how close the coordinate is to each pixel centre.
Interpolation is what lets the mesh reconstruct smooth source morphologies. We can print the mappings and weights of
the first image pixel to confirm it pairs with four source pixels.
"""
interpolator = pixelization.mesh.interpolator_from(
    source_plane_data_grid=relocated_grid,
    source_plane_mesh_grid=mesh_grid,
)

print(interpolator.mappings[0])
print(interpolator.weights[0])

"""
__Mapper__

The interpolator creates a `Mapper`, the object tutorials 1 and 2 introduced. It packages up the mapping between
every image pixel and every rectangular source pixel.

The key attribute is `pix_indexes_for_sub_slim_index`, mapping every image-pixel index (`sub_slim_index`) to the
source-pixel indexes (`pix_indexes`) it interpolates onto, alongside the number of mappings per image pixel and
their interpolation weights.
"""
mapper = al.Mapper(interpolator=interpolator)

pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index

print(pix_indexes_for_sub_slim_index[0:9])
print(mapper.pix_sizes_for_sub_slim_index[0:9])
print(mapper.pix_weights_for_sub_slim_index[0:9])

"""
__Mapping Matrix__

The `mapping_matrix` expresses these image-pixel to source-pixel mappings as a single 2D matrix, with
dimensions `(total_image_pixels, total_source_pixels)`.

Each column is the "image" of one source pixel: entries are the interpolation weights for image pixels which map to
that source pixel and zero everywhere else.
"""
mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=mapper.over_sampler.sub_fraction,
)

plt.imshow(mapping_matrix, aspect=(mapping_matrix.shape[1] / mapping_matrix.shape[0]))
plt.show()
plt.close()

"""
Because each column is an image of zeros and interpolation weights, we can plot any column as a 2D image showing all
image pixels its source pixel maps to. For a source pixel near the mesh centre these image pixels trace the multiple
images of that patch of the source.
"""
source_pixel_index = 210

array_2d = al.Array2D(
    values=mapping_matrix[:, source_pixel_index], mask=masked_dataset.mask
)

aplt.plot_array(array=array_2d, title="Image of Source Pixel 210")

"""
__Blurred Mapping Matrix__

The imaging data is blurred by the telescope's PSF, so the model must be too. Since each column of the mapping
matrix is an image, we simply convolve each column with the PSF via 2D convolution.

This produces the `blurred_mapping_matrix`, with the same dimensions `(total_image_pixels, total_source_pixels)`.

In WD03 this matrix is denoted $f_{ij}$, where $i$ runs over all $I$ source pixels and $j$ over all $J$ image
pixels. For example:

 - $f_{0, 2} = 0.3$ indicates that image-pixel $2$ maps to source-pixel $0$ with a weight of $0.3$ after PSF
   convolution.
 - $f_{4, 8} = 0$ indicates that image-pixel $8$ does not map to source-pixel $4$, even after PSF convolution.

(The indexing of the code's `mapping_matrix` is transposed relative to WD03's $f$: image pixels are the first index
in the code, but the second index in the equations.)

PSF blurring has an important consequence: whereas before, no two source pixels shared an image pixel, blurring
correlates neighbouring columns, so the images of nearby source pixels now overlap.
"""
blurred_mapping_matrix = masked_dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=masked_dataset.mask
)

plt.imshow(
    blurred_mapping_matrix,
    aspect=(blurred_mapping_matrix.shape[1] / blurred_mapping_matrix.shape[0]),
)
plt.colorbar()
plt.show()
plt.close()

"""
__Data Vector (D)__

We now pose the reconstruction as a linear inversion, converting the blurred mapping matrix, data and noise-map
into two objects: the data vector $D$ and the curvature matrix $F$.

The data vector has dimensions `(total_source_pixels,)` and is given by (WD03 / N15):

 $\vec{D}_{i} = \sum_{j=1}^{J} f_{ij} (d_{j} - b_{j}) / \sigma_{j}^2 \, \, .$

Where:

 - $d_{j}$ are the image-pixel data values.
 - $b_{j}$ are the lens light model values ($d_{j} - b_{j}$ is the lens-subtracted image; zero for this dataset).
 - $\sigma_{j}^2$ are the statistical uncertainties of each image pixel.

Each entry of $D$ is therefore the noise-weighted overlap between one source pixel's blurred image and the data:
it measures how much evidence the data provides for flux in that source pixel, with the PSF fully accounted for.
"""
data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix=blurred_mapping_matrix,
    image=np.array(masked_dataset.data),
    noise_map=np.array(masked_dataset.noise_map),
)

plt.imshow(
    data_vector.reshape(data_vector.shape[0], 1), aspect=10.0 / data_vector.shape[0]
)
plt.colorbar()
plt.show()
plt.close()

"""
__Curvature Matrix (F)__

The curvature matrix has dimensions `(total_source_pixels, total_source_pixels)` and is given by (WD03):

 ${F}_{ik} = \sum_{j=1}^{J} f_{ij} f_{kj} / \sigma_{j}^2 \, \, .$

Every entry of $F$ is the noise-weighted overlap between the blurred images of two source pixels: $F_{ik}$ sums the
product of columns $i$ and $k$ of $f$ over all image pixels. For $F_{ik}$ to be non-zero, source pixels $i$ and $k$
must share at least one image pixel, which (interpolation aside) only happens because of PSF blurring.

$F$ describes how degenerate pairs of source pixels are with one another: two source pixels whose blurred images
overlap heavily can trade flux between themselves whilst fitting the data almost equally well.
"""
curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix, noise_map=masked_dataset.noise_map
)

plt.imshow(curvature_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__Unregularized Solve__

The inversion seeks the source-pixel fluxes $s$ (a vector with one entry per source pixel) that minimize the
chi-squared:

 $\chi^2 = \sum_{j=1}^{J} \bigg[ \frac{(\sum_{i=1}^{I} s_{i} f_{ij}) + b_{j} - d_{j}}{\sigma_{j}} \bigg]^2$

Setting the derivative of $\chi^2$ with respect to each $s_{i}$ to zero gives the linear system whose solution is
(equation 5 of WD03):

 $s = F^{-1} D$

We can solve this directly with NumPy. (Without regularization the curvature matrix is often singular, so the loop
below adds a tiny value to its diagonal to avoid a `LinAlgError` -- it is a numerical crutch, not part of the
formalism.)
"""
for i in range(curvature_matrix.shape[0]):
    curvature_matrix[i, i] += 1e-8

reconstruction = np.linalg.solve(curvature_matrix, data_vector)

print(reconstruction)

"""
The reconstructed source-pixel fluxes are a noisy, unsmooth mess -- exactly the over-fitting we saw in tutorial 4
when we set the regularization coefficient to zero. The linear inversion is fitting the noise in the data, because
this system of equations is ill-posed: we need a smoothness prior.

__Regularization Matrix (H)__

Regularization adds a linear regularization term $G_{L}$ to the merit function we minimize (equation 11 of WD03):

 $G = \chi^2 + \lambda \, G_{L}$

where $\lambda$ is the `regularization_coefficient` controlling the degree of smoothing. The `Constant` scheme uses
gradient regularization (equation 14 of WD03):

 $G_{L} = \sum_{i}^{I} \sum_{n=1}^{N} [s_{i} - s_{i, n}]^2$

In words: for every source pixel, compare its flux with each of its $N$ neighbours $n$, and penalize solutions where
the differences are large. This is precisely the "smoothness prior" of tutorial 4, now written as an equation.

To fold this into the linear algebra we define the regularization matrix $H$, with
dimensions `(total_source_pixels, total_source_pixels)` (equation 13 of WD03):

 $H_{ik} = \frac{1}{2} \frac{\partial^{2} G_{L}}{\partial s_{i} \partial s_{k}}$

$H$ has the coefficient $\lambda$ folded into it. Its non-zero off-diagonal entries mark pairs of source pixels
which are neighbours and therefore regularized with one another; most entries are zero because most source pixels
are not neighbours.
"""
regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
    coefficient=source_galaxy.pixelization.regularization.coefficient,
    neighbors=mapper.neighbors,
    neighbors_sizes=mapper.neighbors.sizes,
)

plt.imshow(regularization_matrix)
plt.colorbar()
plt.show()
plt.close()

"""
__Source Reconstruction (s)__

$H$ enters the linear system as follows (equation 12 of WD03):

 $s = [F + H]^{-1} D$

We add the two matrices and solve again. The diagonal jitter used above is no longer needed, because $H$ makes the
system well-posed.
"""
curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

print(reconstruction)

"""
The reconstructed fluxes are now smooth and physical: regularization has suppressed the noisy solution and the
reconstruction actually looks like a galaxy, without over-fitting the noise.

__Image Reconstruction__

Using the reconstructed source-pixel fluxes, we map the source back to the image-plane via the blurred mapping
matrix (so the reconstructed image includes PSF blurring) to produce the model image of the lensed source.
"""
mapped_reconstructed_data = (
    al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

mapped_reconstructed_data = al.Array2D(
    values=mapped_reconstructed_data, mask=masked_dataset.mask
)

aplt.plot_array(array=mapped_reconstructed_data, title="Reconstructed Image")

"""
__Likelihood Function__

We now quantify the goodness-of-fit of the source reconstruction, computing the quantity tutorial 4 called the
Bayesian evidence. The log evidence consists of five terms:

 $-2 \, \mathrm{ln} \, \epsilon = \chi^2 + s^{T} H s + \mathrm{ln} \, [ \mathrm{det} (F + H) ] - \mathrm{ln} \, [ \mathrm{det} (H) ] + \sum_{j=1}^{J} \mathrm{ln} \, [2 \pi (\sigma_{j})^2 ] \, .$

This expression was first derived by Suyu et al. 2006 (https://arxiv.org/abs/astro-ph/0601493), equation (19), and
is given in **PyAutoLens** notation by Dye et al. 2008 (https://arxiv.org/abs/0804.4002), equation (5).

We now compute each term in turn.

__Chi Squared__

The first term is the $\chi^2$ statistic from the merit function above, computed as:

 - `model_data` = the reconstructed image of the lensed source (plus the lens light model, zero here).
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residual_map`) ** 2.0
 - `chi_squared` = sum(`chi_squared_map`)

High chi-squared values indicate image pixels the reconstruction fits poorly, lowering the likelihood.
"""
model_image = mapped_reconstructed_data

residual_map = masked_dataset.data - model_image
normalized_residual_map = residual_map / masked_dataset.noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

print(chi_squared)

"""
__Regularization Term__

The second term, $s^{T} H s$, is the $\lambda \, G_{L}$ regularization penalty evaluated at the solution: the summed
difference in flux between all neighbouring source pixels, weighted by the regularization coefficient (which is
already folded into $H$).

Less smooth solutions have larger values of this term and therefore lower likelihoods.
"""
regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

print(regularization_term)

"""
__Complexity Terms__

Up to this point, nothing has justified our choice of `regularization_coefficient=1.0`. We cannot choose it using
the two terms above, because increasing the coefficient smooths the solution more, which *both* worsens the
chi-squared *and* (for a fixed solution) raises the regularization penalty. Optimizing those two terms alone would
drive the coefficient to zero and put us right back at the over-fitted mess.

The two log determinant terms, $\mathrm{ln} \, [ \mathrm{det} (F + H) ]$ and $- \mathrm{ln} \, [ \mathrm{det} (H) ]$,
fix this. Together they measure how *complex* the source reconstruction is -- roughly, how many effective degrees
of freedom the source uses after regularization correlates its pixels -- and penalize more complex solutions.
Lowering the regularization coefficient frees the source to use more of its flexibility, increasing this complexity
penalty.

These terms therefore counteract the chi-squared and regularization terms, so the highest evidence goes to solutions
which fit the data well with the *simplest* source reconstruction. This is the Occam's razor behaviour that
tutorial 4 demonstrated empirically.
"""
log_curvature_reg_matrix_term = np.linalg.slogdet(curvature_reg_matrix)[1]
log_regularization_matrix_term = np.linalg.slogdet(regularization_matrix)[1]

print(log_curvature_reg_matrix_term)
print(log_regularization_matrix_term)

"""
__Noise Normalization Term__

The likelihood function assumes the imaging data consists of independent Gaussian noise in every image pixel, and
the final term is the normalization of those Gaussians: the sum of the log of every noise-map value squared.

Because the noise-map is fixed, this term is constant throughout lens modeling and has no impact on the model
we infer -- it simply normalizes the likelihood.
"""
noise_normalization = float(np.sum(np.log(2 * np.pi * masked_dataset.noise_map**2.0)))

print(noise_normalization)

"""
__Log Evidence__

We can now combine the five terms into the log evidence of the source reconstruction.
"""
log_evidence = float(
    -0.5
    * (
        chi_squared
        + regularization_term
        + log_curvature_reg_matrix_term
        - log_regularization_matrix_term
        + noise_normalization
    )
)

print(log_evidence)

"""
__Fit__

Everything above is what the `FitImaging` object does internally when it fits a tracer whose source galaxy has a
pixelization. We can see this by performing the fit and comparing its `log_evidence` to ours.

The two values are close but not identical, because the real fit improves on our simplified solve in two ways
mentioned along the way: it uses the positive-only solver (tutorial 3), which forbids the negative source-pixel
fluxes our unconstrained `np.linalg.solve` permits, and it zeroes the pixels at the edge of the mesh. Neither
changes the formalism -- the same $f$, $D$, $F$ and $H$ feed a solver with extra constraints.
"""
fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    settings=al.Settings(use_border_relocator=True),
)

print(fit.log_evidence)

aplt.subplot_fit_imaging(fit=fit)

"""
__Wrap Up__

We have walked through the complete linear algebra of a pixelized source reconstruction:

 - The `mapping_matrix` and PSF-blurred mapping matrix $f$, whose columns are the blurred images of each
   source pixel.

 - The data vector $D$ and curvature matrix $F$, the noise-weighted overlaps of those images with the data and with
   each other.

 - The regularization matrix $H$, which encodes the smoothness prior, and the linear solve $s = [F + H]^{-1} D$ for
   the source reconstruction.

 - The five terms of the Suyu et al. 2006 log evidence -- chi-squared, the regularization penalty, the two log
   determinant complexity terms and the noise normalization -- and their Bayesian interpretation as an Occam's
   razor which favours the simplest source reconstruction the data allows.

During lens modeling, this whole calculation is one likelihood evaluation: the non-linear search varies the mass
model (and, later in this chapter, the pixelization and regularization parameters), and each sample triggers the
full ray-trace, solve and evidence computation above.

Two simplifications are worth remembering: real fits use over sampling (each image pixel contributes several
sub-pixel rows to $f$) and **PyAutoLens** uses a positive-only solver for $s$ rather than the unconstrained
`np.linalg.solve` used here (see tutorial 3). The workspace
guide `autolens_workspace/*/imaging/features/pixelization/likelihood_function.ipynb` repeats this walk-through with
lens light included and additional visualization of every step.

In the next tutorial, we return to hands-on territory and look at borders, which deal with the demagnified traced
coordinates whose relocation we performed in a single line above.
"""
