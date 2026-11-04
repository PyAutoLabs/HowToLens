"""
Tutorial 4: Linear Profiles
===========================

In the previous tutorial, we made a lens model easier to fit by tuning priors, linking parameters or searching more
thoroughly. Each approach involved a trade-off between flexibility, assumptions and computation.

Linear light profiles offer another possibility: reduce the number of parameters explored by the non-linear search
while retaining the same light-profile shapes.

Their intensities are still fitted to the data, but are calculated using linear algebra for each proposed model.
Nautilus can therefore concentrate on the remaining parameters.

We briefly used a linear source profile in Tutorial 1. Here, we examine how this works, then combine many linear
profiles to describe more detailed light distributions without introducing a separate non-linear intensity parameter
for each one.

__Contents__

- **Overview:** Introduce the experiment and the results to inspect.
- **Dataset and Mask:** Load the imaging data and select the region to fit.
- **Linear Light Profiles:** Solve intensities linearly for a given profile geometry.
- **Why This Helps:** Remove intensities from the non-linear parameter search.
- **Model:** Define the components and free parameters of the model.
- **Search and Run Time:** Configure the linear-profile fit and examine its computational cost.
- **Result:** Inspect the inferred parameters and fitted image.
- **Intensities:** Retrieve the solved intensities from the fitted profiles.
- **Visualisation:** Plot profiles with their solved intensities and inspect the fit.
- **Basis:** Combine Gaussian profiles into a flexible light model.
- **A First Gaussian Basis:** Build a fixed-geometry Gaussian basis before fitting its shape.
- **Evaluating the Basis:** Solve the basis intensities at fixed geometry without a new search.
- **Inspecting the Basis Fit:** Assess the residual structure left by circular Gaussians.
- **Lens MGE:** Fit shared geometry for the lens galaxy’s Gaussian profiles.
- **Source MGE:** Give the source an independent Gaussian light model.
- **Combined Model:** Combine lens light, mass, shear and source light.
- **Inspecting the Model:** Check the free parameters and linked Gaussian geometry.
- **Model Fit:** Run the search using the model and analysis.
- **Result:** Inspect the inferred parameters and fitted image.
- **Multi-Gaussian Expansion Benefits:** Explain the flexibility gained from Gaussian light profiles.
- **Benefits Have Costs:** Consider computational cost, interpretation and geometric limits.
- **Positive Intensities:** Require light profiles to contribute non-negative brightness.
- **Other Basis Functions:** Find other ways to expand a galaxy’s light distribution.
- **Wrap Up:** Review the lesson and its connection to the next topic.


__Overview__

We first replace the ordinary lens and source light profiles with their linear equivalents. We will inspect the model,
run the fit and retrieve the intensities calculated during the analysis.

We then combine Gaussian light profiles of different sizes into a **multi-Gaussian expansion**, or **MGE**. Their
summed light can describe a wider range of radial structures than a single profile.

The key idea is to separate the parameters that require non-linear exploration from the intensities that can be solved
together for each proposed model.

__Dataset and Mask__

We use the same `lens_sersic` dataset as before. Look again at the bright central lens galaxy and the lensed source
around it: both contribute light that our model must reproduce.

We retain the 2.5-arcsecond circular mask used in the preceding tutorials. The observations and selected pixels are
unchanged; this time, we change how the model’s light-profile intensities are determined.

The following code loads the dataset, generating it if necessary, applies the mask and plots the image.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/lens_sersic.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

mask_radius = 2.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Linear Light Profiles__

Imagine holding a light profile’s centre, size and shape fixed, then doubling its intensity. Its predicted brightness
doubles at every point, without changing its shape.

The same scaling holds after ray tracing through a fixed mass model and blurring by the PSF. This simple relationship
is what makes the intensity a **linear parameter**.

For a proposed set of profile shapes and mass parameters, we can therefore calculate the intensities that best
reproduce the image using linear algebra. This calculation is often called an **inversion**.

The lens and source intensities are solved together, accounting for the fact that their light overlaps in the observed
image.

The calculation is repeated whenever the non-linear search proposes different parameter values. The intensities are
fitted quantities, not fixed numbers chosen in advance.

__Why This Helps__

With ordinary light profiles, Nautilus must explore intensity alongside parameters such as effective radius and Sérsic
index. Different combinations can produce similar images, making these relationships difficult to explore.

With linear light profiles, each proposed combination of the remaining parameters is evaluated using its best-fitting
intensities, subject to the solver’s constraints. Nautilus no longer needs to search along the intensity dimensions
itself.

In our example, this removes two dimensions from the non-linear search: one intensity for the lens light and one for
the source light. The Sérsic and exponential profile shapes remain available, so we have not simplified the physical
description in the way that linking light and mass did.

This does not remove the underlying relationships between brightness, size and shape. It changes how we handle them
computationally, often making the non-linear exploration easier.

The benefit becomes particularly useful when we combine many light profiles: their intensities can be solved together
rather than adding a new search dimension for every profile.

__Model__

We replace the lens and source light profiles with their equivalents from `al.lp_linear`.

The lens still has a Sérsic light profile and an elliptical isothermal mass profile, alongside external shear. The
source still has an exponential light profile. Their physical roles are unchanged.

Using `al.lp_linear` tells the analysis to solve the two light-profile intensities whenever it evaluates a proposed
model. The other free parameters remain the responsibility of Nautilus.
"""

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp_linear.ExponentialCore,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

"""
Print the model summary and inspect its diagram.

Find the lens and source light profiles. Their intensities no longer have priors for the non-linear search: they are
identified as quantities solved during the fit.

Compare this with the ordinary-profile model from *Practicalities*. There are two fewer non-linear parameters, while
the centres, shapes, sizes and mass parameters remain available for exploration.

This distinction matters when interpreting the parameter count. Fewer non-linear parameters does not mean that we have
stopped fitting the brightness of either galaxy.
"""

print(model.info)

af.ModelPlotter(model).figure()

"""
__Search and Run Time__

We use 100 live points for this example, compared with 200 for the ordinary-profile fit in *Practicalities*. The
simpler non-linear parameter space gives us a reason to try fewer live points, but we must still inspect the result.

Solving the intensities adds work to each likelihood evaluation. In return, the search no longer explores their values
or has to navigate their relationships with the other parameters directly.

The overall run time depends on the balance between these costs. A linear-profile fit is not guaranteed to finish
faster, particularly for a model with only two light profiles. Its advantages become more substantial as we introduce
larger collections of profiles.

This example changes both the treatment of intensities and the live-point count, so it is a practical demonstration
rather than a controlled timing comparison.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_4_linear_profiles",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=2500,
)

analysis = al.AnalysisImaging(dataset=dataset)

result_linear = search.fit(model=model, analysis=analysis)

"""
__Result__

Plot the fitted image and inspect the central lens light and lensed arcs.

Although Nautilus did not explore the intensities, the analysis has adjusted them to reproduce the observed brightness
for each proposed model. Both galaxies should therefore contribute to the final image.

Examine the normalised residuals as before. Changing how intensities are determined does not remove the need to check
whether the model describes the observations.

The result summary describes the parameters explored by the non-linear search. We will retrieve the solved intensities
separately below.
"""

print(result_linear.info)

aplt.subplot_fit_imaging(fit=result_linear.max_log_likelihood_fit)

"""
__Intensities__

The intensities are calculated during the image fit rather than sampled directly by Nautilus.

To inspect their values, retrieve `result_linear.max_log_likelihood_tracer`. This tracer contains ordinary light
profiles with the solved intensities inserted, corresponding to the highest-likelihood fit found by the search.

In this two-galaxy example, the first galaxy is the foreground lens and the second is the background source. Print
their intensities below.

These are the intensities associated with one fitted model. They are not uncertainty estimates: their preferred values
can change as the other profile and mass parameters change.
"""

tracer = result_linear.max_log_likelihood_tracer

print(tracer.galaxies[0].bulge.intensity)
print(tracer.galaxies[1].bulge.intensity)

"""
__Visualisation__

The tracer retrieved above contains ordinary light profiles with their solved intensities. We can therefore use it to
calculate and plot an image.

This shows the light predicted by the tracer before PSF blurring. To assess agreement with the observed image, use the
`FitImaging` plot, which includes that blurring and displays the residuals.

Keep this distinction in mind: the original linear profiles describe components whose intensities still need to be
solved, whereas this tracer contains the intensities determined by the fit.
"""

tracer = result_linear.max_log_likelihood_tracer

aplt.plot_array(
    array=tracer.image_2d_from(grid=dataset.grid),
    title="Model image before PSF blurring",
)

"""
__Basis__

Look again at the foreground lens galaxy. A single Sérsic profile describes its light using smooth ellipses with one
axis ratio and orientation. Real galaxies can have rounder central regions, flatter outer light, or an orientation that
gradually changes with radius. These changes can leave structured residuals when we fit a single Sérsic profile.

An alternative is to add together Gaussian light profiles of different sizes. Narrow Gaussians describe concentrated
central light, while broader Gaussians describe the surrounding emission. Their intensities determine how much each
contributes to the combined brightness distribution.

PyAutoLens groups these profiles into a `Basis`. When they are Gaussians, this description is called a **multi-Gaussian
expansion**, or **MGE**.

The flexibility depends on how we arrange the Gaussians. One group sharing a centre and shape gives a flexible radial
profile. Multiple groups with different shapes and orientations can describe changes in flattening and twists in the
light distribution. Describing genuinely lopsided light requires additional freedom, such as offset centres.

The workspace example uses two groups with different elliptical shapes but a shared centre. These can represent, for
example, rounder central light and a flatter outer component. Their changing contributions with radius let the combined
image vary in shape without assigning independent geometry to every Gaussian.

Crucially, the Gaussian widths can be fixed and their intensities solved by linear algebra. The non-linear search then
explores only the chosen geometric parameters. This gives us a way to describe more complex light distributions while
keeping the non-linear parameter space manageable.

We begin here with a single Gaussian group to understand the method. It will not capture every structure described
above; the workspace demonstrates the more flexible arrangements.

Individual Gaussians need not represent separate physical structures. We interpret their summed light distribution,
rather than identifying each Gaussian as a distinct part of the galaxy.

__A First Gaussian Basis__

We begin with 30 circular Gaussians, all centred at `(0.0, 0.0)` arcseconds. Their sizes are fixed and spaced
logarithmically, giving us a range of narrow and broad components.

The smallest Gaussian has a width of one tenth of the image pixel scale; the largest has a width of 3.0 arcseconds.
These are profile widths, not the extent of the fitted region: our image mask remains at 2.5 arcseconds.

This first example deliberately keeps the geometry simple. We solve the intensities without searching for a better
centre or ellipticity, then inspect what that restriction leaves unexplained.
"""

total_gaussians = 30

sigma_values = np.logspace(
    np.log10(dataset.pixel_scales[0] / 10.0),
    np.log10(3.0),
    total_gaussians,
)

gaussian_list = [
    al.lp_linear.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        sigma=sigma,
    )
    for sigma in sigma_values
]

bulge = al.lp_basis.Basis(profile_list=gaussian_list)

"""
__Evaluating the Basis__

We replace the lens’s Sérsic light profile with this Gaussian basis. We retain the mass, shear and source geometry from
the previous fit.

Creating `FitImaging` solves the linear intensities for this fixed geometry, including the source intensity. It does
not run a new non-linear search.

This lets us examine what the Gaussian basis can achieve before giving it freedom to adjust its centre and shape.
"""

lens = al.Galaxy(
    redshift=0.5,
    bulge=bulge,
    mass=result_linear.instance.galaxies.lens.mass,
)

source = result_linear.instance.galaxies.source

tracer = al.Tracer(
    galaxies=[lens, source],
    fields=result_linear.max_log_likelihood_fields,
)

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Inspecting the Basis Fit__

Plot the fit and examine the foreground galaxy, especially the transition from its bright centre to its fainter outer
light.

The range of Gaussian sizes gives the model flexibility to reproduce this radial brightness distribution. However,
every Gaussian is currently circular, whereas the lens light is slightly elongated.

Look for residual structure around the lens galaxy that could reflect this difference. Adjusting intensities changes
how much light each Gaussian contributes, but cannot make circular Gaussians reproduce an elliptical shape.

We have also fixed their common centre rather than fitted it. In the next step, we let the non-linear search determine
a shared centre and ellipticity while the linear solve continues to determine the individual intensities.

This illustrates two distinct kinds of flexibility: the Gaussian intensities control the radial light distribution,
while their centres and shapes control its geometry. Adding more Gaussians with the same circular shape cannot, by
itself, reproduce an elongated or asymmetric galaxy.

**🤖 Ask AI:** My circular Gaussian basis leaves elongated residuals around the lens galaxy. Why might changing the
Gaussian intensities fail to remove them, and which geometric parameters could I investigate?
"""

aplt.subplot_fit_imaging(fit=fit)

"""
__Lens MGE__

The circular Gaussian basis could adjust its radial brightness distribution, but could not reproduce the slight
elongation of the lens light.

We now let Nautilus fit a common centre and elliptical shape for the 30 Gaussians. Their widths remain fixed, and their
intensities are solved together during each likelihood evaluation.

This gives the lens-light model just four non-linear parameters: two centre coordinates and two elliptical components.

The Gaussians share these parameters, so they move and change shape together. Their individual intensities still allow
the combined brightness to vary with radius. This arrangement remains elliptically symmetric; more complicated geometry
requires the additional Gaussian groups discussed earlier.

We create the centre priors once and assign those same prior objects to every Gaussian. This links their centres,
rather than giving each Gaussian an independent position within the same limits.

We also link every Gaussian’s elliptical components to those of the first Gaussian. Each width is assigned a fixed
numerical value.

Inspect these assignments carefully: giving every Gaussian independent geometry would create a much larger non-linear
search problem.
"""

lens_centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

lens_sigmas = np.logspace(
    np.log10(dataset.pixel_scales[0] / 10.0),
    np.log10(3.0),
    30,
)

lens_gaussians = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(30))

for i, gaussian in enumerate(lens_gaussians):
    gaussian.centre.centre_0 = lens_centre_0
    gaussian.centre.centre_1 = lens_centre_1
    gaussian.ell_comps = lens_gaussians[0].ell_comps
    gaussian.sigma = lens_sigmas[i]

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=list(lens_gaussians),
)

"""
__Source MGE__

We can use the same approach for the source galaxy. Its predicted light is ray-traced through the mass model before
being compared with the observed arcs.

Here, we use 15 Gaussians with widths spanning 0.0001 to 1.0 arcseconds in the source plane. This range includes
compact components and broader emission appropriate for the example; it does not mean the data resolve the smallest
Gaussian individually.

The source Gaussians share a centre and elliptical shape with one another, but these parameters must be independent of
the lens-light MGE. We therefore create new prior objects for the source centre.

This source model can adjust its radial light distribution without making Nautilus explore an intensity or width for
every Gaussian. It still describes one smooth, elliptically symmetric source, rather than arbitrary clumps or multiple
galaxies.
"""

source_centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
source_centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

source_sigmas = np.logspace(-4, 0, 15)

source_gaussians = af.Collection(af.Model(al.lp_linear.Gaussian) for _ in range(15))

for i, gaussian in enumerate(source_gaussians):
    gaussian.centre.centre_0 = source_centre_0
    gaussian.centre.centre_1 = source_centre_1
    gaussian.ell_comps = source_gaussians[0].ell_comps
    gaussian.sigma = source_sigmas[i]

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=list(source_gaussians),
)

"""
__Combined Model__

We now combine the lens-light MGE, the isothermal mass profile, external shear and the source-light MGE.

There are 45 Gaussian light profiles, but only 15 non-linear parameters:

- Four describe the shared lens-light geometry.
- Four describe the shared source-light geometry.
- Five describe the isothermal mass profile.
- Two describe external shear.

The Gaussian intensities are solved during each likelihood evaluation, and their widths are fixed. The number of light
profiles is therefore very different from the number of parameters Nautilus must explore.

Retaining external shear keeps the mass description consistent with our earlier fits.
"""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=al.mp.Isothermal,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=source_bulge,
)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

"""
__Inspecting the Model__

The model summary is long because it lists all 45 Gaussians. Use the diagram alongside it to identify the much smaller
set of independent parameters.

Within each basis, check that the centres and elliptical components are shared, the widths are fixed and the
intensities are solved linearly.

Then check that the lens and source have separate geometry. Accidentally linking them would impose a relationship
between the foreground galaxy and the unlensed source that we do not intend.

This is the main benefit of the construction: many light components can describe the brightness distribution while the
non-linear search handles a comparatively small set of geometric and mass parameters.
"""

print(model.info)

af.ModelPlotter(model).figure()

"""
__Model Fit__

We now fit the lens and source MGEs together with the mass model and external shear.

Nautilus explores their geometric and mass parameters. For every proposed model, the analysis solves the Gaussian
intensities together, accounting for the overlapping lens and lensed-source light.

We use 100 live points as a starting configuration for this example. As before, we must inspect the result rather than
assume that the simpler parameter space guarantees success.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_4_mge",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_basis = search.fit(model=model, analysis=analysis)

"""
__Result__

Inspect the fitted image and normalised residuals. Look separately at the central lens galaxy and the surrounding arcs.

The shared geometry lets each Gaussian group adjust its centre and ellipticity, while its intensities determine the
radial brightness distribution. Compare this fit with the earlier demonstration that fixed the lens Gaussians to be
circular.

The highest-likelihood tracer contains all 45 Gaussian profiles with their solved intensities, so we can plot the
inferred light distribution directly.

Check what the figures show before drawing a conclusion. The model may reproduce this smooth simulated system well, but
that does not demonstrate that the same arrangement can describe every real galaxy.
"""

print(result_basis.info)

tracer = result_basis.max_log_likelihood_tracer

aplt.subplot_tracer(tracer=tracer, grid=dataset.grid)

aplt.subplot_fit_imaging(fit=result_basis.max_log_likelihood_fit)

"""
__Multi-Gaussian Expansion Benefits__

An MGE separates flexibility in the brightness distribution from the number of parameters explored by the non-linear
search.

Our lens and source each use only four non-linear geometric parameters, despite containing 30 and 15 Gaussian profiles
respectively. Their fixed widths cover a range of scales, and the linear solve determines how much light belongs on
each scale.

More flexible arrangements can describe changes in flattening and orientation with radius. The workspace’s
`multi_gaussian_expansion/modeling.py` example demonstrates multiple Gaussian groups with different shapes.

This makes MGE a valuable tool for lens modeling: it can improve the description of galaxy light while keeping the
non-linear exploration manageable.

The approach is also useful for obtaining an initial source model. However, our single-centre source MGE remains a
restricted description, especially when the source contains separate bright knots or irregular features.

__Benefits Have Costs__

Evaluating many Gaussian profiles and solving their intensities adds work to each likelihood calculation. A simpler
non-linear parameter space can compensate for that cost, but the overall speed depends on the dataset and model.

An MGE can also be less straightforward to interpret physically than a Sérsic profile. A Sérsic effective radius and
index directly describe its size and concentration; individual Gaussian widths and intensities are primarily
ingredients of the summed light distribution.

Finally, flexibility depends on the geometry we allow. Adding more Gaussians with one shared centre and shape improves
radial freedom, but cannot create an independently positioned clump.

Inspect residual structure with these restrictions in mind. A detailed-looking model is useful only if it has the
freedom needed to describe the features relevant to the analysis.

__Positive Intensities__

Without a positivity constraint, a linear solver can combine positive and negative Gaussian intensities to reproduce an
image. Large contributions can cancel one another, producing a fitted sum from components that do not represent
physical light emission.

PyAutoLens constrains the Gaussian intensities to be non-negative. A Gaussian can contribute light or have zero
intensity, but cannot subtract light through a negative brightness.

This is particularly useful when many overlapping Gaussians describe the same region. It prevents alternating positive
and negative contributions from being used to manufacture structure.

Non-negative intensities do not guarantee that the overall model is correct. We still need to assess its geometry and
inspect the residuals.

__Other Basis Functions__

Gaussians are not the only functions we can combine to describe galaxy light. PyAutoLens also supports **shapelets**,
which provide another set of spatial patterns for constructing a light distribution.

Different basis functions offer different kinds of flexibility and computational costs. Their usefulness depends on the
structures in the image and the model we want to fit.

You do not need to explore every option to continue HowToLens. The workspace example
`scripts/imaging/features/advanced/shapelets/modeling.py` provides a starting point if your analysis would benefit from
investigating an alternative.

For now, the important distinction is between the individual functions and the freedom of their combined model. A basis
can only reproduce structures allowed by its arrangement and parameters.

__Wrap Up__

Linear light profiles let us solve intensities separately from the non-linear search. We used this first with Sérsic
and exponential profiles, then with Gaussian groups describing the lens and source light.

An MGE gives us flexibility in the brightness distribution without requiring Nautilus to explore an intensity for every
Gaussian. More elaborate arrangements can describe changes in shape and orientation, while our shared-shape example
keeps the geometry simple.

This flexibility still has limits. A source containing several distinct star-forming knots may leave structured
residuals when fitted with one smooth, common-centre MGE. Chapter 3 introduces pixelized sources as a way to describe
more complicated spatial structure.

Before that, *Data and Likelihoods* returns to the observations: which measurements enter the fit, how we handle
contamination, and how the same model–Analysis–search approach applies to the other types of lensing data introduced in
Chapter 1.

We will then bring these tools together in search chaining, balancing model flexibility, reliable fitting and run time.
"""
