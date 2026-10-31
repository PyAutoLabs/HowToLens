"""
Tutorial 4: Lens Modeling
=========================

__Contents__

- Lens modelling with a pixelized source
- Giving the inversion an incorrect mass model
- Comparing with the correct mass model
- Seeing how the mass model changes the mappings
- Why evidence does not remove the search difficulty
- Starting with a simpler source model
- Running and inspecting the initial search
- Requiring corresponding images to trace together
- Replacing the analytic source with a pixelization
- Running the pixelized-source search
- Establishing the foreground light model
- Using that result in the pixelized-source fit
"""

"""
__Lens modelling with a pixelized source__

So far, we have reconstructed sources while keeping the lens mass fixed. We could therefore concentrate on the mesh, mappings and regularization without worrying about whether the lens model itself was correct.

In a scientific analysis, the mass distribution is usually one of the things we want to measure. Changing it changes the ray tracing, which changes how the source must be reconstructed to explain the image.

Pixelizations are powerful here because they can describe complicated source structure without requiring us to specify a separate analytic profile for every bright component. The non-linear search explores the lens parameters and regularization settings, while the inversion solves for the source brightnesses.

However, this flexibility comes with a difficulty. An incorrect mass model can sometimes be compensated for by an unusual source reconstruction. A close fit to the image does not, by itself, establish that we have found the right mass distribution.

We will investigate this using the familiar simulated lens with no foreground light. Its true Einstein radius is `1.6` arcsec, giving us a reference against which to compare deliberately incorrect models.

The published example below makes this ambiguity visible before we explore it in code. Compare the reconstructed source structures with their corresponding image fits: a model can describe much of the observed image while assigning it a very different source morphology.

![Demagnified source solutions, Figure 1 of Maresca et al.](https://raw.githubusercontent.com/PyAutoLabs/HowToLens/main/scripts/chapter_3_pixelizations/images/maresca_fig1.png)

Figure 1 from [Maresca et al. (2021)](https://arxiv.org/abs/2012.04665), also discussed in the [PyAutoLens demagnified-solutions guide](https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html). We reproduce this figure to illustrate the ambiguity; the experiments below use our own simulated dataset.
"""

from autolens import jax_wrapper

from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "simple__no_lens_light"

if al.util.dataset.should_simulate(str(dataset_path)):
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
    radius=2.5,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=1)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
We retain one sample per image pixel for the indexed mapping demonstration below. As before, the sampling accuracy would need checking before interpreting a scientific fit.

__Giving the inversion an incorrect mass model__

Let’s begin with a mass model whose Einstein radius is `0.8` arcsec—half the value used in the simulation.

Everything else remains fixed, including the mass ellipticity, external shear, mesh dimensions and regularization coefficient. We then ask the inversion to reconstruct whatever source best satisfies this model’s mappings and smoothness prior.

This is the kind of incorrect mass model that a non-linear search could encounter while exploring a broad parameter range.
"""

# Match mesh size to capped smoke data; normal tutorial runs retain 25 x 25.
mesh_shape = (8, 8) if dataset.shape_native == al.util.dataset.SMALL_DATASETS_SHAPE_NATIVE else (25, 25)

def fit_at_einstein_radius(einstein_radius, coefficient=1.0):
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=einstein_radius,
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

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.RectangularUniform(shape=mesh_shape),
            regularization=al.reg.Constant(coefficient=coefficient),
        ),
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
        fields=[field],
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
    )


incorrect_fit = fit_at_einstein_radius(einstein_radius=0.8)

aplt.subplot_fit_imaging(fit=incorrect_fit)
aplt.subplot_fit_imaging_of_planes(fit=incorrect_fit)

"""
Inspect both the image residuals and the reconstructed source. The experiment is designed to show how an inversion can explain some of the observed structure even when the lens mass is substantially wrong.

Under insufficient deflection, parts of the observed arcs can remain separated in the source plane. Instead of reconstructing one compact galaxy whose light produces multiple images, the inversion can assign brightness to those separate source regions. The source may begin to resemble a less strongly lensed version of the observed image.

Such solutions are often called **demagnified solutions**. This is a different issue from the demagnified central image coordinates discussed in the borders tutorial: here, the problem concerns an alternative mass-and-source interpretation of the image.

Border relocation does not remove this modelling ambiguity.

__Comparing with the correct mass model__

Now repeat the reconstruction with the Einstein radius restored to `1.6` arcsec.

The inversion still has the same mesh dimensions and regularization coefficient. Only the Einstein radius changes. This gives us a controlled comparison of how the mass model affects the source required to explain the image.
"""

correct_fit = fit_at_einstein_radius(einstein_radius=1.6)

source_vmax = max(
    float(np.max(np.asarray(current_fit.inversion.reconstruction)))
    for current_fit in [incorrect_fit, correct_fit]
)

for label, current_fit in [
    ("Einstein Radius 0.8 Arcsec", incorrect_fit),
    ("Einstein Radius 1.6 Arcsec", correct_fit),
]:
    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=label,
    )

    aplt.plot_array(
        array=(dataset.data - current_fit.model_data) / dataset.noise_map,
        title=f"{label}: Normalized Residuals",
    )

"""
Compare the source morphology and its coordinate extent. With the correct mass model, the different images of the source should be brought into a mutually consistent source reconstruction. With an incorrect mass model, the inversion may need more extended or fragmented structure to account for them.

This is where regularization and evidence become important. The inversion’s flexibility does not mean that all source reconstructions are equally preferred. A reconstruction requiring large variations between neighbouring brightnesses is treated differently from one that explains the data with a smoother source.

Let’s compare image agreement and evidence directly.
"""

for label, current_fit in [
    ("Incorrect mass model", incorrect_fit),
    ("Correct mass model", correct_fit),
]:
    normalized_residuals = np.asarray(
        (dataset.data - current_fit.model_data) / dataset.noise_map
    )

    print(label)
    print("  Chi-squared:", float(np.sum(normalized_residuals**2)))
    print("  Log evidence:", float(current_fit.log_evidence))

"""
The expected lesson is that image agreement and source plausibility must be considered together. We should inspect the calculated evidence values rather than assume that the correct model wins by a particular amount.

We have also held the coefficient fixed. A full model search would allow regularization to change, so these two evaluations do not establish the final relative preference after all model parameters have been explored.

__Seeing how the mass model changes the mappings__

The source plots show the outcome, but the mappings help explain why it happens.

Let’s select two small image regions on opposite sides of the lens and follow them into the source plane under each mass model. The image indexes stay the same: we are examining the same measured pixels through two different ray-tracing calculations.

To isolate the physical ray tracing, we use the tracer directly. These plots show the coordinates before border relocation.
"""

image_grid = dataset.grids.pixelization
image_coordinates = np.asarray(image_grid)

index_groups = [
    np.argsort(
        np.sum(
            (image_coordinates - np.asarray(position)) ** 2,
            axis=1,
        )
    )[:8].tolist()
    for position in [(0.0, 1.6), (0.0, -1.6)]
]

aaplt.plot_grid(
    grid=image_grid,
    indexes=index_groups,
    title="The Same Two Image Regions",
)

for label, current_fit in [
    ("Einstein Radius 0.8 Arcsec", incorrect_fit),
    ("Einstein Radius 1.6 Arcsec", correct_fit),
]:
    traced_grid = current_fit.tracer.traced_grid_2d_list_from(
        grid=image_grid
    )[1]

    aaplt.plot_grid(
        grid=traced_grid,
        indexes=index_groups,
        title=f"{label}: Ray-Traced Coordinates",
    )

"""
Look at the relative positions of the coloured groups. Changing the mass model changes whether image regions can be explained by nearby parts of the source or require brightness in more widely separated regions.

These groups were chosen to illustrate the geometry, not identified as exact images of the same source feature. We should therefore not expect their traced coordinates to coincide perfectly, even for the correct model.

For a stronger constraint, we can identify image positions that genuinely correspond to the same source feature and require them to trace close together. We will introduce that idea when preparing the model search.

**🤖 Ask AI:** How can an incorrect mass model still fit an arc if the source reconstruction is allowed to place light in several separate source regions?

__Why evidence does not remove the search difficulty__

Even if the correct mass model has higher evidence than our deliberately incorrect example, the non-linear search still has to find it.

A pixelized source can provide tolerable image fits across a broad range of incorrect mass models. This can create competing regions of parameter space that require substantial exploration. Solving the source brightnesses efficiently does not make the remaining non-linear problem trivial or guarantee that a search finds the relevant solution.

Let’s examine a simple slice through this problem by changing only the Einstein radius. At every value, we reconstruct a new source and calculate its evidence.
"""

einstein_radii = np.array(
    [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
)

known_fits = {
    0.8: incorrect_fit,
    1.6: correct_fit,
}

log_evidences = []

for einstein_radius in einstein_radii:
    einstein_radius = float(einstein_radius)

    if einstein_radius in known_fits:
        current_fit = known_fits[einstein_radius]
    else:
        current_fit = fit_at_einstein_radius(
            einstein_radius=einstein_radius
        )

    log_evidences.append(float(current_fit.log_evidence))

log_evidences = np.asarray(log_evidences)

if not np.all(np.isfinite(log_evidences)):
    raise ValueError(
        "Inspect the non-finite evidence values before interpreting this scan."
    )

fig, ax = plt.subplots()
ax.plot(
    einstein_radii,
    log_evidences - np.max(log_evidences),
    marker="o",
)
ax.axvline(
    1.6,
    color="black",
    linestyle="--",
    label="Simulation Einstein radius",
)
ax.set_xlabel("Einstein radius (arcsec)")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.set_title("A Fixed-Parameter Slice Through Lens-Model Space")
ax.legend()
plt.show()

"""
This scan shows how the evidence behaves along one restricted direction. It is not a posterior distribution or an uncertainty measurement: the other mass parameters and regularization coefficient remain fixed, and we have sampled only a few radii.

In a full search, those other parameters can vary together. An incorrect Einstein radius may be partly compensated for by changes in ellipticity, shear or source regularization, creating possibilities that this one-dimensional experiment cannot show.

A useful strategy is therefore **search chaining**. We begin with a simpler source model to find a plausible region of lens parameter space, then introduce the pixelized source so that the detailed source structure can be reconstructed.

The initial analytic source is a starting point, not the final description of the galaxy. Its purpose is to help establish the lensing geometry without giving the source enough freedom to reproduce almost any image arrangement.

In the next blocks, we will turn that idea into a model-search setup and explain how corresponding image positions can help reject misleading solutions. We will also introduce foreground lens light, which must be modelled alongside the pixelized source.

__Starting with a simpler source model__

The misleading reconstructions we have just examined arise because a pixelized source has enough freedom to compensate for some incorrect mass models.

A useful starting point is therefore a source model with less freedom. We first fit an analytic light profile, using it to establish a plausible lensing geometry. Once that fit has identified a useful region of parameter space, we replace the analytic source with a pixelization.

This is search chaining, introduced in chapter 2. Here, the reason for chaining is particularly important: we want the pixelized reconstruction to improve the description of the source without beginning its search among a wide range of unrelated mass-and-source configurations.

We use the multi-Gaussian expansion (MGE) from Chapter 2 as our baseline analytic source. Its fixed Gaussian widths and linearly solved intensities provide radial flexibility while a common centre and elliptical shape keep the geometry manageable. For a more complex galaxy, this first model may leave source structure unexplained. We would inspect those residuals and the inferred geometry before using it to initialize the next search.
"""

import autofit as af

mass = af.Model(al.mp.Isothermal)

# Keep the mass centre fixed for this introductory experiment.
mass.centre = (0.0, 0.0)
mass.einstein_radius = af.UniformPrior(
    lower_limit=0.5,
    upper_limit=2.5,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=mass,
)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.model_util.mge_model_from(
        mask_radius=2.5, total_gaussians=20, centre_prior_is_uniform=False,
    ),
)

model_1 = af.Collection(
    galaxies=af.Collection(
        lens=lens,
        source=source,
    ),
    fields=field,
)

print(model_1.info)

"""
Unlike our earlier experiments, the Einstein radius, mass ellipticity and external shear are now free. The MGE group’s common centre and elliptical shape are also free. Its Gaussian widths are fixed, and all Gaussian intensities are solved linearly.

Inspect `model_1.info` to see the complete set of free parameters and their priors. The known simulation values provide a check on the result; they are not supplied as the answer to the search.

__Running and inspecting the initial search__

We now perform a model fit. For each proposal, the search changes the non-linear parameters, predicts the image and evaluates its agreement with the data.

This is different from calling `FitImaging` with specified parameter values. The search explores alternatives and produces samples that describe the inferred parameter distribution.
"""

search_1 = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_4",
    name="search_1_analytic_source",
    unique_tag="simple_no_lens_light",
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(
    model=model_1,
    analysis=analysis_1,
)

print(result_1.info)

initial_fit = result_1.max_log_likelihood_fit

aplt.subplot_fit_imaging(fit=initial_fit)
aplt.subplot_fit_imaging_of_planes(fit=initial_fit)

"""
Before continuing, inspect the model image, residuals and source. The purpose of this first fit is to establish a credible starting geometry, so a poor or clearly misleading solution should not simply be passed to the next search.

For this simulated example, we can also compare the recovered Einstein radius with the input value of `1.6` arcsec. In an observed system, that answer would not be available; we would rely on the fit diagnostics, image configuration and parameter constraints.

The maximum-likelihood fit is useful for visualization, but it is only one model. The result summary and samples are needed to assess uncertainty and whether the search adequately explored the parameters.

__Requiring corresponding images to trace together__

Search chaining gives us a useful starting point, but we can also impose a simple geometric check.

If several image positions represent the same source feature, a plausible mass model should trace them close together in the source plane. An incorrect model that reconstructs them as unrelated pieces of source light fails this check.

The `PositionsLH` object applies a penalty when the traced positions are separated by more than a chosen threshold. Models that bring them within the threshold receive no positions penalty.

`PositionsLH` penalizes demagnified solutions whose corresponding image positions fail to trace sufficiently close together.

We can identify corresponding positions manually, but our initial fit also provides a way to estimate them. The following method uses the fitted source centre and mass model to predict its multiple-image positions and constructs a positions constraint for the next search.
"""

positions_likelihood = result_1.positions_likelihood_from(
    factor=3.0,
    minimum_threshold=0.2,
)

positions = positions_likelihood.positions

print("Image-plane positions:")
print(positions)
print(
    "Source-plane separation threshold:",
    positions_likelihood.threshold,
    "arcsec",
)

aplt.plot_array(
    array=dataset.data,
    positions=positions,
    title="Multiple-Image Positions Predicted by the Initial Fit",
)

traced_positions = initial_fit.tracer.traced_grid_2d_list_from(
    grid=positions
)[1]

aaplt.plot_grid(
    grid=traced_positions,
    title="Predicted Image Positions Traced to the Source Plane",
)

"""
These positions are predictions from the initial model, not independent measurements. Check that they correspond sensibly to the observed image configuration before using them.

The threshold should be loose enough to allow the mass model to improve. We are trying to reject grossly inconsistent mappings, not force every subsequent model to reproduce the initial solution exactly. Here, `factor` enlarges the separation estimated from the initial result, while `minimum_threshold` prevents the returned threshold from becoming smaller than `0.2` arcsec.

This positions penalty is a modelling aid rather than a replacement for the image likelihood. It does not by itself establish that the mass model is correct, and an inappropriate set of positions or an overly restrictive threshold can exclude useful solutions.

__Replacing the analytic source with a pixelization__

We are ready to construct the second model.

The lens mass and shear remain free, but their priors are initialized from the first result using the prior-passing machinery introduced in chapter 2. The source profile is replaced by a rectangular mesh and a regularization model.

The distinction between `result_1.model` and `result_1.instance` matters here. Using the model carries free parameters into the next search. Using the instance would instead fix those components to particular fitted values.
"""

regularization = af.Model(al.reg.Constant)
regularization.coefficient = af.LogUniformPrior(
    lower_limit=1.0e-3,
    upper_limit=1.0e3,
)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=regularization,
)

pixelized_source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=pixelization,
)

model_2 = af.Collection(
    galaxies=af.Collection(
        lens=result_1.model.galaxies.lens,
        source=pixelized_source,
    ),
    fields=result_1.model.fields,
)

print(model_2.info)

"""
Inspect the resulting priors rather than assuming that they are suitable simply because they came from an earlier fit. In particular, a restrictive analytic source can bias an initial mass estimate; its result should guide the next search without preventing the mass model from changing.

The mesh shape stays fixed. The regularization coefficient is now a non-linear parameter, explored alongside the mass and shear parameters.

The hundreds of source brightnesses do not appear as hundreds of additional search parameters. For each proposal, the inversion reconstructs those values using the proposed mass model and coefficient.

__Running the pixelized-source search__

The second search combines the pixelized model with the positions constraint.

For each proposed mass model, the analysis checks the positions condition and evaluates the source reconstruction as appropriate. The search can adjust the lens parameters and regularization coefficient, while each inversion solves for the source brightnesses.
"""

analysis_2 = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    settings=al.Settings(use_border_relocator=True),
)

search_2 = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_4",
    name="search_2_pixelized_source",
    unique_tag="simple_no_lens_light",
    n_live=100,
)

result_2 = search_2.fit(
    model=model_2,
    analysis=analysis_2,
)

print(result_2.info)

pixelized_fit = result_2.max_log_likelihood_fit

aplt.subplot_fit_imaging(fit=pixelized_fit)
aplt.subplot_fit_imaging_of_planes(fit=pixelized_fit)

"""
Compare this result with the analytic-source fit. Inspect the source structure and image residuals, then examine how the inferred mass parameters and their uncertainties have changed.

For this simple simulated source, an MGE can already describe its smooth light distribution, so a pixelization need not produce a dramatic improvement. Its value becomes clearer for irregular sources that cannot be described adequately by the initial profile. This example teaches the modelling procedure without requiring that extra complication.

We can also check whether the displayed pixelized model satisfies the positions threshold.
"""

final_source_positions = pixelized_fit.tracer.traced_grid_2d_list_from(
    grid=positions
)[1]

coordinates = np.asarray(final_source_positions)
separations = np.linalg.norm(
    coordinates[:, None, :] - coordinates[None, :, :],
    axis=-1,
)
maximum_separation = float(np.max(separations))

print(
    "Maximum source-plane separation:",
    maximum_separation,
    "arcsec",
)
print(
    "Positions threshold:",
    positions_likelihood.threshold,
    "arcsec",
)
print(
    "Within threshold:",
    maximum_separation <= positions_likelihood.threshold,
)

"""
Passing this check means the selected positions trace sufficiently close together under this model. It does not replace inspecting the residuals, uncertainties or search behaviour.

We have now performed a chained fit with a pixelized source and a free mass model. The next blocks introduce foreground lens light, explaining how its analytic profiles are fitted alongside the source reconstruction and why errors in that light model can affect the inferred source.

__Establishing the foreground light model__

We can model the foreground lens galaxy’s light and reconstruct a pixelized background source within the same fit. The foreground light profile and the lensed source contribute together to the predicted image.

However, fitting everything from the outset gives the search several problems to solve at once. It must establish the mass distribution, separate foreground light from source light, and choose an appropriate source reconstruction.

Search chaining helps us approach these problems in stages. We begin with MGE light models for both galaxies, following Chapter 2. This simpler source description provides a useful starting lens geometry and lets us establish the foreground light model before introducing a pixelization.

Let’s perform that initial fit using a simulated dataset containing foreground lens light. This is a new dataset, so we run a new search rather than reuse the result from our earlier no-lens-light example.
"""

from pathlib import Path
import subprocess
import sys

import autofit as af
import autolens as al
import autolens.plot as aplt

foreground_dataset_path = Path("dataset") / "imaging" / "lens_sersic"

if al.util.dataset.should_simulate(str(foreground_dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/simulator/lens_sersic.py"],
        check=True,
    )

foreground_dataset = al.Imaging.from_fits(
    data_path=foreground_dataset_path / "data.fits",
    noise_map_path=foreground_dataset_path / "noise_map.fits",
    psf_path=foreground_dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=foreground_dataset.shape_native,
    pixel_scales=foreground_dataset.pixel_scales,
    radius=2.5,
)
foreground_dataset = foreground_dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=foreground_dataset.grid,
    sub_size_list=[32, 8, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
foreground_dataset = foreground_dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=2,
)

aplt.subplot_imaging_dataset(dataset=foreground_dataset)

lens_bulge = al.model_util.mge_model_from(
    mask_radius=2.5, total_gaussians=20, gaussian_per_basis=2,
    centre_prior_is_uniform=True,
    sigma_min=foreground_dataset.pixel_scales[0] / 10.0,
)
for gaussian in lens_bulge.profile_list:
    gaussian.centre = (0.0, 0.0)

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)
mass.einstein_radius = af.UniformPrior(
    lower_limit=0.5,
    upper_limit=2.5,
)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=lens_bulge,
    mass=mass,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.model_util.mge_model_from(
        mask_radius=2.5, total_gaussians=20, centre_prior_is_uniform=False,
    ),
)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=af.Model(al.mp.ExternalShear),
)

foreground_model_1 = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

foreground_search_1 = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_4",
    name="foreground_search_1_analytic_source",
    unique_tag="lens_sersic",
    n_live=100,
)

foreground_analysis_1 = al.AnalysisImaging(
    dataset=foreground_dataset,
)

foreground_result_1 = foreground_search_1.fit(
    model=foreground_model_1,
    analysis=foreground_analysis_1,
)

print(foreground_result_1.info)

aplt.subplot_fit_imaging(
    fit=foreground_result_1.max_log_likelihood_fit,
)
aplt.subplot_fit_imaging_of_planes(
    fit=foreground_result_1.max_log_likelihood_fit,
)

"""
We fix the foreground light and mass centres for this introductory example, while allowing their other parameters to vary. The foreground MGE uses two Gaussian groups, which can have different elliptical shapes; the source uses one. Their Gaussian widths are fixed and all Gaussian intensities are solved linearly.

Before continuing, inspect the foreground model and residuals. We want a satisfactory description of the lens light and a plausible source geometry. For an irregular source, the analytic profile may leave structure unexplained, but obvious foreground residuals should not simply be passed into the next stage.

This is a genuine non-linear search, so it will take longer than evaluating a single `FitImaging`.

__Using that result in the pixelized-source fit__

We now replace the analytic source with a pixelization and carry the foreground model into the second search.

The distinction between `.model` and `.instance` is useful here. We use `.instance` to hold the foreground profile’s fitted shape fixed, reducing the number of non-linear parameters the next search must explore. We use `.model` for the mass and shear so that they remain free, with priors passed from the first result.

Because the foreground MGE uses linear Gaussian profiles, fixing its geometry does **not** fix its intensities. The second fit solves all Gaussian intensities alongside the pixelized source brightnesses. The foreground subtraction can therefore adjust without reopening the non-linear Gaussian geometry.

Search chaining also supplies the initial geometry from which we construct the positions constraint. **`PositionsLH` penalizes demagnified solutions whose corresponding image positions fail to trace sufficiently close together.**
"""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=foreground_result_1.instance.galaxies.lens.bulge,
    mass=foreground_result_1.model.galaxies.lens.mass,
)

regularization = af.Model(al.reg.Constant)
regularization.coefficient = af.LogUniformPrior(
    lower_limit=1.0e-3,
    upper_limit=1.0e3,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularUniform(shape=mesh_shape),
        regularization=regularization,
    ),
)

foreground_model_2 = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=foreground_result_1.model.fields,
)

foreground_positions_likelihood = (
    foreground_result_1.positions_likelihood_from(
        factor=3.0,
        minimum_threshold=0.2,
    )
)

aplt.plot_array(
    array=foreground_dataset.data,
    positions=foreground_positions_likelihood.positions,
    title="Positions Predicted by the Initial Foreground-Light Fit",
)

print(foreground_model_2.info)

foreground_analysis_2 = al.AnalysisImaging(
    dataset=foreground_dataset,
    positions_likelihood_list=[foreground_positions_likelihood],
    settings=al.Settings(use_border_relocator=True),
)

foreground_search_2 = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_4",
    name="foreground_search_2_pixelized_source",
    unique_tag="lens_sersic",
    n_live=100,
)

foreground_result_2 = foreground_search_2.fit(
    model=foreground_model_2,
    analysis=foreground_analysis_2,
)

print(foreground_result_2.info)

aplt.subplot_fit_imaging(
    fit=foreground_result_2.max_log_likelihood_fit,
)
aplt.subplot_fit_imaging_of_planes(
    fit=foreground_result_2.max_log_likelihood_fit,
)

"""
Inspect the predicted positions before running the second search, as we did earlier. They come from the initial model and should sensibly represent the multiple-image configuration.

The second search now starts with an established foreground light shape and lens geometry. It can refine the mass and regularization while reconstructing the source, rather than solving the entire problem from broad initial priors.

Compare the two results, paying attention to the foreground subtraction, source reconstruction and residuals. If the fixed foreground shape leaves significant residuals, a later chained search can free its parameters again. Fixing it is a useful intermediate step, not a requirement that the initial light model must remain unchanged throughout the analysis.

__Wrap Up__

This completes the practical chain: fit MGE lens and source light, carry forward the foreground geometry and mass priors, then fit a pixelized source with the positions constraint.
"""
