"""
Tutorial 4: Lens Modeling
=========================

Fit lens mass and foreground light alongside a pixelized source using search chaining and a positions likelihood penalty.

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
- Wrap Up

__Lens modelling with a pixelized source__

In the previous tutorials, we supplied the mass model used to simulate the image. This let us concentrate on reconstructing the source.

For an observed lens, we usually do not know the mass distribution. Measuring it is one of the main reasons we perform lens modelling!

Changing the mass model changes where the image pixels trace in the source plane. For every proposed mass model, we must therefore reconstruct the source again and calculate the corresponding log evidence.

Pixelizations are useful because they can describe complicated source structure without requiring us to choose a separate light profile for every component. However, this flexibility also creates a difficulty: **an incorrect mass model can sometimes fit the image by reconstructing an unusual source**.

For example, instead of tracing several arcs back to a common source component, an incorrect model may reconstruct separate source features to explain them. The image fit can look convincing even though the inferred source and lens mass are wrong.

This problem is illustrated by Figure 1 from [Maresca et al. (2021)](https://arxiv.org/abs/2012.04665).

![Demagnified source reconstructions, Figure 1 of Maresca et al. (2021)](https://raw.githubusercontent.com/PyAutoLabs/HowToLens/main/scripts/chapter_3_pixelizations/images/maresca_fig1.png)

Figure 1 from Maresca et al. (2021). Compare the reconstructed sources with their corresponding image fits. The experiments below use our own simulated dataset. See also the [PyAutoLens demagnified-solutions guide](https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html).

We return to the familiar lens with no foreground light. Its true Einstein radius is `1.6` arcsec, giving us a reference against which to compare deliberately incorrect models.
"""

from autolens import jax_wrapper

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "simple__no_lens_light"

al.util.dataset.simulate_if_needed(
    dataset_path,
    simulator="scripts/simulator/no_lens_light.py",
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
We use one sample per image pixel to make the mapping demonstrations easy to follow. Before interpreting a scientific fit, we would check that finer sampling does not materially change the result.

First, we will deliberately give the inversion the wrong Einstein radius and inspect how it attempts to explain the image.

__Giving the inversion an incorrect mass model__

Let’s deliberately give the inversion an Einstein radius of `0.8` arcsec—half the value used to simulate the image.

We keep the mass ellipticity and external shear at their simulation values. We also fix the mesh dimensions and regularization coefficient. This lets us investigate what happens when just the Einstein radius is wrong.

The helper below reconstructs the source for whichever Einstein radius we supply.
"""

mesh_shape = (25, 25)


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
Inspect the reconstructed source alongside the model image and residuals. How much of the observed image can the inversion reproduce despite using the wrong mass model?

With insufficient deflection, the observed arcs do not trace back towards one compact source. They remain more widely separated in the source plane. The inversion can nevertheless assign brightness to those separate regions, reconstructing a source that begins to resemble a less strongly lensed version of the image.

This type of incorrect mass-and-source reconstruction is called a **demagnified solution**. The inversion uses its freedom to describe the source to compensate for an incorrect mass model.

This differs from the demagnified central pixels in the borders tutorial. There, we investigated individual coordinates that stretched the mesh. Here, we have an alternative interpretation of the whole lens system. Border relocation is enabled, but it does not prevent this solution.

---

__Comparing with the correct mass model__

Now restore the Einstein radius to `1.6` arcsec and reconstruct the source again.

Only the Einstein radius changes. We use the same mesh dimensions and regularization coefficient, and display both reconstructions with a common brightness scale.
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
Compare the size and structure of the two sources. With the correct mass model, the multiple images can be explained by a common compact source. With the incorrect mass model, the inversion may need a more extended or fragmented brightness distribution.

Recall our discussion of Bayesian evidence. A close image fit is only part of the comparison: the evidence also accounts for the source reconstruction and its regularization prior.

Let’s compare both chi-squared and log evidence.
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
Which model gives the closer image fit? Which gives the higher evidence? Connect these numbers back to the source and residual plots.

We have evaluated only two mass models at one regularization coefficient. As tutorial 2 showed, the preferred coefficient can change when the reconstruction changes. These numbers are therefore a first comparison, rather than the outcome of a full search over the mass and regularization parameters.

---

__Seeing how the mass model changes the mappings__

The source reconstructions show the result. Let’s now inspect the mappings that produce it.

We select two small groups of image pixels on opposite sides of the lens. We then trace the **same pixels** through each mass model, keeping their colours consistent between plots.

Here we use the tracer directly, so the plots show the ray-traced coordinates before border relocation.
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
    title="Two Selected Image Regions",
)

for label, current_fit in [
    ("Einstein Radius 0.8 Arcsec", incorrect_fit),
    ("Einstein Radius 1.6 Arcsec", correct_fit),
]:
    traced_grid = current_fit.tracer.traced_grid_2d_list_from(grid=image_grid)[1]

    aaplt.plot_grid(
        grid=traced_grid,
        indexes=index_groups,
        title=f"{label}: Ray-Traced Coordinates",
    )

"""
Follow the coloured groups. The mass model determines whether these image regions trace to nearby parts of the source or require brightness in widely separated regions.

The inversion must work with whichever mappings we give it. It cannot change the mass model to bring the groups closer together; it can only change the source brightness values. Exploring different mass models is the job of the non-linear search.

We selected these groups by their image positions, rather than identifying exact multiple images of the same source feature. Their traced coordinates therefore need not coincide perfectly, even under the correct model.

Later, we will identify positions that **do** correspond to the same source feature and require them to trace close together. That gives us a way to steer the search away from demagnified solutions.

First, we will examine why calculating the evidence does not, by itself, ensure that a search finds the correct mass model.

__Why evidence does not remove the search difficulty__

Suppose the correct mass model gives a higher log evidence than our incorrect example. Does that mean a non-linear search will automatically find it?

Unfortunately, no. The physically meaningful solution can occupy a very small region of the non-linear parameter space. The search must find a combination of mass parameters that traces the different arcs back to a common source.

Incorrect mass models occupy a much larger volume of parameter space. With a pixelized source, many of these models can still reproduce much of the image: the inversion compensates for the incorrect deflections by placing source light in separate regions.

This creates a huge landscape of competing local maxima. There are many ways to obtain a reasonably good image fit with an incorrect mass model and an unphysical source reconstruction, while the region containing the correct solution can be narrow and difficult to discover.

A search started with broad priors can therefore spend most of its effort exploring these incorrect solutions and converge on one of them without finding the physically meaningful solution—even when that solution has a higher evidence. For difficult systems, this can happen repeatedly across independent searches.

Efficiently solving the source brightnesses does not solve this search problem. We need to help the non-linear search reach the relevant region of mass-model parameter space before giving the source the full flexibility of a pixelization.

Let’s explore this by varying the Einstein radius while holding the other parameters fixed. At every radius, we reconstruct the source again and calculate the log evidence.
"""

einstein_radii = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

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
        current_fit = fit_at_einstein_radius(einstein_radius=einstein_radius)

    log_evidences.append(float(current_fit.log_evidence))

log_evidences = np.asarray(log_evidences)

if not np.all(np.isfinite(log_evidences)):
    raise ValueError("Inspect non-finite evidence values before interpreting the scan.")

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
ax.legend()

plt.show()

"""
This one-dimensional scan greatly understates the difficulty of the full search. We have fixed the mass ellipticity, shear and regularization coefficient, whereas a real search varies them together. The resulting parameter space can contain many competing maxima that are absent from this simple slice.

This is why we begin with an MGE source and use search chaining. Its restricted geometry makes it harder to compensate for an incorrect mass model by reconstructing arbitrary source structure. We first establish a plausible lensing geometry, then introduce the pixelization and additional constraints to help keep the search away from unphysical solutions.

__Starting with a simpler source model__

We will begin with the multi-Gaussian expansion (MGE) from Chapter 2.

The Gaussians have different fixed widths, allowing their combined light to describe a flexible radial brightness distribution. Their intensities are solved linearly. However, they share a common centre and elliptical shape, so they cannot independently place source components wherever an incorrect mass model requires them.

This makes the MGE a useful starting point for establishing the lensing geometry. It may leave some complicated source structure unexplained, but we can introduce the pixelization once we have inspected the initial fit.

Let’s construct the model.
"""

import autofit as af

mass = af.Model(al.mp.Isothermal)

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
        mask_radius=2.5,
        total_gaussians=20,
        centre_prior_is_uniform=False,
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
Unlike the earlier experiments, the Einstein radius, mass ellipticity and external shear are now free parameters. We keep the mass centre fixed for this introductory example.

The MGE’s common centre and elliptical shape are also free. Its Gaussian widths remain fixed, and the inversion solves for the Gaussian intensities at every non-linear proposal.

Inspect `model_1.info` to see these parameters and their priors. Although we know the simulated Einstein radius is `1.6` arcsec, we have not supplied that answer to the search: it must infer the radius from the image.

---

__Running and inspecting the initial search__

We are now ready to run the first search.

Previously, we supplied parameter values directly to `FitImaging`. Here, Nautilus proposes different combinations of mass and source parameters. For each proposal, the Gaussian intensities are solved linearly and the resulting image is compared with the data.
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
Inspect the model image, residuals and reconstructed source before continuing.

Does the model reproduce the positions and overall shapes of the arcs? Does the source look like a compact galaxy whose multiple images explain the observed configuration? Or has the search found a misleading arrangement?

For this simulated example, we can also compare the inferred Einstein radius with `1.6` arcsec. In an observed lens, we would not have that answer available.

The initial MGE need not reproduce every detail of a complicated source. However, a fit that misses an arc or gives a clearly incorrect lensing configuration is not a suitable starting point for the next search. Passing its parameters forward could steer the pixelized fit towards the same problem.

Finally, remember that the plotted maximum-likelihood fit is one model. Use the result summary and parameter distributions, as in Chapter 2, to assess the constraints and whether further exploration is needed.

Next, we will use corresponding image positions to help the pixelized-source search avoid demagnified solutions.

__Requiring corresponding images to trace together__

In Chapter 2, Tutorial 5, we introduced positions resampling and the **positions likelihood penalty**. We now use that same approach to help the pixelized-source search avoid the unphysical solutions discussed above.

Recall the underlying idea: image positions corresponding to the same source feature should trace close together in the source plane. An incorrect mass model may leave them widely separated, allowing a pixelized reconstruction to explain them by placing unrelated source features in different locations.

The positions likelihood penalty discourages these solutions. When the maximum separation between the traced positions exceeds our threshold, the model receives a likelihood penalty. Models that bring the positions within the threshold receive no positions penalty.

Search chaining provides informed starting priors, while this penalty helps prevent the search from wandering into the much larger regions of parameter space occupied by unphysical reconstructions.

Following Chapter 2, Tutorial 5, we use the initial fit to generate the positions and threshold for the next search. Here, those positions are predicted from the fitted MGE source centre and mass model.
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

traced_positions = initial_fit.tracer.traced_grid_2d_list_from(grid=positions)[1]

aaplt.plot_grid(
    grid=traced_positions,
    title="Predicted Positions Traced to the Source Plane",
)

"""
Inspect the marked positions on the image. They come from our fitted model, so we should check that they sensibly follow the observed multiple-image configuration before using them.

We allow some source-plane separation because the initial mass model is approximate. The next search must have room to improve it.

The `factor` enlarges the separation used to set the threshold from the initial result. The `minimum_threshold` prevents that threshold from becoming smaller than `0.2` arcsec. These are source-plane distances, not distances between the observed images.

The purpose is to exclude grossly inconsistent mappings while allowing useful changes to the mass model. An excessively tight threshold could instead force the second search to remain too close to an imperfect initial fit.

---

__Replacing the analytic source with a pixelization__

We are ready to construct the second model.

We retain the lens mass and external shear from the initial result, using the prior-passing machinery introduced in Chapter 2. We then replace the MGE source with a rectangular pixelization.

The mass parameters remain free. We want the first fit to guide the second search towards the relevant region of parameter space, while allowing the improved source description to change the inferred mass model.

This is why we use `result_1.model`, rather than `result_1.instance`. The model carries free parameters and their updated priors into the next search; the instance would fix them to particular fitted values.
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
The regularization coefficient is now a non-linear parameter, explored alongside the mass and shear parameters. Its logarithmic prior allows the search to explore several orders of magnitude in smoothing strength.

The mesh dimensions remain fixed. Its hundreds of source brightness values are solved by the inversion at every proposal; they do not become hundreds of additional non-linear parameters.

Inspect the priors printed by `model_2.info`. The initial MGE may have left source structure unexplained, potentially shifting its preferred mass model. Prior passing should guide the pixelized search without restricting it so tightly that it cannot correct that initial estimate.

---

__Running the pixelized-source search__

We now combine the pixelized model with the **positions likelihood penalty**, following the same procedure as Chapter 2, Tutorial 5.

Search chaining and the penalty serve complementary purposes. Prior passing guides the search towards the initial mass solution. The positions likelihood penalty discourages proposals that trace corresponding image positions too far apart.

For proposals within the threshold, no positions penalty is applied, and the pixelized imaging log evidence determines their fit statistic. The inversion reconstructs the source using each proposed mass model and regularization coefficient.
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
Compare the source reconstruction and residuals with the initial MGE fit. Then inspect the inferred mass parameters and their uncertainties.

Our simulated source is smooth, so the MGE may already describe it well. We should not expect a dramatic improvement simply because we introduced a pixelization. For irregular sources, its additional spatial freedom becomes more useful.

Let’s also check whether the displayed model brings our selected image positions within the source-plane threshold.
"""

final_source_positions = pixelized_fit.tracer.traced_grid_2d_list_from(grid=positions)[
    1
]

coordinates = np.asarray(final_source_positions)

separations = np.linalg.norm(
    coordinates[:, None, :] - coordinates[None, :, :],
    axis=-1,
)

maximum_separation = float(np.max(separations))

print("Maximum source-plane separation:", maximum_separation, "arcsec")
print("Positions threshold:", positions_likelihood.threshold, "arcsec")
print(
    "Within threshold:",
    maximum_separation <= positions_likelihood.threshold,
)

"""
Passing this check confirms that these selected positions trace sufficiently close together. We still need to inspect the image residuals, parameter constraints and search behaviour to assess the overall fit.

We have now fitted a free mass model and a pixelized source using search chaining. Next, we will include the foreground lens galaxy’s light and examine how it is fitted alongside the source reconstruction.

__Establishing the foreground light model__

So far, our images have contained only the lensed source. In most observed systems, the foreground lens galaxy also contributes light, often overlapping the arcs.

We can fit both contributions together. The predicted image is the sum of the foreground galaxy’s light and the lensed background source, including telescope blurring.

However, this gives the search several things to determine at once: the mass distribution, the foreground light, and the background source. An error in the foreground light model can leave residuals that the pixelized source attempts to explain.

We therefore use search chaining again. Following Chapter 2, we first fit MGEs for both the foreground galaxy and source. This establishes an initial lens-light model and lensing geometry before we introduce a pixelization.

We use a new simulated dataset containing foreground light, so this requires a new initial search.
"""

foreground_dataset_path = Path("dataset") / "imaging" / "lens_sersic"

al.util.dataset.simulate_if_needed(
    foreground_dataset_path,
    simulator="scripts/simulator/lens_sersic.py",
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

"""
The foreground galaxy has a bright centre where its light changes rapidly across individual image pixels. We use finer sampling there when evaluating the light profiles, following the oversampling approach introduced earlier.

For the foreground MGE, we use two Gaussian groups that can have different elliptical shapes. This allows more flexibility than a single group. We fix their centres and the mass centre to the simulation’s origin for this introductory example.
"""

lens_bulge = al.model_util.mge_model_from(
    mask_radius=2.5,
    total_gaussians=20,
    gaussian_per_basis=2,
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
        mask_radius=2.5,
        total_gaussians=20,
        centre_prior_is_uniform=False,
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

print(foreground_model_1.info)

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
The Gaussian widths are fixed, while their intensities are solved linearly. The search explores the remaining free geometry, mass and shear parameters.

Inspect the foreground light model, source and residuals. We want the initial model to separate the foreground galaxy from the arcs sensibly. Significant foreground residuals should be investigated before introducing a more flexible source that might absorb them.

---

__Using that result in the pixelized-source fit__

Now replace the analytic source with a pixelization.

We hold the fitted **foreground MGE geometry** fixed using `.instance`, while keeping the mass and shear free using `.model` and its updated priors.

There is an important distinction here. The MGE contains **linear** Gaussian profiles, so fixing their geometry does not fix their intensities. The next fit solves those intensities together with the pixelized source brightnesses.

We therefore continue to fit the foreground and background light simultaneously. We are not subtracting a fixed foreground image and treating that subtraction as exact.
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

print(foreground_model_2.info)

"""
As in Chapter 2, Tutorial 5, we use the initial result to construct the **positions likelihood penalty**. This helps discourage the unphysical mass-and-source solutions discussed earlier.
"""

foreground_positions_likelihood = foreground_result_1.positions_likelihood_from(
    factor=3.0,
    minimum_threshold=0.2,
)

aplt.plot_array(
    array=foreground_dataset.data,
    positions=foreground_positions_likelihood.positions,
    title="Positions Predicted by the Initial Fit",
)

"""
Check these positions against the observed arcs before running the second search. They are predictions from the initial model, so their usefulness depends on that model having found a sensible lensing geometry.
"""

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
Compare the foreground subtraction, reconstructed source and residuals with the initial fit. The mass parameters, regularization coefficient and linearly solved foreground intensities can all respond to the new source description.

If the fixed foreground geometry leaves significant residuals, we can free its parameters in a later chained search. Holding it fixed is a useful intermediate step, rather than a requirement for the final analysis.

---

__Wrap Up__

We have now fitted a lens mass model alongside a pixelized source, both with and without foreground lens light.

The main difficulty was not solving for the source brightnesses. The inversion does that efficiently for every proposed model. The difficulty was finding the physically meaningful mass model among the many incorrect models that a flexible source can partly compensate for.

We addressed this using the tools introduced in Chapter 2:

- **Search chaining:** begin with an MGE source to establish the lensing geometry, then introduce a pixelization.
- **Prior passing:** keep the mass parameters free while using the initial result to guide the next search.
- **The positions likelihood penalty:** discourage models that trace corresponding image positions too far apart.

When foreground light is present, we also establish its geometry in the initial fit. Its linear intensities can then be solved alongside the pixelized source, allowing both contributions to adjust together.

However, finding the appropriate mass model does not guarantee that our source reconstruction can describe all the observed structure.

What happens if the source has a very compact bright centre, but our mesh places too few pixels there? What if that centre needs weak smoothing while the faint outskirts need strong smoothing?

Even the evidence-preferred constant regularization coefficient must compromise between these demands.

In the next tutorial, **Fit Problems**, we will deliberately make the source more compact while keeping the true mass model fixed. This will reveal limitations of the mesh and regularization themselves, and motivate the adaptive methods that follow.

**🤖 Ask AI:** Why can an incorrect mass model fit the arcs with a pixelized source, and how do search chaining and the positions likelihood penalty address this?
"""
