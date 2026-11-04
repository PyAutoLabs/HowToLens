"""
Tutorial 5: Fit Problems
========================

Investigate why compact sources can challenge mesh resolution and constant regularization even when the mass model is correct.

__Contents__

- When the best coefficient is not enough
- Three progressively compact sources
- Simulating the images
- Reconstructing the sources
- Finding the preferred regularization
- Starting with the extended source
- Making the source more compact
- A very compact source
- Why the mesh becomes a limitation
- Why one regularization coefficient becomes a compromise
- Why maximizing the evidence does not remove the compromise
- Wrap Up
"""

"""
__When the best coefficient is not enough__

We have learned how to reconstruct a source, choose its regularization coefficient using Bayesian evidence, and fit the lens mass model.

Does this mean we can now reconstruct any source accurately?

Let’s investigate using three galaxies whose light becomes progressively more concentrated towards their centres. We will simulate their images using exactly the same lens mass model, then supply that correct mass model when reconstructing each source.

This removes the search difficulty from the previous tutorial. If a reconstruction leaves significant residuals, we cannot blame the search for finding an incorrect mass model: we have already supplied the answer.

We will also choose the regularization coefficient separately for each dataset by maximizing its log evidence. This lets us ask a more interesting question than whether an arbitrary coefficient works:

**What happens when the preferred coefficient still does not give a satisfactory reconstruction?**

There are two possible limitations to investigate.

First, the mesh may place too few source pixels across a compact bright component. Changing the regularization coefficient cannot add the missing resolution.

Second, one coefficient controls smoothing across the entire source. The bright centre may require weak smoothing to preserve its steep brightness gradient, while the faint outskirts benefit from stronger smoothing. A single coefficient must compromise between them.

Our Goldilocks coefficient is therefore “just right” **within the mesh and regularization scheme we have provided**. It cannot change those choices for us.

We will work through:

- Three progressively compact sources.
- Simulating their lensed images.
- Selecting a regularization coefficient for each dataset.
- Inspecting the reconstructions and residuals.
- Separating the limitations of mesh resolution and constant regularization.
- Motivating adaptation to the source’s brightness distribution.

---
"""

"""
__Three progressively compact sources__

We use three Sérsic profiles with the same centre and elliptical shape.

For the first, we choose a relatively large effective radius and a Sérsic index of `1.0`. For the next two, we decrease the effective radius and increase the Sérsic index. Their light becomes more concentrated, with a steeper brightness gradient near the centre.
"""

from autolens import jax_wrapper

import numpy as np
import matplotlib.pyplot as plt
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

source_parameters = [
    ("Extended", 0.5, 1.0),
    ("Compact", 0.2, 2.5),
    ("Very compact", 0.1, 4.0),
]

sources = {}

for label, effective_radius, sersic_index in source_parameters:
    sources[label] = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=(0.0, 0.15),
            intensity=0.2,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        ),
    )

"""
The Sérsic `intensity` is the brightness at the effective radius, rather than the central brightness. Keeping it fixed while changing the effective radius and Sérsic index changes both the central brightness and total flux.

These examples therefore represent different source morphologies and signal-to-noise distributions. They are not an experiment in which only the source size changes.

We place all three sources behind the same lens:
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.111111, 0.0),
        einstein_radius=1.6,
    ),
)

true_tracers = {
    label: al.Tracer(galaxies=[lens_galaxy, source])
    for label, source in sources.items()
}

"""
The lens produces exactly the same deflection and magnification fields in all three cases. Changing the source brightness does not change where an image coordinate traces in the source plane.

This distinction will become important: the lensing geometry stays the same, but the spatial resolution needed to describe the source’s light changes.

---
"""

"""
__Simulating the images__

We now simulate the three lensed images using the same image grid and telescope blurring.

To make the comparison easier to interpret, we also use the same Gaussian noise map and the same noise realization. Any change between the images then comes from the source light, rather than a different random draw of the noise.

We first generate images without added Poisson noise, then add our common Gaussian noise explicitly.
"""

simulation_grid = al.Grid2D.uniform(
    shape_native=(150, 150),
    pixel_scales=0.05,
)

simulation_grid = simulation_grid.apply_over_sampling(
    over_sample_size=8,
)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.05,
    pixel_scales=0.05,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
)

noise_sigma = 0.05

noise_values = np.random.default_rng(1).normal(
    loc=0.0,
    scale=noise_sigma,
    size=simulation_grid.shape_native,
)

common_noise_map = al.Array2D.no_mask(
    values=np.full(simulation_grid.shape_native, noise_sigma),
    pixel_scales=0.05,
)

datasets = {}

for label, tracer in true_tracers.items():
    noiseless = simulator.via_tracer_from(
        tracer=tracer,
        grid=simulation_grid,
    )

    datasets[label] = al.Imaging(
        data=al.Array2D.no_mask(
            values=np.asarray(noiseless.data.native) + noise_values,
            pixel_scales=0.05,
        ),
        noise_map=common_noise_map,
        psf=psf,
    )

    print(label)
    aplt.subplot_imaging_dataset(dataset=datasets[label])

"""
Look at how the lensed emission changes. Which parts become narrower or more sharply peaked as the source becomes compact? These are the features our reconstruction will need to explain after accounting for telescope blurring.

We use the same circular mask for all three datasets. Before fitting, check its border against the full images so that we can see whether it includes the lensed emission.
"""

mask = al.Mask2D.circular(
    shape_native=simulation_grid.shape_native,
    pixel_scales=0.05,
    radius=3.0,
)

for label, dataset in datasets.items():
    aplt.plot_array(
        array=dataset.data,
        positions=mask.derive_grid.border,
        title=f"{label}: Image and Mask Border",
    )

datasets = {
    label: dataset.apply_mask(mask=mask).apply_over_sampling(
        over_sample_size_pixelization=2,
    )
    for label, dataset in datasets.items()
}

"""
We now have three datasets with a common lens model, image sampling, PSF, mask and noise realization.

Next, we will reconstruct them using the same rectangular mesh and constant regularization scheme. For each dataset, we will scan the coefficient and inspect its evidence-preferred reconstruction.

We will compare alternative reconstructions **within each dataset**. We will not rank the three different simulated galaxies against one another using their absolute evidence values.
"""

"""
__Reconstructing the sources__

We will reconstruct all three sources using a `25 × 25` rectangular mesh and `Constant` regularization.

We supply the same lens galaxy used to simulate the images. Its mass parameters remain fixed, so the inversion cannot compensate for limitations of the source reconstruction by changing the lens model.

The function below takes a dataset and regularization coefficient, reconstructs the source, and returns the fit.
"""

mesh_shape = (25, 25)


def fit_source(dataset, coefficient):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=mesh_shape),
        regularization=al.reg.Constant(coefficient=coefficient),
    )

    reconstructed_source = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, reconstructed_source],
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        settings=al.Settings(use_border_relocator=True),
    )

"""
The image grid, mask and lens model are identical across our three datasets. The rectangular mesh is therefore constructed from the same mapped coordinates, regardless of how compact the source light becomes.

This is the limitation we want to explore. The source can become smaller and more centrally concentrated, but the mesh does not respond by placing more points across its bright centre.

For now, the only reconstruction setting we vary is the regularization coefficient.

---
"""

"""
__Finding the preferred regularization__

As in tutorial 2, we evaluate a range of coefficients and compare their log evidences. We do this separately for each dataset: a coefficient suitable for the extended source need not be suitable for either compact source.

We first make a broad scan, then sample more closely around its highest point.
"""

coefficient_grid = np.logspace(-3, 3, 25)

preferred_fits = {}
preferred_coefficients = {}

for label, dataset in datasets.items():
    # Retain scalar evidence values, not every full inversion in the scan.
    trial_evidences = {}
    best_fit = None
    best_evidence = -np.inf

    for coefficient in coefficient_grid:
        coefficient = float(coefficient)
        current_fit = fit_source(dataset=dataset, coefficient=coefficient)
        evidence = float(current_fit.log_evidence)
        trial_evidences[coefficient] = evidence
        if evidence > best_evidence:
            best_fit, best_evidence = current_fit, evidence

    values = np.array([trial_evidences[float(c)] for c in coefficient_grid])

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: inspect non-finite evidence values.")

    best_index = int(np.argmax(values))
    if best_index in (0, len(coefficient_grid) - 1):
        raise ValueError(f"{label}: extend the coefficient range before continuing.")

    refined_grid = np.geomspace(
        coefficient_grid[best_index - 1],
        coefficient_grid[best_index + 1],
        15,
    )

    for coefficient in refined_grid:
        coefficient = float(coefficient)
        if coefficient not in trial_evidences:
            current_fit = fit_source(dataset=dataset, coefficient=coefficient)
            evidence = float(current_fit.log_evidence)
            trial_evidences[coefficient] = evidence
            if evidence > best_evidence:
                best_fit, best_evidence = current_fit, evidence

    if not np.all(np.isfinite(list(trial_evidences.values()))):
        raise ValueError(f"{label}: inspect non-finite evidence values.")

    preferred_coefficient = max(trial_evidences, key=trial_evidences.get)
    preferred_coefficients[label] = preferred_coefficient
    preferred_fits[label] = best_fit

    coefficients = np.array(sorted(trial_evidences))
    evidences = np.array([trial_evidences[c] for c in coefficients])

    fig, ax = plt.subplots()
    ax.semilogx(
        coefficients,
        evidences - np.max(evidences),
        marker="o",
    )
    ax.set_xlabel("Regularization coefficient")
    ax.set_ylabel("Log evidence relative to this dataset's maximum")
    ax.set_title(label)
    plt.show()

    print(f"{label}: preferred coefficient = {preferred_coefficient:.4g}")

"""
This performs many inversions, so it will take longer than the earlier three-coefficient comparisons.

Inspect each curve. Have we resolved its peak sufficiently well, or would further sampling be useful? These are the highest-evidence coefficients we have evaluated, rather than mathematically exact optima.

Each plot has its own maximum subtracted. We are examining how regularization affects each dataset, not comparing evidence values between different simulated images.

We can now inspect the reconstructions knowing that we have selected their smoothing strengths systematically.

---
"""

"""
__Starting with the extended source__

Let’s begin with the extended source.

Its brightness varies relatively gradually across the source plane. We therefore expect the rectangular mesh to describe it more easily than the compact sources, whose central brightness changes over smaller distances.

Inspect the reconstruction and image residuals at its preferred coefficient.
"""

extended_fit = preferred_fits["Extended"]
extended_mapper = extended_fit.inversion.cls_list_from(al.Mapper)[0]

print(
    "Extended-source coefficient:",
    preferred_coefficients["Extended"],
)

aplt.subplot_fit_imaging(fit=extended_fit)

aaplt.plot_mapper(
    mapper=extended_mapper,
    solution_vector=extended_fit.inversion.reconstruction,
    mesh_grid=extended_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Extended Source: Preferred Reconstruction",
)

aplt.subplot_fit_imaging_of_planes(fit=extended_fit)

"""
Look at how many mesh points lie across the bright source region. Are its brightness variations represented over several neighbouring points, or concentrated into only one or two?

Then inspect the normalized residual map. A satisfactory reconstruction should leave fluctuations broadly consistent with the simulated noise, without coherent patches following the arcs.

We can summarize the residuals numerically, too.
"""

normalized_residuals = np.asarray(
    (datasets["Extended"].data - extended_fit.model_data)
    / datasets["Extended"].noise_map
)

print(
    "Mean squared normalized residual:",
    float(np.mean(normalized_residuals**2)),
)

"""
A value around one is a useful reference for noise-level residuals, but it does not tell us whether those residuals form a spatial pattern. Keep the residual map alongside this number.

This extended-source fit provides our baseline. Next, we will make the source more compact without giving the mesh any additional resolution, and examine where the reconstruction begins to struggle.
"""

"""
__Making the source more compact__

Now inspect the compact source.

Its effective radius is smaller and its central brightness gradient is steeper. However, we have given the reconstruction exactly the same mesh as before. We have changed the regularization coefficient to the value preferred by this dataset, but we have not added any source pixels.
"""

compact_fit = preferred_fits["Compact"]
compact_mapper = compact_fit.inversion.cls_list_from(al.Mapper)[0]

print(
    "Compact-source coefficient:",
    preferred_coefficients["Compact"],
)

aplt.subplot_fit_imaging(fit=compact_fit)

aaplt.plot_mapper(
    mapper=compact_mapper,
    solution_vector=compact_fit.inversion.reconstruction,
    mesh_grid=compact_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Compact Source: Preferred Reconstruction",
)

aplt.subplot_fit_imaging_of_planes(fit=compact_fit)

"""
Look closely at the bright source centre and the corresponding bright regions of the arcs. Does the reconstruction reproduce their sharp brightness peaks, or do coherent patches remain in the residual map?

The source reconstruction may look broadly reasonable while still missing these details. A small error in the bright centre can produce significant residuals because the image measures that emission with high signal-to-noise.
"""

normalized_residuals = np.asarray(
    (datasets["Compact"].data - compact_fit.model_data)
    / datasets["Compact"].noise_map
)

print(
    "Mean squared normalized residual:",
    float(np.mean(normalized_residuals**2)),
)

"""
If structured residuals remain, remember what we have already controlled: the mass model is correct, and the coefficient was selected using the evidence.

The question is therefore no longer simply “Have we chosen the wrong coefficient?” We need to ask whether our mesh and regularization scheme can represent this source adequately.

---
"""

"""
__A very compact source__

Our final source has an effective radius of just `0.1` arcsec and a Sérsic index of `4.0`. Its light rises steeply towards the centre, placing a greater demand on the reconstruction.

Before looking at the result, consider the two difficulties this creates.

The mesh must have sufficiently closely spaced points to represent the central brightness peak. At the same time, regularization must allow large brightness differences between those points without allowing unnecessary fluctuations across the faint outskirts.

Let’s inspect the evidence-preferred reconstruction.
"""

very_compact_fit = preferred_fits["Very compact"]
very_compact_mapper = very_compact_fit.inversion.cls_list_from(al.Mapper)[0]

print(
    "Very-compact-source coefficient:",
    preferred_coefficients["Very compact"],
)

aplt.subplot_fit_imaging(fit=very_compact_fit)

aaplt.plot_mapper(
    mapper=very_compact_mapper,
    solution_vector=very_compact_fit.inversion.reconstruction,
    mesh_grid=very_compact_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Very Compact Source: Preferred Reconstruction",
)

aplt.subplot_fit_imaging_of_planes(fit=very_compact_fit)

"""
Compare the residuals with those of the extended and compact sources. Pay particular attention to the bright parts of the arcs: are there neighbouring positive and negative residuals suggesting that the model is spreading the light differently from the data?

We can summarize all three fits together.
"""

for label, current_fit in preferred_fits.items():
    dataset = datasets[label]

    normalized_residuals = np.asarray(
        (dataset.data - current_fit.model_data) / dataset.noise_map
    )

    print(
        f"{label}: "
        f"coefficient={preferred_coefficients[label]:.4g}, "
        f"mean squared normalized residual="
        f"{np.mean(normalized_residuals**2):.3f}"
    )

"""
Use these numbers alongside the maps. They summarize the overall residual amplitude, whereas the maps show where the fit struggles.

These galaxies also have different brightness distributions and signal-to-noise ratios. We should not interpret the comparison as a measurement of source-size effects alone. Its purpose is to expose the practical difficulty of reconstructing increasingly concentrated light with an unchanged mesh and one smoothing coefficient.

---
"""

"""
__Why the mesh becomes a limitation__

Let’s examine the first problem: where our source pixels are placed.

The rectangular mesh covers the same source-plane region in all three fits. Its points do not move towards the bright centre when we make the source more compact. Consequently, fewer points lie across the region where most of the central brightness variation occurs.

We can illustrate this by counting mesh points within a circular aperture whose radius equals each source’s effective radius.
"""

for label, effective_radius, _ in source_parameters:
    current_fit = preferred_fits[label]
    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    mesh_coordinates = np.asarray(mapper.source_plane_mesh_grid)

    radii = np.sqrt(
        np.sum(mesh_coordinates**2, axis=1)
    )

    points_within_radius = int(
        np.count_nonzero(radii <= effective_radius)
    )

    print(
        f"{label}: effective radius={effective_radius:.2f} arcsec, "
        f"mesh points inside circular aperture={points_within_radius}"
    )

"""
This is a simple geometric diagnostic. Our sources are elliptical, so the circular aperture is not their exact half-light contour. Nor does the count tell us how many brightness values are independently constrained. It shows how the mesh samples progressively smaller central regions.

Bilinear interpolation can calculate brightness between mesh points, but it cannot introduce an independently adjustable peak between them. If a sharp source component falls within a region sampled by too few points, changing the coefficient cannot supply the missing spatial freedom.

We could increase the resolution everywhere. However, that also adds points across faint regions where the data may provide little information. Those extra brightness values increase the computational work and may require strong regularization.

A more efficient approach is to place additional points where the source needs them.

Magnification offers one useful guide because it tells us where lensing provides finer source-plane resolution. However, the magnification field is identical for our three sources: it does not respond when their light becomes more concentrated. Adapting to magnification alone therefore cannot guarantee enough resolution across every bright source component.

This motivates adapting the mesh to the **source’s brightness distribution**, which we will explore in the next tutorial.

First, there is a second limitation to understand. Even with enough mesh points, should we apply the same regularization coefficient to the bright centre and faint outskirts?
"""

"""
__Why one regularization coefficient becomes a compromise__

Suppose we had enough mesh points to resolve the source’s bright centre. Would that solve the problem?

We would still need to choose how strongly to regularize those points.

Recall that `Constant` regularization penalizes differences between neighbouring source brightness values. One coefficient controls the strength of that penalty across the entire mesh.

However, the brightness gradients differ substantially between the centre and outskirts of our compact source.

Near the centre, the brightness rises steeply. Neighbouring mesh points need to take quite different values to reproduce that rise. Strong regularization discourages those differences and can flatten the reconstructed peak.

In the outskirts, the brightness changes more gradually, or there is little source emission at all. Here, stronger regularization can suppress noise-driven fluctuations without removing the structure needed to explain the image.

Our single coefficient must serve both regions. Let’s compare weaker and stronger regularization with the evidence-preferred value for the very compact source.
"""

preferred_coefficient = preferred_coefficients["Very compact"]

regularization_comparisons = [
    (
        "Weaker Regularization",
        preferred_coefficient / 10.0,
        fit_source(
            dataset=datasets["Very compact"],
            coefficient=preferred_coefficient / 10.0,
        ),
    ),
    (
        "Evidence-Preferred Regularization",
        preferred_coefficient,
        preferred_fits["Very compact"],
    ),
    (
        "Stronger Regularization",
        preferred_coefficient * 10.0,
        fit_source(
            dataset=datasets["Very compact"],
            coefficient=preferred_coefficient * 10.0,
        ),
    ),
]

source_vmax = max(
    float(np.max(np.asarray(current_fit.inversion.reconstruction)))
    for _, _, current_fit in regularization_comparisons
)

for label, coefficient, current_fit in regularization_comparisons:
    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    print(f"{label}: coefficient={coefficient:.4g}")

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=label,
    )

    aplt.subplot_fit_imaging(fit=current_fit)

"""
Compare the bright centre and faint outskirts separately.

Does weaker regularization help reproduce the central peak? Does it also introduce more variation in the faint regions? Does stronger regularization suppress those variations at the cost of leaving residuals around the bright arcs?

These changes need not be equally visible in every dataset. The important point is that changing one coefficient changes the smoothing preference everywhere, even though different source regions may benefit from different strengths.

---
"""

"""
__Why maximizing the evidence does not remove the compromise__

Why does the Bayesian evidence not simply select a coefficient that fixes both regions?

Because we have only given it one coefficient to choose.

The evidence evaluates the complete reconstruction. It balances improvements in the image fit against the source complexity, but it cannot introduce a separate smoothing strength for the centre and outskirts unless our regularization model allows that.

Let’s compare the evidence of the three reconstructions.
"""

comparison_evidences = np.array([
    float(current_fit.log_evidence)
    for _, _, current_fit in regularization_comparisons
])

highest_evidence = float(np.max(comparison_evidences))

for (label, coefficient, _), log_evidence in zip(
    regularization_comparisons,
    comparison_evidences,
):
    print(
        f"{label}: coefficient={coefficient:.4g}, "
        f"log evidence difference={log_evidence - highest_evidence:.3f}"
    )

"""
The coefficient selected by our earlier scan should give the highest evidence of these examples. If either additional trial improves it, we should revisit the scan before treating its maximum as established.

Even at the preferred coefficient, the reconstruction may compromise: smoothing the centre more than we would like while allowing more variation in the outskirts than those regions require.

Our Goldilocks analogy now has a limitation. We have been choosing one amount of smoothing for the whole galaxy, but what is “just right” for the bright centre may not be “just right” for the outskirts.

An **adaptive regularization scheme** gives us more control. It can allow weaker smoothing in bright regions with steep gradients and stronger smoothing in faint regions.

The evidence still decides whether that additional flexibility is useful. We compare the constant and adaptive schemes on the same dataset, allowing their parameters to vary appropriately. Adaptation is a way to improve the available reconstructions, rather than a guarantee that every fit will improve.

---
"""

"""
__Wrap Up__

We began with three sources whose light became progressively more concentrated, while keeping the lens mass model fixed to its true values.

For each dataset, we selected the regularization coefficient using Bayesian evidence. This allowed us to investigate limitations that remain even when the mass is correct and the smoothing strength has been chosen systematically.

We identified two distinct problems:

- **Mesh resolution:** a fixed rectangular mesh can place too few points across a compact bright component. Changing regularization cannot add the missing resolution.
- **Constant regularization:** one smoothing coefficient must compromise between steep central brightness gradients and faint, slowly varying outskirts.

These problems are connected, but solving one does not automatically solve the other. Adding more points around the centre will not help fully if regularization forces their brightnesses to be too similar. Weakening regularization will not resolve a feature that the mesh cannot represent.

The next tutorial brings together the adaptive methods needed to address both.

We will first introduce a Delaunay mesh whose source-plane points are obtained by tracing an image-plane mesh through the lens model. This lets the source sampling respond to the lensing geometry.

We will then use a fitted brightness distribution to guide a Hilbert image mesh, concentrating sampling where the lensed source is bright. Finally, we will introduce adaptive regularization and explain how its smoothing acts across the irregular mesh.

At each stage, we will return to the questions raised here: does the mesh resolve the source’s bright structure, and does regularization preserve that structure while controlling fluctuations elsewhere?

We will finish by using these methods in a lens-model fit, with the search-chaining procedure and positions likelihood penalty already introduced.
"""

"""
**🤖 Ask AI:** How would you distinguish insufficient source-mesh resolution from over-smoothing when compact-source residuals remain at the evidence-preferred coefficient?

__Env__

Keep full-size datasets during automated runs so compact sources remain resolved
and the evidence scans test the same examples as the lesson.

ENV: full_datasets
"""
