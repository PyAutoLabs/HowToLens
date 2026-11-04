"""
Tutorial 2: Bayesian Regularization
===================================

Explore regularization and use Bayesian evidence to choose its strength for a source reconstruction.

__Contents__

- Interpolation and Bayesian regularization
- Interpolation and our starting reconstruction
- Reducing regularization
- Increasing regularization
- Bayesian evidence
- Choosing the coefficient
- Exploring the regularization coefficient
- Inspecting the preferred reconstruction
- What is linear, and what requires a search?
- Mesh resolution and regularization belong together
- Wrap Up

__Interpolation and Bayesian regularization__

So far, we have:

- Used pixelizations and mappers to connect source mesh points to image pixels.
- Used an inversion to reconstruct the source’s brightness.
- Mapped that reconstruction back into the image, including telescope blurring, to compare it with the data.

Our explanation of how an inversion works has deliberately left some things unexplained. In particular, you will
have noticed the regularization input that we passed when creating our pixelization. We used a coefficient of
`1.0`, but have not established why that value should give an appropriate reconstruction.

This tutorial explores what regularization does and why we need it. We will change its strength, inspect the
resulting source reconstructions, and see why obtaining a close fit to the image is not enough on its own.

First, we will revisit interpolation. It is easy to confuse interpolation with smoothing, because both affect the
appearance of the reconstructed source. Understanding the distinction will help us interpret the experiments that
follow.

We use the same complex source as tutorial 1. The lens mass remains fixed to the simulation values, so changes in
the reconstruction come from our source modelling choices.
"""

from autolens import jax_wrapper

from pathlib import Path
import subprocess
import sys

import numpy as np
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

dataset_path = Path("dataset") / "imaging" / "source_complex"

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/source_complex.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.05,
)

mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.1,
    outer_radius=3.2,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=1)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=(0.17647, 0.0),
    ),
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
The single sample per image pixel keeps the indexed mapping example straightforward. For scientific modelling, we
would also check that our image sampling is sufficiently accurate.

__Interpolation and our starting reconstruction__

Let’s look at our reconstruction from the previous tutorial in more detail.

The `RectangularUniform` mesh describes the source using brightness values at regularly spaced points. A ray-traced
image coordinate generally falls between these points. Its predicted brightness is obtained by combining the four
surrounding values, with weights determined by its position. This is bilinear interpolation.

Interpolation tells us how to calculate brightness between mesh points, but it does not tell us what those mesh
values should be. Four neighbouring values could be similar, or one could be much brighter than the others. Both
can be interpolated. In the second case, however, the brightness changes much more sharply across the source.

Regularization influences the brightness values we infer, discouraging variations that are not sufficiently
supported by the data. Interpolation then determines how those values contribute between mesh points.

We are going to evaluate several reconstructions, so the function below keeps their common setup in one place. Only
the regularization coefficient changes.
"""

mesh_shape = (8, 8) if dataset.shape_native == al.util.dataset.SMALL_DATASETS_SHAPE_NATIVE else (25, 25)


def fit_with_coefficient(coefficient):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=mesh_shape),
        regularization=al.reg.Constant(coefficient=coefficient),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(dataset=dataset, tracer=tracer)


fit = fit_with_coefficient(coefficient=1.0)

aplt.subplot_fit_imaging(fit=fit)
aplt.subplot_fit_imaging_of_planes(fit=fit)

"""
Here is the interpolation calculation for one image pixel. As in tutorial 1, the mapper supplies the contributing
source indexes and their weights. This time, we also have reconstructed brightness values, so we can see how the
contributions combine.
"""

mapper = fit.inversion.cls_list_from(al.Mapper)[0]
reconstruction = np.asarray(fit.inversion.reconstruction)

# Select an image pixel near the right-hand side of the lens.
image_coordinates = np.asarray(mapper.image_plane_data_grid)
image_index = int(
    np.argmin(
        np.sum((image_coordinates - np.array([0.0, 1.6])) ** 2, axis=1)
    )
)

weights = np.asarray(mapper.mapping_matrix[image_index])
source_indexes = np.flatnonzero(np.abs(weights) > 1.0e-12)

for source_index in source_indexes:
    brightness = reconstruction[source_index]
    weight = weights[source_index]
    print(
        f"Source index {source_index}: "
        f"brightness={brightness:.5g}, weight={weight:.5g}, "
        f"contribution={brightness * weight:.5g}"
    )

print(
    "Combined brightness before telescope blurring:",
    float(np.dot(weights, reconstruction)),
)

"""
This is the image prediction before the PSF redistributes light between image pixels. We are inspecting a small
part of the calculation, not replacing the inversion’s treatment of telescope blurring.

__Reducing regularization__

Our starting reconstruction uses a regularization coefficient of `1.0`. Let’s reduce it substantially and examine
what changes.

We use `0.01` rather than exactly zero, retaining a small smoothness penalty. Keep the lens model, mesh and data
unchanged: we want to isolate the effect of weakening regularization.
"""

weak_regularization_fit = fit_with_coefficient(coefficient=0.01)

aplt.subplot_fit_imaging(fit=weak_regularization_fit)
aplt.subplot_fit_imaging_of_planes(fit=weak_regularization_fit)

"""
To understand this comparison, we need to look more closely at what regularization actually does.

When the inversion reconstructs the source, it does not only seek brightness values that reproduce the image. With
`Constant` regularization, it also compares neighbouring brightness values on the rectangular source mesh. Interior
mesh points have four immediate neighbours. Large differences between their brightnesses contribute a larger
penalty.

You can think of this as applying a **smoothness prior** to the reconstructed source. It expresses a preference for
neighbouring regions to have similar brightness, unless the image provides enough evidence for them to differ.

The penalty depends on squared differences, so positive and negative differences cannot cancel. The coefficient
controls its strength. Reducing that coefficient makes a given amount of variation less costly; increasing it makes
the same variation more costly.

In the language of linear algebra, this prior contributes a regularization matrix to the system the inversion
solves. We do not need to construct that matrix ourselves to understand its effect.

Without sufficient regularization, the inversion has greater freedom to fit fluctuations in the observed image,
including noise. Some of those fluctuations can be explained by adding irregular structure to the source. The image
fit may improve even though the inferred source becomes less faithful to the galaxy that produced the data.

Inspect the weakly regularized reconstruction for isolated bright values or rapid changes between neighbouring
points. Also look at its faint regions, where the image provides less information about the source.

A few extreme values can dominate the colour scale. We can compare both reconstructions using the starting
reconstruction’s brightness range to see the remaining structure more clearly.
"""

source_vmax = float(np.max(np.asarray(fit.inversion.reconstruction)))

for label, current_fit in [
    ("Starting Reconstruction", fit),
    ("Weak Regularization", weak_regularization_fit),
]:
    current_mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    aaplt.plot_mapper(
        mapper=current_mapper,
        solution_vector=current_fit.inversion.reconstruction,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=label,
    )

"""
Values above this plotting limit share the highest colour, so use these figures alongside the original plots rather
than treating them as the full brightness range.

The important point is that recovering recognizable source structure does not mean every reconstructed feature is
reliable. Genuine structure and noise-fitting can appear in the same reconstruction.

__Increasing regularization__

Now let’s move in the other direction and use a much larger coefficient.

With stronger regularization, the inversion pays a greater penalty for differences between neighbouring
brightnesses. This discourages the sharp variations that can fit noise. However, the real source also contains
brightness variations: compact components, bright centres and the changes between those regions and their
surroundings.

Regularization cannot identify which variations are genuine simply by examining the source. The image fit and the
prior must work together. If we make the smoothness preference too strong, the reconstruction can lose genuine
structure because reproducing it carries too large a penalty.
"""

strong_regularization_fit = fit_with_coefficient(coefficient=100.0)

strong_mapper = strong_regularization_fit.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_mapper(
    mapper=strong_mapper,
    solution_vector=strong_regularization_fit.inversion.reconstruction,
    vmin=0.0,
    vmax=source_vmax,
    zoom_to_brightest=False,
    title="Strong Regularization",
)

aplt.subplot_fit_imaging(fit=strong_regularization_fit)

"""
Compare the bright components with the starting reconstruction. Look for features becoming flatter or merging
together, then inspect the corresponding regions of the image residuals.

This illustrates the difficulty in choosing a coefficient. Too little smoothing can introduce structure that
explains noise. Too much can remove structure needed to explain the galaxy. Neither the smoothest source nor the
closest possible image fit is automatically the reconstruction we want.

We can quantify image agreement using chi-squared: the sum of squared residuals divided by their noise variances.
"""

fit_comparisons = [
    ("Weak", 0.01, weak_regularization_fit),
    ("Starting", 1.0, fit),
    ("Strong", 100.0, strong_regularization_fit),
]

for label, coefficient, current_fit in fit_comparisons:
    normalized_residuals = np.asarray(
        (dataset.data - current_fit.model_data) / dataset.noise_map
    )

    chi_squared = float(np.sum(normalized_residuals**2))
    print(f"{label}: coefficient={coefficient:g}, chi-squared={chi_squared:.3f}")

"""
A lower chi-squared means closer agreement with the image under the assumed noise map. It does not distinguish
fitting genuine structure from fitting noise. This is why choosing the coefficient solely by image agreement would
favour giving the source too much freedom.

**🤖 Ask AI:** Why might a weakly regularized reconstruction have a lower chi-squared but a less trustworthy source
than a more strongly regularized reconstruction?
"""

"""
__Bayesian evidence__

Choosing the regularization coefficient is a Goldilocks problem. Too little regularization gives the source enough freedom to fit the image’s noise. Too much regularization smooths away source structure that we need to explain the image. We want an amount of regularization that is **“just right”**.

But how do we determine what “just right” means without choosing the reconstruction that looks best by eye?

We use the **Bayesian evidence**, which accounts for both how accurately we reconstruct the image and the complexity of the source reconstruction.

With **too much regularization**, neighbouring source pixels are forced towards similar brightness values. This makes it difficult to reconstruct steep changes in brightness, such as a compact source’s bright centre. The source is too smooth, the corresponding image fit leaves residuals, and the evidence decreases.

With **too little regularization**, neighbouring source pixels have greater freedom to vary independently. The image fit improves, but the reconstruction can use this freedom to reproduce noise as additional source structure. The evidence accounts for the extra complexity, so a closer fit to the image does not necessarily give a higher evidence.

The **“just right” coefficient** balances these effects: enough freedom to reconstruct the source structure supported by the data, with sufficient smoothing to discourage fitting noise. We identify it by finding the coefficient that gives the **highest Bayesian evidence**.

Complexity here means more than the number of source pixels. Our examples use exactly the same mesh. Stronger regularization couples neighbouring brightness values more closely, reducing their freedom to vary independently without changing the number of pixels.

In **PyAutoLens**, we obtain the natural logarithm of the evidence using `fit.log_evidence`. Taking the logarithm preserves the ranking, so choosing the largest log evidence is equivalent to choosing the largest evidence.

Let’s compare our three reconstructions.
"""

log_evidences = []

for label, coefficient, current_fit in fit_comparisons:
    log_evidence = float(current_fit.log_evidence)
    log_evidences.append(log_evidence)

    print(
        f"{label}: coefficient={coefficient:g}, "
        f"log evidence={log_evidence:.3f}"
    )

"""
__Choosing the coefficient__

How do we interpret these numbers?

Suppose, purely as an illustration, that coefficients of `0.01`, `1.0` and `100.0` gave log evidences of `4200`, `4250` and `4100`. We would choose `1.0` from these three values because `4250` is the largest. The same rule applies if the values are negative: `-100` is larger than `-150`.

There is no particular log evidence value that we are trying to reach. We compare the values obtained by fitting **the same dataset**, keeping the lens model, mesh, mask and noise map fixed while changing the regularization coefficient.

The code below selects the coefficient with the highest log evidence and prints how far each alternative lies below it.
"""

log_evidences = np.asarray(log_evidences)

best_index = int(np.argmax(log_evidences))
best_label, best_coefficient, best_fit = fit_comparisons[best_index]

for (label, coefficient, _), log_evidence in zip(
    fit_comparisons, log_evidences
):
    print(
        f"{label}: coefficient={coefficient:g}, "
        f"difference from highest log evidence="
        f"{log_evidence - log_evidences[best_index]:.3f}"
    )

print(f"\nPreferred coefficient among these examples: {best_coefficient:g}")

aplt.subplot_fit_imaging(fit=best_fit)
aplt.subplot_fit_imaging_of_planes(fit=best_fit)

"""
Compare this reconstruction with the weakly and strongly regularized examples. How much of the irregular source structure has been smoothed away? Are the bright components still reconstructed accurately? What remains in the image residuals?

We have selected the best of **three coefficients**, rather than found the overall maximum. A coefficient between our trial values could give a higher log evidence. Next, we will evaluate a wider range and plot the evidence against the coefficient to locate its peak.
"""

"""
__Exploring the regularization coefficient__

Our three examples illustrate too little, moderate and too much regularization. However, there is nothing special about the starting coefficient of `1.0`. A value of `0.5`, `2.0`, or something else entirely might give a higher log evidence.

Let’s explore this systematically. We keep the lens mass model, source mesh and dataset fixed, and calculate the log evidence for a range of regularization coefficients.

For every coefficient, we perform a new inversion. We are not taking one reconstructed source and smoothing it afterwards: changing the coefficient changes the brightness values that the inversion solves for.

Regularization coefficients can span several orders of magnitude, so we use logarithmically spaced values. This gives us evenly spaced steps between `0.01` and `0.1`, between `0.1` and `1.0`, and so on.
"""

import matplotlib.pyplot as plt

coefficient_grid = np.logspace(-2, 2, 13)

# Reuse the three reconstructions we have already calculated.
fits_by_coefficient = {
    0.01: weak_regularization_fit,
    1.0: fit,
    100.0: strong_regularization_fit,
}

evidence_grid = []
best_evidence = -np.inf
preferred_fit = None

for coefficient in coefficient_grid:
    coefficient = float(coefficient)

    current_fit = fits_by_coefficient.get(coefficient)
    if current_fit is None:
        current_fit = fit_with_coefficient(coefficient=coefficient)

    evidence = float(current_fit.log_evidence)
    evidence_grid.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        preferred_fit = current_fit

evidence_grid = np.asarray(evidence_grid)

if not np.all(np.isfinite(evidence_grid)):
    raise ValueError("Inspect fits with non-finite evidence before comparing them.")

fig, ax = plt.subplots()

ax.semilogx(
    coefficient_grid,
    evidence_grid - np.max(evidence_grid),
    marker="o",
)

ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")

plt.show()

"""
We have subtracted the highest log evidence from every value. This places the highest point at zero and the other points below it, without changing their ranking.

Look for the peak of the curve. Towards small coefficients, the source has more freedom to fit noise. Towards large coefficients, smoothing increasingly prevents it from reproducing the source structure. The peak identifies our Goldilocks coefficient **among the values we have tested**.

If the highest point lies at either end of the range, we should extend the range in that direction. We have not yet located a peak between weaker and stronger regularization.
"""

"""
__Inspecting the preferred reconstruction__

Let’s take the coefficient with the highest log evidence and look at its source reconstruction and image residuals.

Before looking, think back to our earlier examples. Weak regularization allowed sharp variations between neighbouring source pixels, including in faint regions. Strong regularization suppressed these variations but could also flatten the bright source components.

Does the evidence-preferred reconstruction provide a better balance?
"""

preferred_index = int(np.argmax(evidence_grid))
preferred_coefficient = float(coefficient_grid[preferred_index])

print(f"Preferred sampled coefficient: {preferred_coefficient:.4g}")
print(f"Log evidence: {evidence_grid[preferred_index]:.3f}")

preferred_mapper = preferred_fit.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_mapper(
    mapper=preferred_mapper,
    solution_vector=preferred_fit.inversion.reconstruction,
    vmin=0.0,
    vmax=source_vmax,
    zoom_to_brightest=False,
    title="Source at the Preferred Regularization Coefficient",
)

aplt.subplot_fit_imaging(fit=preferred_fit)
aplt.subplot_fit_imaging_of_planes(fit=preferred_fit)

"""
The source plot uses the same brightness limits as our earlier comparisons. Look at the bright components and the faint regions around them, then examine the corresponding image residuals.

Choosing the highest evidence does not mean the residuals must now be satisfactory. We have found the preferred coefficient for **this particular mesh and regularization scheme**. If the mesh has too few pixels across a compact source component, changing the coefficient cannot supply the missing resolution.

There is also a compromise in applying one coefficient across the whole source. The bright centre may need weaker smoothing to preserve its steep brightness gradient, while the faint outskirts benefit from stronger smoothing. A single coefficient must serve both regions.

Keep these limitations in mind. Our later **Fit Problems** tutorial will show why choosing the coefficient is only part of the solution, motivating meshes and regularization that adapt to the source.
"""

"""
__What is linear, and what requires a search?__

Our reconstruction contains hundreds of source brightness values. Does this mean we have added hundreds of parameters to the non-linear search?

Fortunately, no! This is the same advantage we encountered with linear light profiles in Chapter 2. There, we solved for the intensities of Gaussian components. Here, we solve for the brightness values on the source mesh.

For a fixed lens model, mesh and regularization coefficient, these brightness values enter the predicted image linearly. The inversion therefore solves for them together using linear algebra, including regularization in that calculation.

Choosing the regularization coefficient is a separate, non-linear problem. Each coefficient produces a different reconstruction and log evidence. Our loop explored this by trying a grid of coefficients and selecting the one with the highest evidence.

Let’s compare how many source brightness values we reconstructed with how many coefficients we tried.
"""

source_brightnesses = np.asarray(
    preferred_fit.inversion.reconstruction
)

print("Number of reconstructed brightness values:", source_brightnesses.size)
print("Number of regularization coefficients tried:", coefficient_grid.size)

"""
Every coefficient in our grid required a solution for **all** those brightness values. We did not search through their possible combinations individually.

During lens modelling, we also vary the lens mass parameters. Changing the mass model changes where the image pixels trace in the source plane, so we must perform a new inversion for each proposed model.

The calculation follows this sequence:

1. The non-linear search proposes a lens mass model and regularization coefficient.
2. We ray-trace the image pixels and construct their mappings to the source mesh.
3. The inversion solves for the source brightness values.
4. We calculate the log evidence, which the search uses to assess that proposal.

This is what makes pixelized sources practical. We can reconstruct hundreds of source brightness values without asking the non-linear search to explore hundreds of additional parameters.
"""

"""
__Mesh resolution and regularization belong together__

We have found a preferred regularization coefficient for our current mesh. Would the same coefficient still be preferred if we changed its resolution?

Not necessarily. A finer mesh introduces more brightness values and changes the distances between neighbouring points. It gives the source more freedom to describe small-scale structure, but also more freedom to fit noise. The regularization coefficient must be chosen alongside that resolution.

Think of our Goldilocks analogy again: changing the mesh changes the problem for which we are finding the “just right” amount of smoothing.

Let’s repeat our coefficient scan with coarser and finer meshes. We first extend our helper function so that we can specify the mesh shape.
"""

def fit_with_coefficient(coefficient, mesh_shape=mesh_shape):
    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=mesh_shape),
        regularization=al.reg.Constant(coefficient=coefficient),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(dataset=dataset, tracer=tracer)

"""
We reuse the evidence values for our existing mesh and calculate two additional curves. This takes longer than the previous experiment because each additional mesh requires another complete coefficient scan.
"""

evidence_by_shape = {mesh_shape: evidence_grid}

for shape in [(15, 15), (35, 35)]:
    values = []

    for coefficient in coefficient_grid:
        current_fit = fit_with_coefficient(
            coefficient=float(coefficient),
            mesh_shape=shape,
        )
        values.append(float(current_fit.log_evidence))

    evidence_by_shape[shape] = np.asarray(values)

if not all(
    np.all(np.isfinite(values))
    for values in evidence_by_shape.values()
):
    raise ValueError("Inspect non-finite evidence values before comparing meshes.")

reference_evidence = max(
    float(np.max(values))
    for values in evidence_by_shape.values()
)

fig, ax = plt.subplots()

for shape, values in sorted(evidence_by_shape.items()):
    ax.semilogx(
        coefficient_grid,
        values - reference_evidence,
        marker="o",
        label=f"{shape[0]} × {shape[1]}",
    )

ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.legend()

plt.show()

"""
Look at **where each curve peaks** and **how high that peak is**.

The position tells us which coefficient is preferred for that mesh. The height lets us compare the best sampled reconstructions across the different resolutions. We subtract the same reference value from every curve so that these differences remain visible.

A finer mesh may increase the evidence by resolving source structure that the coarser mesh cannot describe. However, more source pixels do not automatically mean a better reconstruction: the additional freedom must be justified by the image data.

Before drawing a conclusion, check that each peak lies inside the coefficient range. If a curve is still rising at either end, extend its range. Once a peak is bracketed, try more closely spaced coefficients around it to refine the comparison.
"""

"""
__Wrap Up__

We can now explain the regularization coefficient that we first supplied without much discussion in tutorial 1.

It controls how strongly we discourage differences between neighbouring source brightness values. Too little regularization allows the reconstruction to fit noise; too much smooths away structure needed to explain the image. We find the Goldilocks balance by maximizing the Bayesian evidence.

For a final comparison, let’s print the preferred sampled coefficient for each mesh.
"""

for shape, values in sorted(evidence_by_shape.items()):
    index = int(np.argmax(values))
    coefficient = float(coefficient_grid[index])
    boundary = index in (0, len(coefficient_grid) - 1)

    print(
        f"Mesh {shape}: coefficient={coefficient:.4g}, "
        f"log evidence={values[index]:.3f}"
        + (" — extend the coefficient range" if boundary else "")
    )

"""
The preferred coefficient depends on the mesh, as well as the data and lens model. We should therefore infer it as part of an analysis, rather than assume that a value which worked for one reconstruction will work for another.

Throughout this tutorial, we fixed the lens mass model to the values used to simulate the image. Later, we will fit the mass parameters alongside the regularization coefficient, performing an inversion for every model proposed by the non-linear search.

Before that, we need to examine another practical issue. Choosing the number of source pixels does not, by itself, determine our source resolution. We also need to consider **how much area those pixels cover**.

---
"""

"""
For the underlying equations, the workspace’s [pixelization likelihood guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/imaging/features/pixelization/likelihood_function.py) follows the mapping, blurring, regularization and evidence calculations step by step.
"""
