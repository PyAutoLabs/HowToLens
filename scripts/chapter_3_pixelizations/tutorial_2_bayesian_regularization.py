"""
Tutorial 2: Bayesian Regularization
===================================

__Contents__

- Interpolation and Bayesian regularization
- Interpolation and our starting reconstruction
- Reducing regularization
- Increasing regularization
- Bayesian evidence
- Exploring the regularization coefficient
- Inspecting the preferred reconstruction
- What is linear, and what requires a search?
- Mesh resolution and regularization belong together
- What we have learned, and what comes next
"""

"""
__Interpolation and Bayesian regularization__

So far, we have:

- Used pixelizations and mappers to connect source mesh points to image pixels.
- Used an inversion to reconstruct the source’s brightness.
- Mapped that reconstruction back into the image, including telescope blurring, to compare it with the data.

Our explanation of how an inversion works has deliberately left some things unexplained. In particular, you will have noticed the regularization input that we passed when creating our pixelization. We used a coefficient of `1.0`, but have not established why that value should give an appropriate reconstruction.

This tutorial explores what regularization does and why we need it. We will change its strength, inspect the resulting source reconstructions, and see why obtaining a close fit to the image is not enough on its own.

First, we will revisit interpolation. It is easy to confuse interpolation with smoothing, because both affect the appearance of the reconstructed source. Understanding the distinction will help us interpret the experiments that follow.

We use the same complex source as tutorial 1. The lens mass remains fixed to the simulation values, so changes in the reconstruction come from our source modelling choices.
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

if al.util.dataset.should_simulate(str(dataset_path)):
    subprocess.run(
        [sys.executable, "scripts/simulator/source_complex.py"],
        check=True,
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
The single sample per image pixel keeps the indexed mapping example straightforward. For scientific modelling, we would also check that our image sampling is sufficiently accurate.

__Interpolation and our starting reconstruction__

Let’s look at our reconstruction from the previous tutorial in more detail.

The `RectangularUniform` mesh describes the source using brightness values at regularly spaced points. A ray-traced image coordinate generally falls between these points. Its predicted brightness is obtained by combining the four surrounding values, with weights determined by its position. This is bilinear interpolation.

Interpolation tells us how to calculate brightness between mesh points, but it does not tell us what those mesh values should be. Four neighbouring values could be similar, or one could be much brighter than the others. Both can be interpolated. In the second case, however, the brightness changes much more sharply across the source.

Regularization influences the brightness values we infer, discouraging variations that are not sufficiently supported by the data. Interpolation then determines how those values contribute between mesh points.

We are going to evaluate several reconstructions, so the function below keeps their common setup in one place. Only the regularization coefficient changes.
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
Here is the interpolation calculation for one image pixel. As in tutorial 1, the mapper supplies the contributing source indexes and their weights. This time, we also have reconstructed brightness values, so we can see how the contributions combine.
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
This is the image prediction before the PSF redistributes light between image pixels. We are inspecting a small part of the calculation, not replacing the inversion’s treatment of telescope blurring.

__Reducing regularization__

Our starting reconstruction uses a regularization coefficient of `1.0`. Let’s reduce it substantially and examine what changes.

We use `0.01` rather than exactly zero, retaining a small smoothness penalty. Keep the lens model, mesh and data unchanged: we want to isolate the effect of weakening regularization.
"""

weak_regularization_fit = fit_with_coefficient(coefficient=0.01)

aplt.subplot_fit_imaging(fit=weak_regularization_fit)
aplt.subplot_fit_imaging_of_planes(fit=weak_regularization_fit)

"""
To understand this comparison, we need to look more closely at what regularization actually does.

When the inversion reconstructs the source, it does not only seek brightness values that reproduce the image. With `Constant` regularization, it also compares neighbouring brightness values on the rectangular source mesh. Interior mesh points have four immediate neighbours. Large differences between their brightnesses contribute a larger penalty.

You can think of this as applying a **smoothness prior** to the reconstructed source. It expresses a preference for neighbouring regions to have similar brightness, unless the image provides enough evidence for them to differ.

The penalty depends on squared differences, so positive and negative differences cannot cancel. The coefficient controls its strength. Reducing that coefficient makes a given amount of variation less costly; increasing it makes the same variation more costly.

In the language of linear algebra, this prior contributes a regularization matrix to the system the inversion solves. We do not need to construct that matrix ourselves to understand its effect.

Without sufficient regularization, the inversion has greater freedom to fit fluctuations in the observed image, including noise. Some of those fluctuations can be explained by adding irregular structure to the source. The image fit may improve even though the inferred source becomes less faithful to the galaxy that produced the data.

Inspect the weakly regularized reconstruction for isolated bright values or rapid changes between neighbouring points. Also look at its faint regions, where the image provides less information about the source.

A few extreme values can dominate the colour scale. We can compare both reconstructions using the starting reconstruction’s brightness range to see the remaining structure more clearly.
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
Values above this plotting limit share the highest colour, so use these figures alongside the original plots rather than treating them as the full brightness range.

The important point is that recovering recognizable source structure does not mean every reconstructed feature is reliable. Genuine structure and noise-fitting can appear in the same reconstruction.

__Increasing regularization__

Now let’s move in the other direction and use a much larger coefficient.

With stronger regularization, the inversion pays a greater penalty for differences between neighbouring brightnesses. This discourages the sharp variations that can fit noise. However, the real source also contains brightness variations: compact components, bright centres and the changes between those regions and their surroundings.

Regularization cannot identify which variations are genuine simply by examining the source. The image fit and the prior must work together. If we make the smoothness preference too strong, the reconstruction can lose genuine structure because reproducing it carries too large a penalty.
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
Compare the bright components with the starting reconstruction. Look for features becoming flatter or merging together, then inspect the corresponding regions of the image residuals.

This illustrates the difficulty in choosing a coefficient. Too little smoothing can introduce structure that explains noise. Too much can remove structure needed to explain the galaxy. Neither the smoothest source nor the closest possible image fit is automatically the reconstruction we want.

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
A lower chi-squared means closer agreement with the image under the assumed noise map. It does not distinguish fitting genuine structure from fitting noise. This is why choosing the coefficient solely by image agreement would favour giving the source too much freedom.

**🤖 Ask AI:** Why might a weakly regularized reconstruction have a lower chi-squared but a less trustworthy source than a more strongly regularized reconstruction?

__Bayesian evidence__

For inversions, we therefore need more than image agreement to choose an appropriate level of regularization.

We use the **Bayesian evidence**, which accounts for both the fit to the image and the source prior. It helps us compare reconstructions that have different amounts of freedom, rather than simply choosing whichever one follows the observed pixel values most closely.

There are two useful parts to this intuition.

First, the reconstruction must explain the image. If strong regularization prevents it from reproducing genuine source structure, the resulting residuals count against that choice. Preferring a smooth source is not useful if that source cannot explain the observed arcs.

Second, allowing a more flexible source comes at a cost. Weak regularization permits many more patterns of brightness across neighbouring mesh points. Some of that extra freedom may be needed to describe the galaxy, but some may only help reproduce the particular noise fluctuations in this image.

Evidence weighs the improvement in fit against the freedom allowed by the prior. An improvement must be sufficient to justify that additional flexibility.

This is also why complexity is not simply the number of source mesh points. Our three examples have exactly the same mesh. Stronger regularization ties neighbouring brightnesses together more closely, reducing their freedom to vary independently. We can therefore change the effective flexibility of the reconstruction without changing the number of mesh points.

Let’s compare the evidence values returned by `FitImaging`.
"""

log_evidences = []

for label, coefficient, current_fit in fit_comparisons:
    log_evidence = float(current_fit.log_evidence)
    log_evidences.append(log_evidence)

    print(
        f"{label}: coefficient={coefficient:g}, "
        f"log evidence={log_evidence:.3f}"
    )

best_index = int(np.argmax(log_evidences))
best_label, best_coefficient, _ = fit_comparisons[best_index]

print(
    f"\nHighest evidence among these examples: "
    f"{best_label}, coefficient={best_coefficient:g}"
)

"""
Compare this ranking with the source plots and the chi-squared values. The highest evidence need not belong to the reconstruction with the lowest chi-squared, because image agreement is only part of the comparison.

We still need to inspect the residuals ourselves. Evidence does not explicitly examine their spatial pattern or guarantee that a preferred model is correct. Structured residuals can reveal shortcomings in the source, lens mass, PSF or other assumptions that this three-coefficient comparison does not address.

For now, we have compared only three choices. The starting value of `1.0` has no special status, and the best of these three is not necessarily the best coefficient overall. Next, we will explore a broader range and see how this choice fits into the wider modelling process.

__Exploring the regularization coefficient__

We have compared weak, moderate and strong regularization, but those three values were chosen to illustrate different behaviours. They do not tell us where the evidence reaches its highest value.

Let’s explore the coefficient more systematically. We will keep the lens mass, mesh resolution, mask and noise map fixed, and reconstruct the source for a sequence of coefficients.

Regularization coefficients can span several orders of magnitude, so we use logarithmically spaced values. This gives us comparable coverage between `0.01` and `0.1`, between `0.1` and `1.0`, and so on.

Every point in this experiment requires a new source reconstruction. We are not taking one reconstructed source and smoothing it by different amounts afterwards. The coefficient changes the problem that the inversion solves.
"""

import matplotlib.pyplot as plt

coefficient_grid = np.logspace(-2, 2, 13)

# Reuse the three reconstructions already calculated.
known_fits = {
    0.01: weak_regularization_fit,
    1.0: fit,
    100.0: strong_regularization_fit,
}

evidence_grid = []

for coefficient in coefficient_grid:
    coefficient = float(coefficient)

    if coefficient in known_fits:
        current_fit = known_fits[coefficient]
    else:
        current_fit = fit_with_coefficient(coefficient=coefficient)

    evidence_grid.append(float(current_fit.log_evidence))

evidence_grid = np.asarray(evidence_grid)

if not np.all(np.isfinite(evidence_grid)):
    raise ValueError(
        "Some evidence values are not finite. Inspect those fits before comparing."
    )

fig, ax = plt.subplots()
ax.semilogx(
    coefficient_grid,
    evidence_grid - np.max(evidence_grid),
    marker="o",
)
ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.set_title("Choosing the Strength of Regularization")
plt.show()

"""
Look for the part of the curve where the evidence is highest. Moving towards weaker regularization gives the source more freedom; moving towards stronger regularization increasingly restricts its variations.

We may find a clear peak, a broad region with similar evidence, or a highest value at one end of the range. These outcomes mean different things. A boundary maximum tells us that this experiment has not bracketed the preferred coefficient—we should extend the range before claiming to have located it.

The plotted points are also discrete samples. Even an interior maximum only identifies the best coefficient we have tried so far.

__Inspecting the preferred reconstruction__

Let’s inspect the reconstruction with the highest evidence in our sampled range.

It is tempting to treat the highest number as the end of the analysis. However, we should connect that number back to the source and image. We want to understand what this choice preserves, what it smooths away, and what the resulting model still fails to explain.

Compare it with the weakly and strongly regularized examples. In particular, examine the distinct bright source components and the faint regions between them. Then look at the image residuals where those components are lensed.
"""

best_index = int(np.argmax(evidence_grid))
preferred_coefficient = float(coefficient_grid[best_index])

if preferred_coefficient in known_fits:
    preferred_fit = known_fits[preferred_coefficient]
else:
    preferred_fit = fit_with_coefficient(
        coefficient=preferred_coefficient
    )

print("Preferred sampled coefficient:", preferred_coefficient)

if best_index in (0, len(coefficient_grid) - 1):
    print(
        "The highest evidence lies at the boundary. "
        "Extend the coefficient range before interpreting this as a peak."
    )

preferred_mapper = preferred_fit.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_mapper(
    mapper=preferred_mapper,
    solution_vector=preferred_fit.inversion.reconstruction,
    zoom_to_brightest=False,
    title="Source at the Preferred Sampled Coefficient",
)

aplt.subplot_fit_imaging(fit=preferred_fit)
aplt.subplot_fit_imaging_of_planes(fit=preferred_fit)

"""
A preferred coefficient does not establish that every visible feature in the source is real. The reconstruction remains conditional on the lens model, source mesh, noise map and telescope blurring.

Nor does a good-looking source excuse coherent residuals in the image. The source and residual plots provide complementary information: one shows what we inferred, while the other shows what that inference leaves unexplained.

**🤖 Ask AI:** If the evidence prefers a particular regularization coefficient but the residuals still follow the shape of an arc, what modelling assumptions should I investigate next?

__What is linear, and what requires a search?__

Before going further, let’s separate two parts of the calculation that are easy to confuse.

For each coefficient we supplied, the inversion solved for the source brightnesses. We did not give a non-linear search hundreds of independent source brightness parameters and ask it to explore their possible combinations. Instead, the source enters the predicted image linearly, allowing these brightnesses to be solved together using linear algebra.

Regularization is included in that solve. The source is not first reconstructed without a prior and then smoothed afterwards. The inferred brightnesses already reflect the balance between fitting the image and satisfying the smoothness prior.

Choosing the regularization coefficient is a different problem. We supplied several coefficients, performed an inversion for each, and compared their evidence values. Our loop is a simple grid search over one quantity.

We can see the difference in scale directly:
"""

source_brightnesses = np.asarray(
    preferred_fit.inversion.reconstruction
)

print("Number of source mesh values:", preferred_mapper.pixels)
print("Number of reconstructed entries:", source_brightnesses.size)
print("Number of coefficients evaluated:", coefficient_grid.size)

"""
The reconstruction contains an entry for each source mesh point in this single-mapper example. Some values may be constrained to zero at the mesh boundary, and regularization couples neighbouring values, so the number of entries is not the number of independently constrained source features.

During lens modelling, the search also changes the mass parameters. Each proposed mass model changes the ray tracing and therefore the mappings between source and image. The source must be reconstructed again for that proposal.

The sequence is therefore:

1. Propose the non-linear parameters, including the lens mass and regularization coefficient.
2. Construct the corresponding mappings.
3. Solve for the source brightnesses.
4. Evaluate how well that model explains the data, accounting for the source prior.

This is what makes pixelized sources practical: the search explores the non-linear model choices, while the inversion handles the many source brightness values.

__Mesh resolution and regularization belong together__

We have held the source mesh at `(25, 25)`. The preferred coefficient we found belongs to that choice of mesh.

Changing the mesh changes the locations and number of source brightness values, their neighbours, and how they contribute to the image. The same numerical coefficient therefore need not give an equivalent amount of smoothing on another mesh.

This matters when testing resolution. If we increase the number of source points but keep the coefficient fixed, any change in the reconstruction reflects both the new resolution and its interaction with the prior. We cannot assume that we have made a fair comparison of the best reconstructions available at each resolution.

Let’s repeat the coefficient experiment for `(15, 15)` and `(35, 35)`, alongside our existing `(25, 25)` results. We extend our helper so the mesh shape is an explicit input.
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


evidence_by_shape = {mesh_shape: evidence_grid}

for shape in ([(6, 6), (10, 10)] if mesh_shape == (8, 8) else [(15, 15), (35, 35)]):
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
ax.set_title("Mesh Resolution and Regularization")
ax.legend()
plt.show()

"""
This experiment performs 26 additional reconstructions, so it takes longer than our earlier three-way comparison.

Look at both the locations and heights of the curves’ maxima. The preferred coefficient may shift as the mesh changes. A finer mesh may help describe source structure, but adding points does not guarantee a higher evidence.

All three curves use the same image pixels and noise model. We also subtract one common reference evidence, preserving differences between the meshes. Subtracting each curve’s own maximum would hide those differences.

As before, a boundary maximum means that the coefficient range needs extending. These are controlled resolution checks, not a demonstration that mesh dimensions should become free parameters in every lens-model search.

__Wrap Up__

We can now give the regularization input a more meaningful interpretation.

The source mesh supplies brightness values that can vary to explain the image. Interpolation defines the brightness between those values. Regularization places a prior on their variation, discouraging unnecessary differences between neighbours.

Too little regularization can allow the reconstruction to explain noise through spurious source structure. Too much can suppress real structure and leave it visible in the image residuals. Evidence helps compare these choices by accounting for both image agreement and the freedom allowed by the source prior.

We also saw that the coefficient is not a universal setting. Its appropriate value depends on the dataset, lens model, mesh and regularization scheme. A value that works for one reconstruction should not simply be carried into another analysis without checking it.

For a final comparison, print the preferred sampled coefficient for each mesh:
"""

for shape, values in sorted(evidence_by_shape.items()):
    index = int(np.argmax(values))
    coefficient = float(coefficient_grid[index])
    boundary = index in (0, len(coefficient_grid) - 1)

    print(
        f"Mesh {shape}: coefficient={coefficient:.4g}, "
        f"log evidence={values[index]:.3f}"
        + (" — boundary maximum; extend the range" if boundary else "")
    )

# Retain the original mesh's preferred fit for subsequent inspection.
fit = preferred_fit

"""
Throughout this tutorial, we fixed the lens mass to the simulation values. We have selected source-reconstruction settings under that assumption; we have not inferred the mass or its uncertainty.

Later, these source reconstructions will sit inside a lens-model search. Before that, the next tutorial examines another practical issue: how the source mesh covers the ray-traced image coordinates, and why a small number of extreme coordinates can affect the resolution available to reconstruct the source.

For readers who want the underlying equations, the workspace’s [pixelization likelihood guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/imaging/features/pixelization/likelihood_function.py) follows the mapping, blurring, regularization and evidence calculations step by step. That derivation complements the experiments here; it is not required to continue using pixelizations.
"""
