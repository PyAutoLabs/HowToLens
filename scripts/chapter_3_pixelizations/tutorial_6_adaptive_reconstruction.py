"""
Tutorial 6: Adaptive Reconstruction
===================================

Combine Delaunay meshes, Hilbert brightness adaptation and adaptive Split regularization in a chained lens-model fit.

__Contents__

- Adaptive source reconstruction
- Returning to the compact-source problem
- From an image mesh to a source mesh
- Constructing the Delaunay source mesh
- Interpolating within a triangle
- Regularization on an irregular mesh
- Building an adapt image
- Placing points with a Hilbert image mesh
- Controlling the brightness weighting
- Keeping resolution in the faint regions
- Reconstructing with the Hilbert mesh
- Comparing overlay and Hilbert reconstructions
- Smoothing on an irregular mesh
- Reconstructing with ConstantSplit
- Allowing different smoothing strengths
- Controlling the transition with `signal_scale`
- Exploring the adaptive coefficients
- Does adaptive regularization address the fit problem?
- Starting the final modelling chain
- Preparing adaptation and the positions likelihood penalty
- Defining the adaptive Delaunay model
- Running the adaptive Delaunay fit
- Inspecting the source and smoothing pattern
- Wrap Up

__Adaptive source reconstruction__

In **Fit Problems**, we identified two limitations of our reconstruction: too few mesh points across compact source structure, and a single regularization coefficient that must compromise between the bright centre and faint outskirts.

We will now introduce methods that address these problems.

Our first step is to replace the rectangular source mesh with a **Delaunay mesh**. Its points need not lie in regular rows and columns, allowing us to vary their spacing across the source.

We then need to decide where to place those points. We will do this in the image plane, using an **image mesh**, and ray-trace its coordinates to obtain the source-mesh points.

This gives us two useful ways to allocate resolution:

- An evenly spaced image-plane overlay produces source-plane sampling that responds to the lens magnification.
- A brightness-adaptive image mesh places more points in bright image regions, directing more resolution towards the source emission.

Finally, we will introduce adaptive regularization, allowing the smoothing strength to vary across the reconstructed source.

These choices work together. Placing more points across a bright component gives the reconstruction enough resolution to describe it; allowing appropriate brightness differences between those points lets it use that resolution.

We will proceed through:

- Constructing an image-plane overlay.
- Ray-tracing it to build a Delaunay source mesh.
- Understanding interpolation and regularization on that mesh.
- Using a fitted image to guide Hilbert brightness adaptation.
- Introducing Split and adaptive regularization.
- Comparing reconstructions on the same dataset.
- Performing a lens-model fit with an adaptive pixelization.

---

__Returning to the compact-source problem__

We return to the very compact source from **Fit Problems**. Keeping the same example lets us examine whether the adaptive methods address the difficulties we have just investigated.

The setup below reproduces that dataset, including its lens model, PSF and Gaussian noise realization. This makes the tutorial runnable independently.
"""

from autolens import jax_wrapper

import numpy as np
import matplotlib.pyplot as plt
import autolens as al
import autolens.plot as aplt
import autoarray.plot as aaplt

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

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.111111, 0.0),
        einstein_radius=1.6,
    ),
)

true_source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.15),
        intensity=0.2,
        effective_radius=0.1,
        sersic_index=4.0,
    ),
)

true_tracer = al.Tracer(
    galaxies=[lens_galaxy, true_source],
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
)

noiseless = simulator.via_tracer_from(
    tracer=true_tracer,
    grid=simulation_grid,
)

noise_sigma = 0.05
noise_values = np.random.default_rng(1).normal(
    loc=0.0,
    scale=noise_sigma,
    size=simulation_grid.shape_native,
)

dataset = al.Imaging(
    data=al.Array2D.no_mask(
        values=np.asarray(noiseless.data.native) + noise_values,
        pixel_scales=0.05,
    ),
    noise_map=al.Array2D.no_mask(
        values=np.full(simulation_grid.shape_native, noise_sigma),
        pixel_scales=0.05,
    ),
    psf=psf,
)

mask = al.Mask2D.circular(
    shape_native=simulation_grid.shape_native,
    pixel_scales=0.05,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(
    over_sample_size_pixelization=2,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
Initially, we keep the true mass model fixed. This lets us inspect how changing the source mesh and regularization affects the reconstruction before combining them with a non-linear mass-model search.

We know the simulated source profile, but we will not use its brightness distribution to tell the adaptive mesh where to place its points. Brightness adaptation will use a fitted model image, as it would for observed data.

---

__From an image mesh to a source mesh__

Previously, we constructed a rectangular mesh directly in the source plane.

For our Delaunay reconstruction, we instead begin by selecting coordinates in the **image plane**. These coordinates form the image mesh. Ray-tracing them through the lens model gives the points from which we construct the source mesh.

The simplest example is an `Overlay`, which lays a coarse, evenly spaced grid over the image and retains the points inside the mask.
"""

image_mesh = al.image_mesh.Overlay(
    shape=(25, 25),
)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)

print(
    "Number of image-mesh points inside the mask:",
    image_plane_mesh_grid.shape[0],
)

aplt.plot_array(
    array=dataset.data,
    grid=image_plane_mesh_grid,
    title="Image-Plane Mesh: Uniform Overlay",
)

"""
These are mesh coordinates, rather than a selection of the only image pixels we will fit. All unmasked data pixels still contribute to the reconstruction.

The overlay’s shape describes the grid before masking. Its retained number of points is therefore smaller than `25 × 25` for our circular mask.

Now trace those mesh coordinates into the source plane.
"""

traced_mesh_grid = true_tracer.traced_grid_2d_list_from(
    grid=image_plane_mesh_grid
)[1]

aaplt.plot_grid(
    grid=traced_mesh_grid,
    title="Image-Mesh Points Traced into the Source Plane",
)

"""
The evenly spaced image-plane points are no longer evenly spaced after ray tracing.

In highly magnified regions, a given source-plane area corresponds to a larger image-plane area. An evenly sampled image mesh therefore tends to provide more traced points per unit source-plane area there. The sampling can also be stretched differently in different directions.

This lets the source mesh respond to the lensing geometry. However, the overlay does not inspect the image brightness: changing our source from extended to compact would leave these mesh coordinates unchanged for a fixed mask and mass model.

That is why magnification-based sampling is useful but does not completely address the problem from the previous tutorial.

Next, we will connect the traced points into a Delaunay triangulation and use it to reconstruct the source. Border relocation will remain enabled when constructing the fit; the plot above shows the direct ray tracing before that treatment.

__Constructing the Delaunay source mesh__

We now connect the traced mesh points into a **Delaunay triangulation**.

Unlike a rectangular mesh, the triangles can have different sizes and orientations. Their vertices follow the distribution of the ray-traced image-mesh points, allowing finer sampling in some source regions than others.

The source brightness values are assigned to the **vertices**, rather than one constant brightness being assigned to each triangle. We will examine how brightness is calculated between the vertices in the next block.

The helper below constructs a Delaunay reconstruction using our image-plane overlay. It also lets us vary the regularization coefficient.
"""

dataset = dataset.apply_sparse_operator()


def fit_delaunay_overlay(coefficient):
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(
            pixels=image_plane_mesh_grid.shape[0],
        ),
        regularization=al.reg.Constant(
            coefficient=coefficient,
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
    )

    adapt_images = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: image_plane_mesh_grid,
        },
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
    )


overlay_fit = fit_delaunay_overlay(coefficient=1.0)
overlay_mapper = overlay_fit.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_mapper(
    mapper=overlay_mapper,
    solution_vector=overlay_fit.inversion.reconstruction,
    mesh_grid=overlay_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Delaunay Reconstruction from an Image-Plane Overlay",
)

aplt.subplot_fit_imaging(fit=overlay_fit)

"""
`AdaptImages` supplies the image-mesh coordinates associated with the pixelized source galaxy. Here, we are supplying an overlay; we have not yet introduced brightness adaptation.

Inspect the mesh points around the reconstructed source. Their spacing follows the lens mapping rather than a regular source-plane lattice.

The coefficient of `1.0` is only a starting value. We should not judge whether this mesh improves the fit until we have explored its regularization.

---

__Interpolating within a triangle__

How does a triangular mesh predict the brightness at a ray-traced coordinate?

Recall our rectangular mesh: bilinear interpolation combined the brightness values at four surrounding mesh points.

For a Delaunay mesh, we locate the triangle containing the coordinate and interpolate between its **three vertices**. Each vertex contributes according to a weight determined by the coordinate’s position within the triangle.

At a vertex, that vertex supplies the brightness. At the midpoint of an edge, the two edge vertices contribute equally. At the triangle’s centroid, all three vertices contribute equally.

The brightness therefore varies continuously across each triangle. It is not a collection of uniformly bright triangular patches.

We can inspect the mapping weights for an image pixel, using the same approach as tutorial 1.
"""

image_index = int(
    np.argmax(np.asarray(dataset.data.slim))
)

mapping_weights = np.asarray(
    overlay_mapper.mapping_matrix[image_index]
)

source_indexes = np.flatnonzero(
    np.abs(mapping_weights) > 1.0e-12
)

reconstruction = np.asarray(
    overlay_fit.inversion.reconstruction
)

for source_index in source_indexes:
    weight = mapping_weights[source_index]
    brightness = reconstruction[source_index]

    print(
        f"Source vertex {source_index}: "
        f"weight={weight:.5g}, "
        f"brightness={brightness:.5g}, "
        f"contribution={weight * brightness:.5g}"
    )

print(
    "Predicted image-pixel brightness before PSF blurring:",
    float(np.dot(mapping_weights, reconstruction)),
)

"""
We have enabled image-pixel oversampling. Each sub-pixel sample is interpolated within its triangle, and those contributions are combined for the image pixel. Consequently, the printed row can contain contributions from more than three source vertices if its samples fall in different triangles.

The PSF then redistributes the predicted light between image pixels, as in our rectangular reconstruction.

Interpolation still does not determine the vertex brightnesses. Those are solved by the inversion, with regularization controlling how freely they can vary.

---

__Regularization on an irregular mesh__

We begin with the same `Constant` regularization scheme used earlier.

For the rectangular mesh, an interior point had four immediate neighbours. On a Delaunay mesh, neighbours are connected by triangle edges. The number of neighbours can vary from one vertex to another; belonging to a triangular mesh does not mean every vertex has only three neighbours.

Regularization penalizes brightness differences between connected vertices. One coefficient controls the strength of this smoothing across the mesh.

The neighbour connections and spacings differ from the rectangular case, so we should not assume that its preferred coefficient transfers to the Delaunay mesh.

Let’s evaluate a range of coefficients.
"""

overlay_coefficients = np.logspace(-3, 3, 13)
overlay_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_delaunay_overlay(coefficient=coefficient)
    evidence = float(current_fit.log_evidence)
    overlay_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        overlay_fit = current_fit

overlay_evidences = np.asarray(overlay_evidences)
if not np.all(np.isfinite(overlay_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(overlay_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the overlay peak.")

overlay_coefficient = float(overlay_coefficients[best_index])

fig, ax = plt.subplots()
ax.semilogx(
    overlay_coefficients,
    overlay_evidences - np.max(overlay_evidences),
    marker="o",
)
ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.set_title("Delaunay Overlay: Constant Regularization")
plt.show()

print("Preferred sampled coefficient:", overlay_coefficient)

aplt.subplot_fit_imaging(fit=overlay_fit)
aplt.subplot_fit_imaging_of_planes(fit=overlay_fit)

"""
As before, inspect the peak and refine the coefficient sampling around it before making a close quantitative comparison.

Now examine the source centre and bright arc residuals. Has the overlay placed enough vertices across the compact component? Does the preferred smoothing reproduce its steep brightness gradient?

We have changed the mesh geometry, but two limitations remain. The overlay does not respond to the source brightness, and one coefficient still controls regularization everywhere.

Next, we will use a fitted image to guide where the image-mesh points are placed, concentrating more sampling on the source emission.

__Building an adapt image__

Our overlay responds to the lensing geometry, but it does not distinguish bright arcs from regions containing only background noise.

To direct more mesh points towards the source emission, we need an image describing where that emission is located. We call this the **adapt image**.

We use the model image from our initial reconstruction. Unlike the observed image, it does not contain the individual noise fluctuations that the model leaves unexplained. It provides a fitted estimate of where the lensed source light lies.
"""

adapt_image = overlay_fit.model_data

aplt.plot_array(
    array=overlay_fit.model_data,
    title="Adapt Image: Fitted Lensed Source Light",
)

aplt.subplot_fit_imaging(fit=overlay_fit)

"""
Compare the adapt image with the data and residuals. Does it locate the bright arcs correctly, even if it does not yet reproduce their sharpest features?

The initial reconstruction need not be perfect to provide useful guidance. However, if it misses a source component entirely, the adapt image may allocate too little resolution to that component. We should therefore inspect the starting fit before using it to guide the next mesh.

Our dataset contains no foreground lens light, so the complete model image represents the lensed source. When foreground light is present, we must use the model contribution from the **source galaxy**, rather than allowing the foreground galaxy’s bright centre to determine where source-mesh points are placed.

We will keep this adapt image fixed while comparing mesh settings. It guides the placement of points; it does not fix the brightness values reconstructed on the new mesh.

---

__Placing points with a Hilbert image mesh__

We now replace the evenly spaced overlay with a **Hilbert image mesh**.

The Hilbert approach orders locations along a space-filling path through the image. Brightness-derived weights determine how points are distributed along that path, allocating more sampling to regions with larger weights.

The path itself is not the source mesh. The resulting image-plane coordinates are ray-traced and connected into a Delaunay mesh, just as our overlay coordinates were.

Let’s construct a brightness-weighted image mesh using the same number of points as our retained overlay. This allows us to examine a change in their distribution without deliberately increasing their total number.
"""

hilbert_pixels = image_plane_mesh_grid.shape[0]

hilbert_image_mesh = al.image_mesh.Hilbert(
    pixels=hilbert_pixels,
    weight_power=1.0,
    weight_floor=0.05,
)

hilbert_image_plane_mesh_grid = (
    hilbert_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=adapt_image,
    )
)

for label, mesh_grid in [
    ("Uniform Overlay", image_plane_mesh_grid),
    ("Brightness-Weighted Hilbert", hilbert_image_plane_mesh_grid),
]:
    print(f"{label}: {mesh_grid.shape[0]} image-mesh points")

    aplt.plot_array(
        array=dataset.data,
        grid=mesh_grid,
        title=label,
    )

"""
Compare the spacing of the points across the bright arcs and the surrounding background.

The overlay distributes its points evenly across the image mask. The Hilbert mesh uses the adapt image to concentrate more points on the emission, while the non-zero weight floor retains some sampling in faint regions.

Now trace the Hilbert points into the source plane.
"""

hilbert_traced_mesh_grid = true_tracer.traced_grid_2d_list_from(
    grid=hilbert_image_plane_mesh_grid
)[1]

aaplt.plot_grid(
    grid=hilbert_traced_mesh_grid,
    title="Brightness-Weighted Points Traced into the Source Plane",
)

"""
The point distribution now responds to both the image brightness and the lens mapping.

This addresses the distinction raised in **Fit Problems**: the magnification field remains unchanged when the source becomes more compact, but the brightness distribution changes. A brightness-weighted image mesh can respond by placing more points across the compact emission.

Next, let’s examine how its weights are calculated.

---

__Controlling the brightness weighting__

Two parameters control how the adapt image becomes a weight map: `weight_power` and `weight_floor`.

The calculation first takes the absolute adapt-image values and divides them by their maximum. For our positive model image, this gives normalized brightnesses between zero and one.

It then raises those values to `weight_power` and replaces any weight below `weight_floor` with that minimum value.

Consider three powers:

- `0.0` gives equal weights, removing the brightness preference.
- `1.0` retains the normalized brightness weighting.
- A value greater than `1.0` increases the contrast between bright and faint regions.

For example, a normalized brightness of `0.5` becomes `0.03125` when raised to the fifth power, whereas the brightest value remains `1.0`. Larger powers therefore concentrate the sampling more strongly on the brightest emission.

Let’s inspect the weight maps and resulting point distributions together.
"""

for weight_power in [0.0, 1.0, 5.0]:
    trial_image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels,
        weight_power=weight_power,
        weight_floor=0.05,
    )

    weights = trial_image_mesh.weight_map_from(
        adapt_data=adapt_image,
    )

    weight_map = al.Array2D(
        values=weights,
        mask=dataset.mask,
    )

    trial_mesh_grid = trial_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=adapt_image,
    )

    aplt.plot_array(
        array=weight_map,
        vmin=0.0,
        vmax=1.0,
        title=f"Weight Map: Power {weight_power:g}",
    )

    aplt.plot_array(
        array=dataset.data,
        grid=trial_mesh_grid,
        title=f"Image-Mesh Points: Power {weight_power:g}",
    )

"""
Follow how increasing the power changes the allocation of points. Does the mesh resolve the brightest parts more densely? How much sampling remains across the fainter emission?

A higher power is not automatically better. Concentrating too many points on the brightest peaks can leave too few to describe the rest of the source.

The weight floor helps moderate this effect. Here, every location has a weight of at least `0.05`. This is a minimum weight, not a statement that five per cent of the mesh points must lie in the background.

Next, we will vary that floor and reconstruct the source with the Hilbert mesh, checking the residuals and evidence rather than judging the mesh by its appearance alone.

__Keeping resolution in the faint regions__

Increasing `weight_power` concentrates points on the brightest parts of the adapt image. However, the source may also have faint extended emission that needs to be reconstructed.

The **weight floor** prevents those regions from receiving arbitrarily small weights. After applying the power, any weight below the floor is raised to that value.

For example, with a floor of `0.05`, weights of `0.001` and `0.02` both become `0.05`, while a weight of `0.8` remains unchanged.

Let’s compare three floors while keeping the power and number of mesh points fixed.
"""

for weight_floor in [0.0, 0.05, 0.5]:
    trial_image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels,
        weight_power=5.0,
        weight_floor=weight_floor,
    )

    weights = trial_image_mesh.weight_map_from(
        adapt_data=adapt_image,
    )

    weight_map = al.Array2D(
        values=weights,
        mask=dataset.mask,
    )

    trial_mesh_grid = trial_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=adapt_image,
    )

    aplt.plot_array(
        array=weight_map,
        vmin=0.0,
        vmax=1.0,
        title=f"Weight Map: Floor {weight_floor:g}",
    )

    aplt.plot_array(
        array=dataset.data,
        grid=trial_mesh_grid,
        title=f"Image-Mesh Points: Floor {weight_floor:g}",
    )

"""
Compare the points around the brightest peaks, along the fainter arcs, and in the background.

Increasing the floor reduces the contrast between their weights. This retains more sampling away from the brightest emission, but leaves fewer points available to concentrate on its peaks.

Because the background can cover much more area than the arcs, even a modest floor can affect the allocation substantially. We should inspect the resulting mesh rather than interpret the floor as a fixed fraction of points assigned to faint regions.

For our first reconstruction, we return to the mesh already constructed with `weight_power=1.0` and `weight_floor=0.05`. These are starting choices to investigate, rather than settings we have established as optimal.

---

__Reconstructing with the Hilbert mesh__

We now use the Hilbert coordinates to build a Delaunay source mesh.

The reconstruction proceeds as before: ray-trace the image-mesh points, construct the source triangulation, and solve for its vertex brightnesses. The difference is where we placed the image-mesh points.

We retain `Constant` regularization for now, so that we can investigate brightness-based mesh placement before introducing adaptive smoothing.
"""

def fit_delaunay_hilbert(coefficient):
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(
            pixels=hilbert_image_plane_mesh_grid.shape[0],
        ),
        regularization=al.reg.Constant(
            coefficient=coefficient,
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
    )

    adapt_images = al.AdaptImages(
        galaxy_image_dict={
            source_galaxy: adapt_image,
        },
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: hilbert_image_plane_mesh_grid,
        },
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
    )


hilbert_fit = fit_delaunay_hilbert(
    coefficient=overlay_coefficient,
)

hilbert_mapper = hilbert_fit.inversion.cls_list_from(al.Mapper)[0]

aaplt.plot_mapper(
    mapper=hilbert_mapper,
    solution_vector=hilbert_fit.inversion.reconstruction,
    mesh_grid=hilbert_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Hilbert Mesh: Initial Source Reconstruction",
)

aplt.subplot_fit_imaging(fit=hilbert_fit)

"""
Inspect the mesh around the compact source centre. Has the brightness weighting placed more vertices across the region where its brightness changes rapidly?

Then inspect the residuals around the bright arcs. Additional resolution gives the reconstruction more freedom to represent their structure, but regularization still determines how freely neighbouring brightnesses can differ.

We have initially reused the overlay’s preferred coefficient. As tutorial 2 showed, changing the mesh can change the preferred smoothing strength. We therefore need to explore regularization again before comparing the two reconstructions.

---

__Comparing overlay and Hilbert reconstructions__

Let’s scan the regularization coefficient for the Hilbert mesh, using the same range as the overlay.

The adapt image and Hilbert mesh coordinates remain fixed during this scan. Each coefficient produces a new reconstruction on that mesh.
"""

hilbert_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_delaunay_hilbert(coefficient=coefficient)
    evidence = float(current_fit.log_evidence)
    hilbert_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        hilbert_fit = current_fit

hilbert_evidences = np.asarray(hilbert_evidences)
if not np.all(np.isfinite(hilbert_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(hilbert_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the hilbert peak.")

hilbert_coefficient = float(overlay_coefficients[best_index])

reference_evidence = max(
    float(np.max(overlay_evidences)),
    float(np.max(hilbert_evidences)),
)

fig, ax = plt.subplots()

for label, values in [
    ("Overlay", overlay_evidences),
    ("Hilbert", hilbert_evidences),
]:
    ax.semilogx(
        overlay_coefficients,
        values - reference_evidence,
        marker="o",
        label=label,
    )

ax.set_xlabel("Regularization coefficient")
ax.set_ylabel("Log evidence relative to the highest sampled value")
ax.legend()
plt.show()

"""
Both curves use the same reference value, preserving the difference between their peaks. Compare their preferred coefficients as well as their peak heights.

We can now inspect each mesh at its own preferred sampled coefficient.
"""

for label, current_fit in [
    ("Overlay", overlay_fit),
    ("Hilbert", hilbert_fit),
]:
    print(f"{label}: log evidence={float(current_fit.log_evidence):.3f}")

    aplt.subplot_fit_imaging(fit=current_fit)
    aplt.subplot_fit_imaging_of_planes(fit=current_fit)

print(
    "Hilbert minus overlay log evidence:",
    float(hilbert_fit.log_evidence - overlay_fit.log_evidence),
)

"""
A positive difference favours this Hilbert reconstruction over this overlay reconstruction under the comparison we have made. A negative difference favours the overlay. Refine the coefficient sampling around both peaks before interpreting a small difference.

Connect the evidence comparison to the residuals. Does the brightness-adaptive mesh better reproduce the compact emission? Are the fainter regions still adequately described?

This tests one Hilbert weighting choice, using a fixed adapt image. It does not establish that every Hilbert mesh outperforms every overlay.

We have now addressed where the source points are placed. Next, we will address how to regularize their brightnesses on an irregular mesh, before allowing the smoothing strength to vary between bright and faint regions.

__Smoothing on an irregular mesh__

Our Delaunay mesh has vertices with different neighbour counts, edge lengths and directions.

With neighbour-based regularization, we penalize brightness differences along those edges. Consequently, the pattern of comparisons changes with the triangulation: one vertex might have several nearby neighbours, while another has fewer neighbours at larger distances.

**Split regularization** provides another way to measure local brightness variations.

Around each source vertex, it defines four nearby sampling positions arranged in a cross. It interpolates the reconstructed brightness at those positions and compares them with the brightness at the central vertex. The size of the cross follows the local mesh scale.

This gives each vertex a common pattern of local comparisons, instead of relying only on its particular set of connected neighbours.

The additional sampling positions are not new source brightness parameters. Their brightnesses are interpolated from the existing reconstruction. We therefore change how smoothing is applied without increasing the number of brightness values solved by the inversion.

It is also important to distinguish **Split** from **adaptive** regularization:

- `ConstantSplit` uses the cross-based comparisons with one coefficient across the mesh.
- Adaptive Split regularization uses those comparisons while allowing the smoothing strength to vary across the source.

We will first examine `ConstantSplit`, then introduce the brightness-dependent coefficients.

---

__Reconstructing with ConstantSplit__

Let’s keep the Hilbert mesh and adapt image fixed and change the regularization scheme.

We generalize our fitting helper so that it accepts a regularization object. This will let us use the same reconstruction setup for both constant and adaptive Split regularization.
"""

def fit_hilbert_with_regularization(regularization):
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(
            pixels=hilbert_image_plane_mesh_grid.shape[0],
        ),
        regularization=regularization,
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
    )

    tracer = al.Tracer(
        galaxies=[lens_galaxy, source_galaxy],
    )

    adapt_images = al.AdaptImages(
        galaxy_image_dict={
            source_galaxy: adapt_image,
        },
        galaxy_image_plane_mesh_grid_dict={
            source_galaxy: hilbert_image_plane_mesh_grid,
        },
    )

    return al.FitImaging(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(use_border_relocator=True),
    )

"""
The coefficient preferred by neighbour-based regularization need not be preferred by Split regularization. We therefore repeat the coefficient scan.
"""

split_evidences = []
best_evidence = -np.inf

# Keep only the preferred full fit; other scan entries are scalar evidences.
for coefficient in overlay_coefficients:
    coefficient = float(coefficient)
    current_fit = fit_hilbert_with_regularization(regularization=al.reg.ConstantSplit(coefficient=coefficient))
    evidence = float(current_fit.log_evidence)
    split_evidences.append(evidence)
    if evidence > best_evidence:
        best_evidence = evidence
        split_fit = current_fit

split_evidences = np.asarray(split_evidences)
if not np.all(np.isfinite(split_evidences)):
    raise ValueError("Inspect non-finite evidence values before continuing.")

best_index = int(np.argmax(split_evidences))
if best_index in (0, len(overlay_coefficients) - 1):
    raise ValueError("Extend the coefficient range to bracket the split peak.")

split_coefficient = float(overlay_coefficients[best_index])

print("Preferred sampled Split coefficient:", split_coefficient)

aplt.subplot_fit_imaging(fit=split_fit)
aplt.subplot_fit_imaging_of_planes(fit=split_fit)

print(
    "ConstantSplit minus Constant log evidence:",
    float(split_fit.log_evidence - hilbert_fit.log_evidence),
)

"""
Compare the source and residuals with the previous Hilbert reconstruction. We have kept the mesh fixed, so the changes arise from the regularization scheme and its selected coefficient.

As before, refine the coefficient sampling around the peaks before interpreting small evidence differences.

Split changes how local brightness variations are penalized, but `ConstantSplit` still applies one coefficient everywhere. It therefore retains the centre-versus-outskirts compromise identified in **Fit Problems**.

---

__Allowing different smoothing strengths__

We now let the regularization strength depend on the source brightness.

The adapt image supplies a brightness-based signal estimate for each source vertex through its mappings to the image. These signals are normalized so that the highest has a value of one.

We use that information to transition between two coefficients:

- `inner_coefficient` controls smoothing in high-signal regions.
- `outer_coefficient` controls smoothing in low-signal regions.

Despite their names, these coefficients are not assigned using distance from the source centre. A bright off-centre component can receive the high-signal coefficient too.

For our compact source, we want to explore weaker smoothing in the bright region and stronger smoothing in the faint outskirts. The `signal_scale` parameter controls how the normalized signal determines the transition between them.

We use `AdaptSplitPower` with `power=1.0`, which keeps the coefficient convention consistent with `ConstantSplit`. Setting its inner and outer coefficients equal then recovers the constant-strength Split scheme.

Let’s first try a deliberately contrasting pair of coefficients to make the spatial variation visible.
"""

adaptive_regularization = al.reg.AdaptSplitPower(
    inner_coefficient=split_coefficient / 10.0,
    outer_coefficient=split_coefficient * 10.0,
    signal_scale=1.0,
    power=1.0,
)

adaptive_fit = fit_hilbert_with_regularization(
    regularization=adaptive_regularization,
)

adaptive_mapper = adaptive_fit.inversion.cls_list_from(al.Mapper)[0]

adaptive_weights = (
    adaptive_mapper.regularization.regularization_weights_from(
        linear_obj=adaptive_mapper,
    )
)

aaplt.plot_inversion_reconstruction(
    mapper=adaptive_mapper,
    pixel_values=adaptive_weights,
    grid=adaptive_mapper.source_plane_mesh_grid,
    cb_unit="",
    zoom_to_brightest=False,
    title="Adaptive Split: Regularization Weights",
)

aaplt.plot_mapper(
    mapper=adaptive_mapper,
    solution_vector=adaptive_fit.inversion.reconstruction,
    mesh_grid=adaptive_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Adaptive Split: Source Reconstruction",
)

aplt.subplot_fit_imaging(fit=adaptive_fit)

r"""
Compare the regularization-weight map with the reconstructed source. Smaller weights should occur where the adapt image indicates stronger source emission, allowing greater local brightness variation. Larger weights encourage more smoothing in low-signal regions.

These weights control the regularization penalty; they are not uncertainties on the reconstructed brightnesses.

The trial coefficients were chosen to illustrate the mechanism. We have not yet established that they improve the fit or maximize the evidence. Next, we will examine the transition controlled by `signal_scale` and explore the adaptive coefficients systematically.

__Controlling the transition with `signal_scale`__

The inner and outer coefficients set the smoothing strengths at the high- and low-signal ends of the source. How do we assign a strength to the regions between them?

We first raise each normalized source signal to `signal_scale`. With `power=1.0`, the regularization weight is then:

\[
w = c_{\mathrm{inner}}\,s
  + c_{\mathrm{outer}}\,(1-s),
\]

where \(s\) is the normalized signal **after** applying `signal_scale`.

At \(s=1\), the weight equals the inner coefficient. At \(s=0\), it equals the outer coefficient. Intermediate values blend the two.

Let’s plot this relationship while keeping the coefficients fixed.
"""

normalized_signal = np.linspace(0.0, 1.0, 200)

inner_coefficient = split_coefficient / 10.0
outer_coefficient = split_coefficient * 10.0

fig, ax = plt.subplots()

for signal_scale in [0.5, 1.0, 3.0]:
    scaled_signal = normalized_signal**signal_scale

    weights = (
        inner_coefficient * scaled_signal
        + outer_coefficient * (1.0 - scaled_signal)
    )

    ax.plot(
        normalized_signal,
        weights,
        label=f"signal_scale={signal_scale:g}",
    )

ax.set_xlabel("Normalized source signal before scaling")
ax.set_ylabel("Regularization weight")
ax.legend()
plt.show()

"""
For a normalized signal between zero and one, increasing `signal_scale` reduces its scaled value. In this example, that moves its weight towards the stronger outer coefficient. Only the highest-signal regions retain weights close to the weaker inner coefficient.

Decreasing `signal_scale` has the opposite effect: more of the source receives weights closer to the inner coefficient.

These weights are based on the fixed adapt image and its mappings. They are not recalculated from each newly reconstructed brightness distribution during this experiment.

Also distinguish `signal_scale` from the Hilbert mesh’s `weight_power`. Both transform brightness-derived information, but they control different things: `weight_power` changes **where mesh points are placed**, whereas `signal_scale` changes **how their brightnesses are regularized**.

---

__Exploring the adaptive coefficients__

We now have three regularization parameters: the inner coefficient, outer coefficient and signal scale.

Let’s explore a small grid of their values, keeping the mass model, Hilbert mesh and adapt image fixed. This makes the experiment more expensive than our one-coefficient scans, so we retain only the evidence values and the best fit encountered.

We centre the coefficient grid on the preferred `ConstantSplit` value. The grid includes equal inner and outer coefficients, allowing the adaptive scheme to recover constant-strength smoothing.

We also allow either coefficient to be larger. Although weaker smoothing in bright regions motivates the scheme, we should inspect what the evidence actually prefers.
"""

adaptive_coefficient_grid = (
    split_coefficient * np.logspace(-1, 1, 5)
)
signal_scales = [0.5, 1.0, 3.0]

adaptive_trials = []
best_adaptive_fit = None
best_adaptive_parameters = None
best_adaptive_evidence = -np.inf

for inner_coefficient in adaptive_coefficient_grid:
    for outer_coefficient in adaptive_coefficient_grid:
        for signal_scale in signal_scales:
            regularization = al.reg.AdaptSplitPower(
                inner_coefficient=float(inner_coefficient),
                outer_coefficient=float(outer_coefficient),
                signal_scale=signal_scale,
                power=1.0,
            )

            current_fit = fit_hilbert_with_regularization(
                regularization=regularization,
            )
            log_evidence = float(current_fit.log_evidence)

            if not np.isfinite(log_evidence):
                raise ValueError(
                    "Inspect the non-finite adaptive evidence before continuing."
                )

            parameters = (
                float(inner_coefficient),
                float(outer_coefficient),
                signal_scale,
            )

            adaptive_trials.append((*parameters, log_evidence))

            if log_evidence > best_adaptive_evidence:
                best_adaptive_evidence = log_evidence
                best_adaptive_parameters = parameters
                best_adaptive_fit = current_fit

best_inner, best_outer, best_signal_scale = best_adaptive_parameters

print("Preferred sampled inner coefficient:", best_inner)
print("Preferred sampled outer coefficient:", best_outer)
print("Preferred sampled signal scale:", best_signal_scale)
print("Highest sampled log evidence:", best_adaptive_evidence)

"""
This performs 75 inversions. It is an introductory exploration, rather than a precise optimization of all three parameters.

Check whether a preferred coefficient lies at either end of its range. If so, extend that range before concluding that the optimum has been located. The three signal-scale values are also widely spaced and may require further exploration.

If the preferred inner and outer coefficients are equal, changing `signal_scale` has no effect: the weight is constant everywhere.

For a complete modelling analysis, these parameters can be explored by a non-linear search alongside the lens parameters. We will use that approach in the final fit.

---

__Does adaptive regularization address the fit problem?__

Let’s inspect the highest-evidence adaptive reconstruction we have evaluated and compare it with `ConstantSplit`.

We keep the mesh fixed between these fits. This isolates the change from one smoothing strength to brightness-dependent smoothing.
"""

comparison_fits = [
    ("ConstantSplit", split_fit),
    ("Adaptive Split", best_adaptive_fit),
]

source_vmax = max(
    float(np.max(np.asarray(current_fit.inversion.reconstruction)))
    for _, current_fit in comparison_fits
)

for label, current_fit in comparison_fits:
    mapper = current_fit.inversion.cls_list_from(al.Mapper)[0]

    aaplt.plot_mapper(
        mapper=mapper,
        solution_vector=current_fit.inversion.reconstruction,
        mesh_grid=mapper.source_plane_mesh_grid,
        vmin=0.0,
        vmax=source_vmax,
        zoom_to_brightest=False,
        title=f"{label}: Source Reconstruction",
    )

    aplt.subplot_fit_imaging(fit=current_fit)

print(
    "Adaptive minus constant Split log evidence:",
    best_adaptive_evidence - float(split_fit.log_evidence),
)

"""
Return to the two regions that motivated this experiment.

At the bright centre, does the adaptive reconstruction preserve a sharper brightness peak and reduce the corresponding arc residuals? In the outskirts, does it suppress fluctuations without removing faint emission needed to explain the image?

Inspect the preferred regularization weights alongside those changes.
"""

best_adaptive_mapper = best_adaptive_fit.inversion.cls_list_from(
    al.Mapper
)[0]

best_adaptive_weights = (
    best_adaptive_mapper.regularization.regularization_weights_from(
        linear_obj=best_adaptive_mapper,
    )
)

aaplt.plot_inversion_reconstruction(
    mapper=best_adaptive_mapper,
    pixel_values=best_adaptive_weights,
    grid=best_adaptive_mapper.source_plane_mesh_grid,
    cb_unit="",
    zoom_to_brightest=False,
    title="Preferred Adaptive Regularization Weights",
)

"""
A higher log evidence shows that this adaptive setting improves the reconstruction statistic for the fixed lens model and mesh. It is not, by itself, a model-comparison evidence marginalized over the additional regularization parameters.

If substantial residuals remain, inspect where they occur. Adaptive regularization still cannot supply resolution missing from the mesh, and both adaptive steps depend on the initial adapt image.

We have now explored the mesh placement and smoothing separately. Next, we will bring them together in an actual lens-model search, using an initial MGE fit, prior passing and the positions likelihood penalty.

__Starting the final modelling chain__

So far, we have supplied the true lens mass model to isolate the effects of mesh placement and regularization.

We will now fit the mass parameters as well. Following tutorial 4, we begin with an MGE source, then use its result to initialize the adaptive pixelized-source search.

This first fit provides three things: an initial mass estimate, an image of the fitted source emission to guide adaptation, and the positions used by the positions likelihood penalty.

We keep the mass centre fixed for this example, while allowing its Einstein radius and ellipticity to vary.
"""

from pathlib import Path
import autofit as af

initial_mass = af.Model(al.mp.Isothermal)
initial_mass.centre = (0.0, 0.0)
initial_mass.einstein_radius = af.UniformPrior(
    lower_limit=0.5,
    upper_limit=2.5,
)

initial_lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=initial_mass,
)

initial_source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.model_util.mge_model_from(
        mask_radius=3.0,
        total_gaussians=20,
        centre_prior_is_uniform=False,
        sigma_min=dataset.pixel_scales[0] / 10.0,
    ),
)

initial_model = af.Collection(
    galaxies=af.Collection(
        lens=initial_lens,
        source=initial_source,
    ),
)

print(initial_model.info)

initial_search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_6",
    name="search_1_mge",
    unique_tag="very_compact",
    n_live=100,
)

initial_analysis = al.AnalysisImaging(dataset=dataset)

initial_result = initial_search.fit(
    model=initial_model,
    analysis=initial_analysis,
)

print(initial_result.info)

initial_model_fit = initial_result.max_log_likelihood_fit

aplt.subplot_fit_imaging(fit=initial_model_fit)
aplt.subplot_fit_imaging_of_planes(fit=initial_model_fit)

"""
The narrow Gaussian components allow the MGE to describe the compact source centre. Their intensities are still solved linearly, while the search explores the shared source geometry and lens mass parameters.

Inspect this result before continuing. Does it explain the multiple-image configuration, and does it recover a sensible compact source? We should not use a misleading initial solution to guide the adaptive fit.

---

__Preparing adaptation and the positions likelihood penalty__

We now derive a new adapt image from the fitted MGE result. This replaces the image used in our fixed-mass experiments.

For the modelling chain, we express the fitted source image in signal-to-noise units and cap it at `3.0`. This reduces the dominance of the very brightest peaks, retaining more weight for fainter detected emission.

The cap affects only the image used to guide adaptation. It does not change the observed data, noise map or image likelihood.

Our example contains no foreground light, so the full model image is the source contribution.
"""

source_model_image = np.asarray(
    initial_model_fit.model_data.slim
)

source_model_snr = (
    source_model_image / np.asarray(dataset.noise_map.slim)
)

search_adapt_image = al.Array2D(
    values=np.clip(source_model_snr, a_min=0.0, a_max=3.0),
    mask=dataset.mask,
)

aplt.plot_array(
    array=search_adapt_image,
    title="Adapt Image for the Search: Capped Model Signal-to-Noise",
)

"""
We construct the Hilbert image mesh from this image. Its image-plane coordinates remain fixed during the next search, while their source-plane positions change as the proposed mass model changes.
"""

search_image_mesh = al.image_mesh.Hilbert(
    pixels=hilbert_pixels,
    weight_power=1.0,
    weight_floor=0.05,
)

search_image_plane_mesh_grid = (
    search_image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=search_adapt_image,
    )
)

search_adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'source')": search_adapt_image,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": search_image_plane_mesh_grid,
    },
)

aplt.plot_array(
    array=dataset.data,
    grid=search_image_plane_mesh_grid,
    title="Hilbert Image Mesh for the Model Search",
)

"""
The dictionaries identify the source by its path in the model, `galaxies.source`. This lets the analysis associate the fixed adaptation inputs with each new source-galaxy instance proposed by the search.

Following Chapter 2, Tutorial 5, we also prepare the **positions likelihood penalty** from the initial result.
"""

search_positions_likelihood = initial_result.positions_likelihood_from(
    factor=3.0,
    minimum_threshold=0.2,
)

aplt.plot_array(
    array=dataset.data,
    positions=search_positions_likelihood.positions,
    title="Positions for the Pixelized-Source Search",
)

print(
    "Source-plane separation threshold:",
    search_positions_likelihood.threshold,
    "arcsec",
)

"""
Inspect both the mesh placement and predicted image positions. They should follow the source emission and multiple-image configuration before we use them in the next fit.

---

__Defining the adaptive Delaunay model__

We now replace the MGE source with a Delaunay pixelization and adaptive Split regularization.

The mass parameters remain free, with priors passed from the initial result. The mesh’s image-plane coordinates and total number of vertices are fixed, but ray tracing changes its source-plane geometry for every proposed mass model.

We also make the inner coefficient, outer coefficient and signal scale free parameters. The search can therefore adjust the smoothing pattern as it refines the mass model.
"""

search_regularization = af.Model(al.reg.AdaptSplitPower)

search_regularization.inner_coefficient = af.LogUniformPrior(
    lower_limit=1.0e-3,
    upper_limit=1.0e3,
)

search_regularization.outer_coefficient = af.LogUniformPrior(
    lower_limit=1.0e-3,
    upper_limit=1.0e3,
)

search_regularization.signal_scale = af.UniformPrior(
    lower_limit=0.1,
    upper_limit=5.0,
)

search_regularization.power = 1.0

search_pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(
        pixels=search_image_plane_mesh_grid.shape[0],
    ),
    regularization=search_regularization,
)

search_source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=search_pixelization,
)

adaptive_model = af.Collection(
    galaxies=af.Collection(
        lens=initial_result.model.galaxies.lens,
        source=search_source,
    ),
)

print(adaptive_model.info)

"""
Inspect the model summary. The source brightness values are still absent from the non-linear parameter list: the inversion solves them for each proposal.

We have not forced the inner coefficient to be smaller than the outer coefficient. That ordering motivated our experiment, but the search is allowed to determine which values the data favour.

The `power` setting remains fixed at `1.0`; it specifies the coefficient convention rather than providing another degree of freedom in this tutorial.

The regularization priors are starting ranges. After fitting, we will check whether the inferred parameters approach their boundaries and whether broader ranges or further exploration are needed.

We are now ready to run the adaptive Delaunay fit and inspect the resulting mass model, source reconstruction and smoothing pattern.

__Running the adaptive Delaunay fit__

The second search combines our Delaunay model, adaptation inputs and positions likelihood penalty.

For each proposed mass model, the image-mesh coordinates are ray-traced into the source plane. The inversion reconstructs the source using the proposed adaptive regularization parameters, and the analysis evaluates the resulting fit.

The positions likelihood penalty discourages proposals that trace the selected corresponding image positions too far apart.
"""

adaptive_analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=search_adapt_images,
    positions_likelihood_list=[search_positions_likelihood],
    settings=al.Settings(use_border_relocator=True),
)

adaptive_search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_3" / "tutorial_6",
    name="search_2_adaptive_delaunay",
    unique_tag="very_compact",
    n_live=100,
)

adaptive_result = adaptive_search.fit(
    model=adaptive_model,
    analysis=adaptive_analysis,
)

print(adaptive_result.info)

final_fit = adaptive_result.max_log_likelihood_fit

aplt.subplot_fit_imaging(fit=final_fit)
aplt.subplot_fit_imaging_of_planes(fit=final_fit)

"""
This is a full non-linear search, rather than the fixed-mass reconstructions we evaluated earlier. It must explore combinations of mass and regularization parameters, so it will take longer.

Inspect the image residuals first. Are there coherent patches around the bright arcs? Has the reconstruction captured the compact source without introducing conspicuous fluctuations elsewhere?

Then examine the parameter distributions. A satisfactory-looking best fit does not establish that the search has explored the relevant solutions or measured reliable uncertainties. Check for parameters approaching prior boundaries and use the search diagnostics introduced in Chapter 2.

---

__Inspecting the source and smoothing pattern__

Let’s look more closely at where the final mesh places its vertices and how strongly those vertices are regularized.
"""

final_mapper = final_fit.inversion.cls_list_from(al.Mapper)[0]

final_regularization_weights = (
    final_mapper.regularization.regularization_weights_from(
        linear_obj=final_mapper,
    )
)

aaplt.plot_mapper(
    mapper=final_mapper,
    solution_vector=final_fit.inversion.reconstruction,
    mesh_grid=final_mapper.source_plane_mesh_grid,
    zoom_to_brightest=False,
    title="Final Adaptive Delaunay Source",
)

aaplt.plot_inversion_reconstruction(
    mapper=final_mapper,
    pixel_values=final_regularization_weights,
    grid=final_mapper.source_plane_mesh_grid,
    cb_unit="",
    zoom_to_brightest=False,
    title="Final Regularization Weights",
)

"""
Return to the two problems from **Fit Problems**.

First, does the mesh place enough vertices across the compact source centre to describe its steep brightness gradient?

Second, does the inferred regularization allow the centre to vary sharply while smoothing the faint regions appropriately? Inspect the fitted inner and outer coefficients rather than assuming that the search must prefer the ordering we anticipated.

We can also check the positions likelihood penalty for the displayed model.
"""

final_positions = final_fit.tracer.traced_grid_2d_list_from(
    grid=search_positions_likelihood.positions
)[1]

position_coordinates = np.asarray(final_positions)

position_separations = np.linalg.norm(
    position_coordinates[:, None, :]
    - position_coordinates[None, :, :],
    axis=-1,
)

maximum_separation = float(np.max(position_separations))

print(
    "Maximum source-plane position separation:",
    maximum_separation,
    "arcsec",
)
print(
    "Positions threshold:",
    search_positions_likelihood.threshold,
    "arcsec",
)
print(
    "Within threshold:",
    maximum_separation <= search_positions_likelihood.threshold,
)

"""
For this simulation, the true Einstein radius is `1.6` arcsec. Compare it with the inferred value and uncertainty in the result summary. In observed data, that answer would not be available, making the residuals, parameter constraints and search behaviour especially important.

Finally, compare the initial MGE fit with the adaptive pixelized fit.
"""

for label, current_fit in [
    ("Initial MGE", initial_model_fit),
    ("Adaptive Delaunay", final_fit),
]:
    print(label)
    aplt.subplot_fit_imaging(fit=current_fit)
    aplt.subplot_fit_imaging_of_planes(fit=current_fit)

"""
Our simulated source is a smooth Sérsic profile, which an MGE can describe well. The adaptive pixelization need not outperform that analytic model dramatically. Here, we are demonstrating how to fit a compact source with a pixelization while allowing the lens mass to vary.

The additional spatial freedom becomes particularly useful for irregular sources whose structure is difficult to describe using a prescribed arrangement of analytic components.

---

__Wrap Up__

We began this chapter by replacing an analytic source profile with brightness values on a mesh.

That gave the source more spatial freedom, but it also introduced choices that affect the reconstruction: where to place the mesh points, how to interpolate between them, and how strongly to regularize their brightnesses.

In **Fit Problems**, we saw why selecting a constant regularization coefficient is not always enough. A compact source may require more resolution around its bright centre and different smoothing strengths across its brightness distribution.

This tutorial addressed those limitations in stages.

An image-plane overlay gave us source sampling that responds to the lens magnification. A Hilbert image mesh then used a fitted brightness distribution to concentrate more sampling on the source emission. Delaunay triangulation connected the traced points and provided interpolation between their brightness values.

Split regularization supplied local smoothing comparisons on that irregular mesh. Adaptive Split regularization allowed the strength of those comparisons to vary with the source signal.

We then brought these methods into a lens-model search, using an initial MGE fit, prior passing and the positions likelihood penalty to help avoid unphysical mass-and-source solutions.

Delaunay meshes with brightness adaptation are a useful choice when the source needs spatially varying resolution. Their flexibility does not remove the need to inspect the fit. An inaccurate adapt image, insufficient mesh resolution or poorly explored regularization parameters can still leave a misleading reconstruction.

When assessing a result, connect three views:

- **The image residuals:** what observed structure remains unexplained?
- **The source mesh and reconstruction:** where is resolution available, and what brightness structure is being inferred?
- **The mass and regularization constraints:** what parameters does the search favour, and how well has it explored them?

MGE models remain valuable for initial fits, foreground light and sources well described by analytic components. Pixelizations complement them when we need greater freedom to reconstruct the source.

We now have the tools to use both together—and to investigate why a reconstruction succeeds or struggles, rather than relying only on how its source plot looks.

**🤖 Ask AI:** How do Hilbert mesh weighting and adaptive Split regularization address different limitations of a compact-source reconstruction?

__Env__

Keep full-size datasets during automated runs so compact sources remain resolved
and the evidence scans test the same examples as the lesson.

ENV: full_datasets
"""
