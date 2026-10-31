"""
Tutorial 4: Point Sources
=========================

In the previous tutorials, we lensed extended galaxies, producing arcs and Einstein rings. We now turn to sources
whose individual lensed images are too small for a telescope to resolve: **point sources**.

Two examples are quasars, whose bright central emission comes from material accreting onto a supermassive black
hole, and supernovae, which are exploding stars. When multiply imaged by a foreground galaxy, they can appear as
several distinct points of light.

The lens equation is unchanged, but we use it to answer a different question. For an extended source, we took
image-plane coordinates and ray-traced them to the source-plane. For a point source at a specified source-plane
position, we want to find every image-plane position where it appears.

We will use the `PointSolver` to find these positions, then calculate their magnifications and time delays. The
lens equation from Tutorial 2 is enough to follow the examples; the optional mathematics in Tutorial 3 explains
the quantities in more detail.

__Contents__

- **Initial Setup:** Construct a lens with a compact extended source.
- **Seeing the Compact Source:** Plot its lensed image and identify the bright regions.
- **Point Sources:** Describe sources whose individual images cannot be resolved.
- **Point Source Tracer:** Place a point source at the compact source's centre.
- **The Lens Equation:** Find image positions for a specified source position.
- **Point Solver:** Search for images by iteratively ray-tracing and refining triangles.
- **Finding the Images:** Calculate the multiple image positions.
- **Multiple Images and Critical Curves:** Compare the positions with the extended image.
- **Magnifications:** Predict each image's flux from its magnification.
- **Interpreting Fluxes:** Understand why observed flux ratios can differ from predictions.
- **Time Delays:** Calculate relative arrival times for the multiple images.
- **Connecting Time Delays to the Images:** Compare positions, brightness and arrival order.
- **Extended Versus Point Computations:** Compare the two ways of predicting observations.
- **Wrap Up:** Summarise the tutorial and continue to realistic data.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

Let's begin with an extended source, using the same galaxy and tracer objects as before.

We give the source an `effective_radius` of just 0.1 arcseconds. This makes its light more concentrated and lets
us see how compact lensed images relate to the point sources we will introduce shortly.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy_extended = al.Galaxy(
    redshift=1.0,
    light=al.lp.ExponentialCore(
        centre=(0.07, 0.07),
        intensity=0.1,
        effective_radius=0.1,
    ),
)

tracer_extended = al.Tracer(
    galaxies=[lens_galaxy, source_galaxy_extended]
)

"""
__Seeing the Compact Source__

Let's plot the lensed image. Look for the bright regions around the lens: each contains light from the same
source, reaching us along a different path.

The source still has a finite extent, so its images can be stretched into arcs. Making a source smaller generally
makes its individual images more compact, but their shapes also depend on where the source lies relative to the
caustics.
"""
aplt.plot_array(
    array=tracer_extended.image_2d_from(grid=grid),
    title="Compact Source Multiple Images",
)

"""
__Point Sources__

Now imagine replacing this compact galaxy with the much smaller emitting region of a quasar or supernova.

If the telescope cannot resolve the extent of an individual lensed image, we can describe the source as a
**point source**. This is an approximation: the source has a physical size, but we do not need to represent that
size to predict its image positions.

An unresolved image does not necessarily occupy a single detector pixel. The telescope spreads its light into
a small pattern that can cover several pixels. We will introduce this blurring in Tutorial 5.

For point-source lensing, we measure the **positions** of the multiple images and may also use their **fluxes**,
which describe how much light we receive. If the source varies over time, we can measure the **time delays**
between the appearances of the same variation in different images.

A quasar or supernova can also sit inside a visible host galaxy. Such a lens may therefore contain both
unresolved point images and extended arcs.
"""

"""
__Point Source Tracer__

PyAutoLens represents a point source using the `Point` profile in `al.ps`. This profile specifies its
source-plane centre, without an extended surface-brightness distribution.

We place it at the same centre as our compact extended source, so we can compare its predicted image positions
with the image above.
"""
point_source = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(
    redshift=1.0,
    point_0=point_source,
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The name `point_0` identifies this source within the galaxy. When fitting observations, PyAutoLens uses it to
match the source to the dataset containing its multiple images. This becomes particularly useful when a lens
contains several background sources.

This tracer now contains the lens mass and the point-source position. Next, we will find the multiple image
positions that this combination predicts.
"""

r"""
__The Lens Equation__

In Tutorial 2, we used the lens equation to map an image-plane coordinate $\theta$ to a source-plane coordinate
$\beta$:

$$
\beta = \theta - \alpha(\theta).
$$

For an extended source, we evaluate this expression at every coordinate on our image grid, then calculate the
source's surface brightness at the resulting coordinates.

For a point source, we specify $\beta$ and ask which image-plane coordinates $\theta$ map back to it. Each
solution is a separate image of the same source.

Finding these solutions is harder than evaluating the equation at a known coordinate. The deflection
$\alpha(\theta)$ changes with position, and several different coordinates can map to the same source. Simple
mass profiles have analytic solutions, but more general models require a numerical search.
"""

"""
__Point Solver__

The `PointSolver` divides the image-plane into triangles and ray-traces their corners to the source-plane. It
then subdivides candidate triangles whose traced counterparts contain the source centre, and repeats the
ray-tracing with progressively smaller triangles to locate each image.

That is enough detail to use the solver here. For more on the method, its settings and limitations, see the
workspace's [Point Solver guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/guides/point_solver/limitations.ipynb).

We use a separate, wider grid to set the search region. The `pixel_scale_precision` of 0.001 arcseconds controls
how finely the solver refines its triangles; it is independent of the pixel size used to display our image.

The `magnification_threshold` excludes solutions with an absolute magnification below 0.1. This is a numerical
selection, rather than a complete description of which images a telescope would detect: detectability also
depends on source brightness, noise and foreground light.
"""
solver_grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=solver_grid,
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)

"""
__Finding the Images__

We now pass the solver our tracer and the point source's centre. The tracer supplies the deflection angles,
and the solver searches for image-plane positions that ray-trace back to that centre.

The returned coordinates are in `(y, x)` order, measured in arcseconds.
"""
positions = solver.solve(
    tracer=tracer,
    source_plane_coordinate=point_source.centre,
)

print("Multiple image positions (y, x) in arcseconds:")
print(positions)

"""
Every returned position represents an image of the same point source. The number of images depends on both
the lens mass distribution and the source position.

Some mass profiles also produce a faint image near the lens centre. Our singular isothermal profile does not
produce a finite central image, so we should not interpret the magnification threshold as removing a fifth
image in this example.
"""

"""
__Multiple Images and Critical Curves__

Let's overlay the solved positions on the compact extended source's image. The markers show where a point
source at its centre would appear; the surrounding light shows how the source's finite extent is stretched
by lensing.

We also plot the tangential critical curve introduced in Tutorial 3. Images close to this curve can be
strongly magnified.
"""
tangential_critical_curve_list = al.LensCalc.from_tracer(
    tracer=tracer
).tangential_critical_curve_list_from(grid=grid)

aplt.plot_array(
    array=tracer_extended.image_2d_from(grid=grid),
    positions=positions,
    lines=tangential_critical_curve_list,
    title="Multiple Images and Critical Curve",
)

"""
Compare the markers with the bright regions of the extended image. Both calculations use the same lens mass
and source centre: one evaluates an extended light profile across a grid, while the other finds the discrete
image positions of a point source.

**🤖 Ask AI:** For a fixed elliptical lens mass profile, why can moving a point source across a caustic change
the number of its multiple images?
"""

r"""
__Magnifications__

Each multiple image has its own magnification. For an unresolved image, we measure this through its total
**flux**, even though we cannot resolve the change in its apparent area.

If the source would have flux $F$ without lensing, an image with magnification $\mu$ has predicted flux

$$
F_{\mathrm{image}} = |\mu|F.
$$

The sign of $\mu$ describes the image's parity: a negative value indicates a mirror reversal. Flux is positive,
so we use the absolute magnification.

We calculate the magnification at each solved position below, then predict the image fluxes for an unlensed
source flux of 1.0 in arbitrary units.
"""
magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

print("Magnification of each multiple image:")
print(magnifications)

flux = 1.0
fluxes = [
    flux * np.abs(magnification)
    for magnification in magnifications
]

print("Predicted image fluxes for an unlensed source flux of 1.0:")
print(fluxes)

"""
The values follow the same order as `positions`. Larger absolute magnifications correspond to brighter
predicted images; a negative magnification does not mean a negative flux.

__Interpreting Fluxes__

The fluxes above are predictions from our smooth lens mass model. Observed image fluxes can differ for
several reasons:

- **Microlensing:** individual stars in the lens galaxy can further magnify or demagnify a compact source,
  with changes over time.
- **Dark matter substructure:** small concentrations of dark matter can perturb the image magnifications.
  These differences can themselves be used to investigate dark matter.
- **Dust:** material in the lens galaxy can absorb different amounts of light along each image's path.
- **Source variability:** because the images have different time delays, observations taken simultaneously
  can show the source at different stages of its variability.

Image positions are generally less sensitive to these effects than flux ratios, although mass perturbations
can also shift positions. We therefore often begin point-source modelling with the positions, adding flux
information when the relevant effects can be accounted for.

__Time Delays__

The multiple images show light that has followed different paths through the lens system. As introduced in
Tutorial 3, differences in both the geometry and the gravitational potential produce differences in arrival time.

If a quasar brightens, we can see the same brightening appear in its multiple images at different times. A lensed
supernova provides another example: its rise and decline can appear with a different delay in each image.

By monitoring these variations, we can measure **time delays** between the images. These depend on the lens
mass distribution and the cosmological distances, so combining measured delays with a lens model can constrain
the Hubble constant.

The next cell calculates the model's arrival times at the solved image positions. We subtract the first value
to express the delays relative to that image.
"""
time_delays = tracer.time_delays_from(grid=positions)

print("Model arrival times (days):")
print(time_delays)

print("Time delays relative to the first image (days):")
print(time_delays - time_delays[0])

"""
The first image in the array is simply our reference; it is not necessarily the earliest arrival. A negative
relative delay means that image arrives before the reference image, and a positive value means it arrives afterwards.

__Connecting Time Delays to the Images__

The positions, magnifications and time delays describe different properties of the same multiple images. Let's
print them together so we can compare each image's location, predicted brightness and arrival time.
"""
relative_time_delays = time_delays - time_delays[0]

for image_index, (position, magnification, delay) in enumerate(
    zip(positions, magnifications, relative_time_delays)
):
    print(
        f"Image {image_index}: "
        f"(y, x) = ({float(position[0]):.3f}, "
        f"{float(position[1]):.3f}) arcsec, "
        f"|magnification| = {float(np.abs(magnification)):.2f}, "
        f"relative delay = {float(delay):.2f} days"
    )

"""
Use the coordinates to identify these images in the earlier plot. The brightest predicted image has the largest
absolute magnification; the earliest arrival has the smallest relative delay. These describe different aspects
of the lens mapping, so brightness alone does not determine arrival order.

**🤖 Ask AI:** A lensed quasar varies over time, and its images have different time delays. Why should I account
for those delays before comparing their observed flux ratios with a lens model?

__Extended Versus Point Computations__

We have now predicted lensed images in two ways.

For an **extended source**, we ray-trace an image-plane grid to the source-plane and evaluate the source's
surface brightness at each traced coordinate. This produces an image that we can compare with observations
pixel by pixel.

For a **point source**, we specify its source-plane position and solve the lens equation to predict its multiple
image positions. We can then calculate magnifications and time delays at those positions, and compare these
predictions with measurements.

Both are ways of predicting observations from a lens model. The difference is the calculation needed to make
that prediction: evaluating light on a grid for an extended source, or searching for discrete image positions
for a point source.

When fitting image positions, we also need to associate each predicted image with an observed image. A trial
model can predict a different number of images from those detected, so this pairing needs care. The workspace
guides cover these fitting choices in more detail.

__Wrap Up__

In this tutorial, we used a point source to predict multiple images of a compact object such as a quasar or
supernova.

- The `Point` profile specifies the source-plane position.
- The `PointSolver` searches for image-plane positions that ray-trace back to that source.
- Magnifications predict the image fluxes, although additional physical effects can alter the observed flux ratios.
- Time delays describe when the same source variation appears in each image, and can be used to constrain
  cosmological distances.

The lens equation connects all of these calculations to the ray tracing in Tutorial 2. Tutorial 3 provides the
optional mathematical description of magnification and arrival times.

For examples of fitting observed point-source data, continue to the `autolens_workspace/scripts/point_source`
examples.

In **Tutorial 5: Data**, we return to extended sources and introduce the ingredients of a realistic observation,
including telescope blurring and noise.
"""
