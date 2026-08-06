"""
Tutorial 4: Point Sources
=========================

In the previous tutorials, the background sources we lensed were galaxies: extended objects whose light spreads over
many thousands of light years. When lensed, their light is warped into the arcs and Einstein rings we produced with
the `Tracer`, spread across many pixels of the image.

However, not every source is a galaxy. Some of the most scientifically valuable strong lenses have a background
source that is physically tiny: a **quasar** (the bright accretion disk around a supermassive black hole, light-days
across) or a **supernova** (an exploding star, even smaller). These are called **point sources**.

When a point source is strongly lensed, we do not see arcs. Instead, we see the same source appear multiple times,
as two or four (or more) distinct, unresolved points of light in the image. Modeling these systems is fundamentally
different from modeling extended sources — different enough that it deserves its own tutorial before we go any
further.

In this tutorial, we will see why. The short version: for an extended source, we ray-trace light *forwards* from the
image-plane to the source-plane, which is computationally simple. For a point source, we must go *backwards* — we
know where the source is, and we must find every image-plane position its light appears at. That means solving the
lens equation, an inverse problem with no analytic solution, which requires a dedicated numerical tool called the
`PointSolver`.

Only the lens equation from tutorial 2 is needed to follow this tutorial. The full lensing formalism — convergence,
potentials, and how deflection angles arise from a mass distribution — is covered in the next tutorial.

Here is an overview of what we'll cover:

- **Initial Setup**: Create a strong lens with a compact extended source, showing how a small source produces
  distinct multiple images rather than arcs.

- **Point Sources**: What a point source is physically, and why quasars and supernovae appear as multiple
  unresolved images.

- **Point Source Tracer**: Represent a point source in PyAutoLens using the `Point` profile and build a `Tracer`.

- **The Lens Equation**: Why finding a point source's multiple images means solving the lens equation — an inverse
  problem with no analytic solution.

- **Point Solver**: The `PointSolver`, which solves the lens equation numerically by ray-tracing triangles.

- **Multiple Images and Critical Curves**: Plot the solved image positions over the lensed image and the tracer's
  critical curves.

- **Magnifications**: Compute the magnification at each multiple image, and why observed fluxes are often
  unreliable for modeling.

- **Time Delays**: The arrival-time differences between multiple images, and why they matter for cosmology.

- **Extended Versus Point Computations**: A recap of why the two regimes require fundamentally different
  calculations and fitting strategies.

__Contents__

- **Initial Setup:** Create a strong lens with a compact extended source, producing distinct multiple images.
- **Point Sources:** What a point source is physically and why it appears as multiple unresolved images.
- **Point Source Tracer:** Represent a point source with the `Point` profile and build a `Tracer`.
- **The Lens Equation:** Finding multiple images means solving the lens equation, an inverse problem.
- **Point Solver:** The `PointSolver` solves the lens equation numerically via triangle ray-tracing.
- **Multiple Images and Critical Curves:** Plot the solved positions over the image and critical curves.
- **Magnifications:** The magnification of each multiple image, and why fluxes are often unreliable.
- **Time Delays:** The relative arrival times of the multiple images and their use in cosmology.
- **Extended Versus Point Computations:** Why the two regimes require fundamentally different calculations.
- **Wrap Up:** Summary of the script and next steps.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

Let's begin with a strong lens system like those of the previous tutorials: an elliptical isothermal mass profile
for the lens galaxy, and an extended light profile for the source galaxy.

There is one deliberate difference: the source's `effective_radius` is just 0.1", making it far more compact than
the sources we lensed before.
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
        centre=(0.07, 0.07), intensity=0.1, effective_radius=0.1
    ),
)

tracer_extended = al.Tracer(galaxies=[lens_galaxy, source_galaxy_extended])

"""
When we plot the image of this system, something new happens: instead of a sweeping arc or Einstein ring, the
source appears as distinct blobs of light — the same source, imaged multiple times at different locations in
the image-plane.

This is a general rule of strong lensing: the more compact the source, the more its lensed image breaks up into
separate multiple images. An extended galaxy blurs its multiple images together into arcs; a compact source keeps
them distinct.
"""
aplt.plot_array(
    array=tracer_extended.image_2d_from(grid=grid),
    title="Compact Source Multiple Images",
)

"""
__Point Sources__

Now imagine shrinking the source further — not to 0.01", but to the physical size of a quasar accretion disk or a
supernova photosphere. These objects are light-days to light-weeks across, which at cosmological distances
corresponds to micro-arcseconds on the sky. That is millions of times smaller than the resolution of any telescope.

Strictly speaking, such a source still has a finite size, but the telescope cannot resolve it. Each multiple image
appears as a single unresolved point of light, with all of its flux effectively contained within one pixel (spread
only by the telescope's blurring, which we will meet in a later tutorial).

This is what we call a **point source**. Lensed quasars and lensed supernovae are the classic examples, and they
are prized objects: the arrival-time differences between their multiple images can be used to measure the expansion
rate of the Universe (the Hubble constant), and their fluxes are sensitive probes of small-scale dark matter
structure.

For a point source, the extended-source picture of lensing breaks down:

- There is no arc or ring of surface brightness to fit, just a handful of image positions.

- The observable quantities are the (y,x) **positions** of the multiple images, and optionally their **fluxes**
  and **time delays**.

- The concept of evaluating a light profile's surface brightness on a ray-traced grid — the engine of the previous
  two tutorials — no longer applies, because the source has no extent to evaluate.

__Point Source Tracer__

PyAutoLens represents a point source with the `Point` profile, found in the `al.ps` module (`ps` for point source).
Unlike a light profile it has no `intensity`, `effective_radius` or elliptical components — it is fully described
by its (y,x) centre in the source-plane.

We give the point source the same centre, (0.07", 0.07"), as the compact extended source above, so its multiple
images will coincide with the bright blobs in the image we just plotted.
"""
point_source = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point_0=point_source)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Note that we attached the point source to its galaxy with the name `point_0`. This name is a label that PyAutoLens
uses when fitting real data to pair each point source in the model with the dataset containing its observed image
positions. With one source the name is a formality, but group- and cluster-scale lenses can contain many point
sources, and the names keep each one matched to its own data.

__The Lens Equation__

In tutorial 2, we met the **lens equation**, which relates a coordinate in the image-plane to the coordinate in
the source-plane its light ray comes from:

$\beta = \theta - \alpha(\theta)$

where $\theta$ is the image-plane (y,x) coordinate, $\alpha(\theta)$ is the deflection angle at that coordinate,
and $\beta$ is the source-plane (y,x) coordinate.

For the extended sources of tutorials 2 and 3, we always used this equation in the *forward* direction: take every
image-plane coordinate $\theta$ on our grid, compute its deflection angles, and subtract to get $\beta$. This is
easy — it is just an evaluation, one subtraction per coordinate.

For a point source, the question is reversed. We *know* the source's position $\beta$ (the centre of the `Point`
profile), and we want to find every image-plane position $\theta$ that satisfies the lens equation for that
$\beta$. Each solution is one of the multiple images.

This is called **solving the lens equation**, and it is much harder than evaluating it:

- The equation is non-linear, because the deflection angles $\alpha(\theta)$ vary with position in a way set by the
  mass profile. For all but the simplest mass profiles, no analytic solution exists.

- There are multiple solutions — that is the whole point! A single $\beta$ maps to two, four or more distinct
  $\theta$ positions, and we must find all of them.

We therefore need a numerical root-finding method that searches the image-plane for every position whose ray-traced
source-plane coordinate lands on the source. (Where the deflection angles themselves come from is the subject of
the next tutorial — for now, we just need the equation.)

__Point Solver__

PyAutoLens solves the lens equation with the `PointSolver`, which uses a triangle-based scheme:

1. Overlay the image-plane with a grid of triangles.

2. Ray-trace the corners of every triangle to the source-plane using the lens equation, giving a set of
   ray-traced source-plane triangles.

3. Keep every triangle that contains the source's (y,x) centre — somewhere inside that image-plane triangle is a
   multiple image.

4. Subdivide the retained triangles into smaller triangles and repeat, homing in on each multiple image with
   progressively finer precision.

The `pixel_scale_precision` input sets the triangle size at which the solver stops refining, and therefore the
precision of the image positions. Smaller values are more precise but cost more computation; 0.001" is a good
balance for most lens modeling.

Mass models also predict a faint "central image" near the centre of the lens, which is usually so heavily
demagnified that real observations never detect it. The `magnification_threshold` input discards solved images
whose magnification is below the threshold, removing this central image to match what the data would contain.
"""
solver_grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=solver_grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
We now solve the lens equation, passing the solver our tracer (which provides the mass model's deflection angles)
and the source-plane centre of the point source.

The result is the set of image-plane (y,x) coordinates of the multiple images.
"""
positions = solver.solve(tracer=tracer, source_plane_coordinate=(0.07, 0.07))

print("Multiple Image Positions (y,x) in arc-seconds:")
print(positions)

"""
__Multiple Images and Critical Curves__

For our elliptical mass profile, the solver finds four multiple images (a fifth, central image existed but was
removed by the `magnification_threshold`). The number of images depends on the mass model:

- Spherical mass profiles produce three images (including the demagnified central image).

- Elliptical mass profiles produce five images (again including the central image).

- More complex systems, with multiple lens galaxies, can produce even more.

To see that the solver got the right answer, we plot the solved positions on top of the compact extended source's
image from the start of the tutorial. The positions land on the centres of the bright multiple images.

We also overlay the tracer's tangential critical curve (introduced in tutorial 3 — the line of infinite
magnification in the image-plane). The multiple images of a strongly lensed point source straddle the critical
curve, and images close to it are the most highly magnified.
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
__Magnifications__

Lensing does not just relocate a point source's light — it magnifies it. Each multiple image has its own
magnification, set by how strongly the mass model focuses light rays at that image-plane position, and it can make
an image tens of times brighter than the unlensed source.

For an extended source, magnification stretches the source over more pixels. For a point source, the image stays
unresolved — so the magnification instead multiplies its observed **flux**. If the source's intrinsic flux is $F$,
the flux of a multiple image with magnification $mu$ is $|mu| * F$.

We compute the magnification at each solved image position below (the sign of $mu$ encodes the image's parity —
whether it is mirror-flipped — which is why we take its absolute value for fluxes).
"""
magnifications = al.LensCalc.from_tracer(
    tracer=tracer
).magnification_2d_via_hessian_from(grid=positions)

print("Magnification of each Multiple Image:")
print(magnifications)

flux = 1.0
fluxes = [flux * np.abs(magnification) for magnification in magnifications]

print("Flux of each Multiple Image (for a source of flux 1.0):")
print(fluxes)

"""
In principle, comparing the observed flux ratios of the images to the model's magnification ratios is a powerful
constraint on the mass model. In practice, point-source fluxes are treated with caution, because effects the smooth
mass model does not include can change them substantially:

- **Microlensing**: individual stars in the lens galaxy lens the point source again on micro-arcsecond scales,
  magnifying or demagnifying each image by unpredictable amounts that change over time.

- **Dark matter substructure**: small invisible clumps of dark matter near an image perturb its magnification
  (this is a systematic for mass modeling, but also exactly why lensed quasars are used to *search* for
  substructure).

- Dust in the lens galaxy and the source's own intrinsic variability further distort the observed flux ratios.

Image positions are essentially immune to all of these, which is why point-source lens modeling is built on
positions first, with fluxes included only when the science demands it and the systematics are under control.

__Time Delays__

There is a third observable unique to point sources. The light of each multiple image travels a different path
through the Universe, and each path takes a different time — partly because the geometric path lengths differ, and
partly because light is slowed as it climbs through the lens galaxy's gravitational field (the Shapiro delay).

For an extended, steady source this is unobservable. But quasars flicker and supernovae explode: when the source
varies, the variation appears in each multiple image at a different time. By monitoring the images, these
**time delays** can be measured — typically days to months apart.

Time delays are a cosmological gold mine: their absolute scale depends on the distances between us, the lens and
the source, so a lens model plus measured delays yields the Hubble constant. This is time-delay cosmography, one
of the headline applications of point-source lensing.

The tracer computes the arrival time at each image position from its mass model and the galaxies' redshifts. Only
the *differences* between images are observable — a delay of the same length along every path is invisible — so it
is the relative values below that matter.
"""
time_delays = tracer.time_delays_from(grid=positions)

print("Time Delay of each Multiple Image (days):")
print(time_delays)

print("Time Delays relative to first image (days):")
print(time_delays - time_delays[0])

"""
__Extended Versus Point Computations__

We can now state precisely why point-source lensing works so differently from everything in tutorials 2 and 3, on
both the computation and the data.

**Extended sources are forward-modeled.** To predict the data, we ray-trace every image-plane pixel to the
source-plane via the lens equation (an evaluation, not a solve) and compute the source light profile's surface
brightness at each ray-traced coordinate. This is computationally cheap, requires no equation solving, and a fit
compares the predicted and observed images pixel-by-pixel.

**Point sources require solving an inverse problem.** The source has no surface brightness to evaluate — the model
must predict the discrete positions of the multiple images, which means numerically solving the lens equation with
the `PointSolver` every time the mass model changes. A fit then compares the predicted image positions to the
observed ones (and optionally the predicted fluxes and time delays to their measurements) — a handful of numbers,
rather than thousands of pixels.

This trade shapes the whole analysis. The point-source dataset is tiny — a few positions with uncertainties — but
each model evaluation involves root finding over the image plane rather than simple forward ray-tracing. Whole
questions that never arose for extended sources, such as how to pair predicted images with observed ones when the
model predicts too many or too few, become central to the likelihood (the workspace covers these in detail).

__Wrap Up__

In this tutorial, we met point sources and the tools PyAutoLens uses to model them. Let's summarise what we've
learnt:

- **Point Sources**: Quasars and supernovae are so much smaller than a telescope's resolution that each of their
  multiple images appears as a single unresolved point of light — no arcs, no rings.

- **The Lens Equation**: Point-source modeling means solving $\beta = \theta - \alpha(\theta)$ for the image
  positions $\theta$ given the source position $\beta$ — a non-linear inverse problem with no analytic solution,
  in contrast to the cheap forward evaluation used for extended sources.

- **Point Solver**: The `PointSolver` solves the lens equation numerically, ray-tracing progressively finer
  triangles until every multiple image is located to sub-pixel precision, and filtering out the demagnified
  central image.

- **Observables**: A point-source dataset consists of image positions, and optionally fluxes and time delays.
  Positions are the bedrock; fluxes are often compromised by microlensing and substructure; time delays enable
  measurement of the Hubble constant.

When you are ready to model real lensed quasars and supernovae — composing mass models, fitting observed positions
with a non-linear search, and including fluxes and time delays — the `autolens_workspace/scripts/point_source`
package is the place to go, starting with its `start_here` example.

In the next tutorial, we return to the lensing formalism itself: where deflection angles come from, and the
quantities (convergence, potential, magnification) that describe a mass distribution's lensing power.
"""
