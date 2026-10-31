"""
Tutorial 2: Ray Tracing
======================

Strong gravitational lensing occurs when the mass of a foreground galaxy curves
space-time, bending the paths of light rays from a background source galaxy.
The source can then appear as arcs, a ring or multiple images.

In the previous tutorial, we used light profiles to describe how a galaxy's
brightness varies across an image. We now introduce **mass profiles**, which
describe how its mass is distributed. From a mass profile, we can calculate
the **deflection angles** that tell us how the galaxy bends light.

These two ingredients let us construct a lensed image. The mass profile tells
us where each light ray came from in the source galaxy, and the source's light
profile tells us how bright that location is.

Let's revisit both views of strong lensing from the previous tutorial. The
schematic shows the geometry from the side: light travels from the source,
past the lens and towards the observer. It is not drawn to scale.

![Side-on illustration of light from a source galaxy passing a lens galaxy and reaching JWST](https://raw.githubusercontent.com/PyAutoLabs/autolens_assistant/main/docs/images/strong_lens_schematic.png)

The JWST image of the COSMOS-Web Ring shows the view on the sky. The bright
central galaxy is the lens, and the surrounding ring is light from the more
distant source galaxy.

![The COSMOS-Web Ring, labelled with the lens galaxy, lensed source galaxy and extra galaxy](https://raw.githubusercontent.com/PyAutoLabs/autolens_assistant/main/docs/images/cosmos_web_ring_rgb_labelled.png)

The extra galaxy labelled in both figures is distinct from the main lens and
lensed source. As in the previous tutorial, we leave systems with additional
galaxies until Chapter 4 and use one lens galaxy and one source galaxy here.

These figures are also used in the [COSMOS-Web Ring guided example](https://github.com/PyAutoLabs/autolens_assistant/blob/main/docs/colab/cosmos_web_ring_colab.py).

In our calculation, we follow the light paths backwards: we start with
positions in the image seen by the observer and trace rays back to positions
at the source.

This process is called **ray tracing**. It allows us to work out what the
source's light looks like after gravitational lensing.

We describe the positions using two coordinate planes:

- The **image plane** contains the angular positions at which we see light
  on the sky.
- The **source plane** contains the angular positions of the source before
  its light is distorted by the lens.

For the lensing calculation, we also represent the foreground galaxy's mass
on a **lens plane**. The lens and source planes are imagined sheets
perpendicular to our line of sight, at the respective distances of the lens
and source. They do not describe the orientations of the galaxies themselves.

In this tutorial, we use one foreground lens galaxy and one background source
galaxy. We will first trace the rays ourselves using grids, then use a `Tracer`
to perform the same calculation for us.

The resulting **lensed image** shows how the foreground galaxy's gravity
distorts the appearance of the source.

__Contents__

- **Grid:** Set up the image-plane coordinates where we will calculate the light.
- **Mass Profiles:** Describe the lens galaxy's mass and calculate its deflection angles.
- **Ray Tracing Grids:** Use the lens equation to map image-plane coordinates to the source plane.
- **Mappings:** Visualise how image-plane positions map to source-plane positions.
- **Ray Tracing Images:** Evaluate the source's light profile on the ray-traced grid.
- **Galaxies:** Combine light and mass profiles into lens and source galaxies.
- **Tracer:** Perform the ray-tracing calculation using a `Tracer`.
- **Wrap Up:** Review what we have learned and the next steps.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import autolens as al
import autoarray as aa
import autoarray.plot as aaplt
import autolens.plot as aplt


"""
__Grid__

In the previous tutorial, we evaluated light profiles on grids of (y,x)
coordinates. Each coordinate specified a position at which to calculate
the brightness.

We begin in the same way here, creating a uniform grid called
`image_plane_grid`. Its coordinates represent positions on the sky where
we want to calculate the lensed image.

The grid has 101 rows and 101 columns, with adjacent pixel centres separated
by 0.1 arcseconds. As before, coordinates are measured relative to the centre
of the grid.

Ray tracing will map each of these image-plane coordinates to a position
in the source plane. Although our starting grid is uniform, the mapped
source-plane coordinates will generally not be evenly spaced.
"""

image_plane_grid = al.Grid2D.uniform(
    shape_native=(101, 101), pixel_scales=0.1
)

"""
__Mass Profiles__

A mass profile is a mathematical description of how mass is distributed
within a galaxy. Just as a light profile lets us calculate brightness,
a mass profile lets us calculate how the galaxy deflects light.

Mass profiles are available through `al.mp`. We will begin with
`IsothermalSph`, a spherical isothermal mass profile. This is a simple
model commonly used to illustrate galaxy-scale lensing.

For this profile, we specify two parameters:

- `centre`: The (y,x) position of the mass profile's centre, in arcseconds.
  We place it at (0.0, 0.0), the centre of our grid.
- `einstein_radius`: The angular scale of the lensing deflection, in
  arcseconds. For this spherical profile, a point source exactly behind
  the lens centre would form a ring at this radius.

We set the Einstein radius to 1.6 arcseconds. For fixed lens and source
distances, a larger Einstein radius corresponds to a more massive lens
within that radius.

Printing the profile shows the parameters we have supplied.
"""

sis_mass_profile = al.mp.IsothermalSph(
    centre=(0.0, 0.0),  # The profile centre in (y,x) arcseconds.
    einstein_radius=1.6,  # The Einstein radius in arcseconds.
)

print(sis_mass_profile)

"""
In the previous tutorial, we called `image_2d_from` to evaluate a light
profile's intensity at every coordinate on a grid.

For a mass profile, we call `deflections_yx_2d_from` to calculate the
deflection angles at those coordinates.

Each position has two deflection components, ordered as (y,x), both
measured in arcseconds. Together, they describe the angular displacement
we will use to map that image-plane position to the source plane.

The following calculation gives us one deflection vector for every
coordinate in `image_plane_grid`.
"""

deflections = sis_mass_profile.deflections_yx_2d_from(
    grid=image_plane_grid
)

"""
As with the grids in the previous tutorial, we can access the deflections
using `native` and `slim`.

- `native` preserves the image layout. Its shape is
  (101, 101, 2): one pair of (y,x) deflection components for each pixel.
- `slim` lists the same pairs in pixel order. Its shape is
  (10201, 2): one row per pixel, with two components in each row.

The final dimension has length 2 because a deflection has both a y
component and an x component.

Below, we print the deflection at the first pixel using both representations.
The two outputs should be identical. We then print the deflection at the
next pixel using `slim`.
"""

print("Deflection at the first pixel, using native:")
print(deflections.native[0, 0])

print("Deflection at the first pixel, using slim:")
print(deflections.slim[0])

print("Deflection at the next pixel, using slim:")
print(deflections.slim[1])

"""
A grid tells us where we are evaluating the lensing calculation. A deflection
vector tells us the angular displacement associated with each of those
positions.

Both use pairs of numbers, but the pairs have different meanings: a grid
stores (y,x) positions, whereas the deflections store (y,x) components of
a displacement.

PyAutoLens stores these deflections in a `VectorYX2D` object. This keeps
the vectors together with the grid of positions where they were calculated.
"""

print(type(deflections))

r"""
The `grid` attribute gives us the positions where the deflections were
calculated. Here, these are the coordinates of `image_plane_grid`.

The `magnitudes` attribute gives the length of each deflection vector,
combining its two components:

$\lvert \alpha \rvert = \sqrt{\alpha_y^2 + \alpha_x^2}$

The magnitude is measured in arcseconds, just like the individual components.

Below, we print the position of the first pixel and the magnitude of its
deflection. For our spherical isothermal profile, the deflection magnitude
away from the centre is equal to the Einstein radius, so we expect
approximately 1.6 arcseconds at this pixel.
"""

print("Position of the first pixel (y,x), in arcseconds:")
print(deflections.grid.native[0, 0])

print("Deflection magnitude at the first pixel, in arcseconds:")
print(deflections.magnitudes.native[0, 0])

"""
We can see how the deflections vary across the image by plotting their
y and x components separately.

For each component, we construct an `Array2D` using the image-plane grid's
mask. This places each value at the pixel where it was calculated.

The colours show the signed deflection component in arcseconds. In the
y-component plot, the sign changes across the horizontal line through the
lens centre. In the x-component plot, it changes across the vertical line.

For this spherical profile, the components vary with direction even though
the total deflection magnitude is constant away from the centre. Compare
the two plots: their patterns should be related by a rotation.
"""

deflections_y = aa.Array2D(
    values=deflections.slim[:, 0],
    mask=image_plane_grid.mask,
)
aplt.plot_array(
    array=deflections_y,
    title="Y Deflection (arcseconds)",
)

deflections_x = aa.Array2D(
    values=deflections.slim[:, 1],
    mask=image_plane_grid.mask,
)
aplt.plot_array(
    array=deflections_x,
    title="X Deflection (arcseconds)",
)

"""
A mass profile also lets us calculate other quantities that describe
the lens and its effect on the source's appearance:

- **Convergence** describes the projected mass per unit area, divided by
  a reference surface density called the critical surface density.
  This reference depends on the lens and source redshifts, which set
  the relevant distances for a chosen cosmology. Convergence is
  therefore dimensionless.

- **Lensing potential** is a mathematical function whose spatial variation
  determines the deflection angles. Its gradient gives the deflection
  vector. With our coordinates and deflections expressed in arcseconds,
  the potential is expressed in arcseconds squared.

- **Magnification** describes how lensing changes the apparent area of a
  small source region. Lensing preserves surface brightness, but a larger
  image covers more sky and therefore contains more total light. The
  absolute magnification gives the local area and flux factor; its sign
  records whether the image orientation is reversed.

We will explore these quantities further as we develop the lensing
formalism. For now, we calculate them on our image-plane grid.

Convergence and potential are calculated directly from the mass profile.
For magnification, we use `LensCalc`, which calculates it from the way
the deflections vary across the grid. Each result is an `Array2D`.
"""

convergence = sis_mass_profile.convergence_2d_from(
    grid=image_plane_grid
)

potential_2d = sis_mass_profile.potential_2d_from(
    grid=image_plane_grid
)

magnification_2d = al.LensCalc.from_mass_obj(
    mass_obj=sis_mass_profile
).magnification_2d_from(grid=image_plane_grid)

"""
Let's plot the convergence and lensing potential.

The convergence is largest near the centre and decreases outwards,
showing how the projected mass density falls with distance from the
centre of this profile.

The spherical isothermal profile is singular at its exact centre:
its idealised density grows without bound there. The central pixel
therefore needs care when interpreting the numerical plot.

The potential has a different pattern, increasing with distance from
the centre for this profile. It is not another image of the mass density.
Its spatial variation determines the deflections.

Both plots should be circularly symmetric because we chose a spherical
mass profile.
"""

aplt.plot_array(
    array=convergence,
    title="Convergence",
)

aplt.plot_array(
    array=potential_2d,
    title="Lensing Potential (arcseconds squared)",
)

"""
We can also examine the convergence along a line extending outwards
from the mass profile's centre.

Because our profile is spherical, the result is the same in every
direction. We will sample along the positive x direction, at radii
from 0.1 to 5.0 arcseconds. Starting away from zero avoids the
singularity at the centre of the isothermal profile.

The convergence should decrease as we move outwards. For this profile,
doubling the radius halves the convergence.
"""

import numpy as np

radii = np.linspace(0.1, 5.0, 100)

radial_grid = al.Grid2DIrregular(
    values=[
        (sis_mass_profile.centre[0], sis_mass_profile.centre[1] + radius)
        for radius in radii
    ]
)

convergence_1d = sis_mass_profile.convergence_2d_from(grid=radial_grid)

plt.plot(radii, convergence_1d)
plt.xlabel("Radius from profile centre (arcseconds)")
plt.ylabel("Convergence")
plt.show()
plt.close()

"""
In the convergence image, the large values near the centre can make
the variation farther out difficult to see.

A logarithmic colour scale helps us display a wide range of positive
values. Equal steps in colour then represent equal ratios of values,
rather than equal differences.

Below, we plot convergence and potential using logarithmic colour
scales. Compare these with the earlier plots: the underlying quantities
are unchanged, but different parts of their variation become easier
to distinguish.

A logarithmic scale only represents positive values. The singular
centre of this idealised profile should still be treated with care.
"""

aplt.plot_array(
    array=convergence,
    title="Convergence (logarithmic colour scale)",
    use_log10=True,
)

aplt.plot_array(
    array=potential_2d,
    title="Lensing Potential (logarithmic colour scale)",
    use_log10=True,
)

r"""
__Ray Tracing Grids__

We can now map image-plane coordinates to the source plane using
the **lens equation**:

$\beta = \theta - \alpha(\theta)$

Here, $\theta$ is an image-plane position, $\alpha(\theta)$ is its
deflection vector, and $\beta$ is the corresponding source-plane
position. Each has (y,x) components measured in arcseconds.

For example, at image-plane position (0.0, 2.0), the deflection is
(0.0, 1.6), giving source-plane position (0.0, 0.4) arcseconds.

The method below performs this subtraction for every grid coordinate.
"""

deflections = sis_mass_profile.deflections_yx_2d_from(
    grid=image_plane_grid
)

source_plane_grid = image_plane_grid.grid_2d_via_deflection_grid_from(
    deflection_grid=deflections
)

"""
Let's plot the image-plane grid and its mapped source-plane coordinates.

The image-plane grid is uniform, but the source-plane coordinates
are unevenly spaced. Each point still corresponds to one original
image-plane pixel.

For our spherical isothermal profile, the deflection magnitude is
1.6 arcseconds everywhere away from the centre. The direction varies
across the grid, producing the pattern below; points nearer the
centre are not deflected more strongly.
"""

aplt.plot_grid(
    grid=image_plane_grid,
    title="Image Plane Grid",
)

aplt.plot_grid(
    grid=source_plane_grid,
    title="Source Plane Grid (Ray Traced)",
)

"""
The source-plane grid covers a wide area, but many of its points
land close together near the centre. Zooming in helps us see
their spacing.

This is related to magnification: a small region of the source
can appear over a much larger area in the image plane. When we
trace those image-plane points back, many therefore land within
that small source-plane region.

We often zoom in when plotting source planes so that this central
detail is easier to see. The plot below shows the central
2-by-2 arcseconds. Only the displayed limits change; the grid
and ray-tracing calculation stay the same.

For these grid displays, we use `autoarray.plot` (imported as `aaplt`),
which provides the `extent` and grouped `indexes` options.
"""

aaplt.plot_grid(
    grid=source_plane_grid,
    title="Source Plane (Zoomed In)",
    extent=(-1.0, 1.0, -1.0, 1.0),
)

"""
__Mappings__

Each image-plane coordinate maps to a source-plane coordinate.
We can show this by highlighting groups of points in matching colours.

As in the previous tutorial, indexes refer to positions in the
grid's `slim` ordering. Each inner list below selects five points;
the plotting function gives each group a different colour.

We use the same indexes in both panels because ray tracing preserves
the order of the points. Follow each coloured group from the image
plane to the source plane and look at how its spacing changes.

These indexes are chosen for our 101-by-101 grid. Try selecting
different indexes to follow other parts of the grid.

To inspect the central points more closely, try changing `extent`
in the source-plane plot to (-1.0, 1.0, -1.0, 1.0). The four values
set the minimum and maximum x coordinates, then the minimum and
maximum y coordinates. Restore the limits below to compare both
planes at the same scale.
"""

indexes = [
    [5118, 5119, 5120, 5121, 5122],
    [3078, 3079, 3080, 3081, 3082],
    [7098, 7099, 7100, 7101, 7102],
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

aaplt.plot_grid(
    grid=image_plane_grid,
    indexes=indexes,
    ax=axes[0],
    title="Image Plane",
    extent=(-5.5, 5.5, -5.5, 5.5),
)

aaplt.plot_grid(
    grid=source_plane_grid,
    indexes=indexes,
    ax=axes[1],
    title="Source Plane",
    extent=(-5.5, 5.5, -5.5, 5.5),
)

plt.tight_layout()
plt.show()
plt.close(fig)

"""
__Ray Tracing Images__

The source-plane grid tells us where the light at each image-plane
position originated. To calculate its brightness, we evaluate a
source light profile at those mapped positions.

We use an `ExponentialCore` light profile centred at (0.1, 0.1)
arcseconds in the source plane. This places the source slightly
away from exact alignment with the lens centre.

Each calculated intensity remains associated with its original
image-plane pixel. The resulting array therefore describes the
lensed image seen on the sky, even though we evaluated the light
profile at source-plane coordinates.

Different parts of the image plane can map onto the same region
of the source. This is how one source galaxy can appear as
multiple images or extended arcs.
"""

source_light_profile = al.lp.ExponentialCore(
    centre=(0.1, 0.1),
    ell_comps=(0.0, 0.1),
    intensity=0.1,
    effective_radius=0.2,
)

lensed_image = source_light_profile.image_2d_from(
    grid=source_plane_grid
)

"""
Let's plot the lensed source. Look for its light stretched around
the lens centre into a ring-like image.

This image is calculated from our chosen mass and light profiles.

Try changing the source profile's `centre` in the preceding cell:

- Start with (0.0, 0.0) arcseconds to align the source with the lens.
- Then try (0.0, 0.5) arcseconds to move the source off-centre.

After each change, rerun the source-profile and `lensed_image`
calculations, then plot the result below. Moving the source off-centre
should make the lensed light less evenly distributed around the lens.

Restore `centre=(0.1, 0.1)` and rerun those cells before continuing.
"""

aplt.plot_array(
    array=lensed_image,
    title="Lensed Source Image",
)

"""
__Galaxies__

In the previous tutorial, we put light profiles inside `Galaxy`
objects. A galaxy can also contain mass profiles.

Let's place our mass profile in a lens galaxy at redshift 0.5,
and our source light profile in a source galaxy at redshift 1.0.

Our lens galaxy contains only a mass profile, so it deflects light
but contributes no light of its own to this example. The source
galaxy contains the light profile whose lensed image we have
just calculated.

Printing the galaxies shows their redshifts and profiles.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=sis_mass_profile,
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=source_light_profile,
)

print(lens_galaxy)
print(source_galaxy)

"""
__Tracer__

A `Tracer` brings our galaxies together and performs the ray tracing
for us.

We supply the lens and source galaxies. Their redshifts identify
which galaxy is in front and which is behind.
"""

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Like a light profile or galaxy, a `Tracer` has an `image_2d_from`
method.

We pass it the original image-plane grid. It calculates the
deflections, traces the grid to the source plane and evaluates
the source's light there.

The plot should reproduce the lensed image we calculated manually,
because we supplied the same mass and light profiles.

A tracer also includes any light emitted by the lens galaxy.
Here, there is none, so the image contains only the lensed source.
"""

image = tracer.image_2d_from(grid=image_plane_grid)

aplt.plot_array(
    array=image,
    title="Lensed Image via Tracer",
)

"""
We can display several views of the lens system together using
`aplt.subplot_tracer`.

Compare the lensed image with the source-plane image, which shows
the source light profile before lensing. The lens-light panel
contains no emission because our lens galaxy has no light profile.

The remaining panels show convergence, potential, the two
deflection components and magnification. These describe the same
mass profile we examined earlier.

**🤖 Ask AI:** How does moving the source away from alignment with the
lens change an Einstein ring into separate arcs or images?
"""

aplt.subplot_tracer(tracer=tracer, grid=image_plane_grid)

"""
The `Tracer` can also return the grids used in its ray-tracing
calculation, through `traced_grid_2d_list_from`.

For our two-galaxy system, the first entry is the original
image-plane grid and the second is the ray-traced source-plane
grid. These correspond to the two grids we calculated manually.
"""

traced_grid_list = tracer.traced_grid_2d_list_from(
    grid=image_plane_grid
)

image_plane_grid_traced = traced_grid_list[0]
source_plane_grid_traced = traced_grid_list[1]

r"""
__Wrap Up__

In this tutorial, we built a lensed image step by step:

- A **mass profile** describes the lens galaxy's mass distribution
  and lets us calculate its deflection angles.
- The **lens equation**, $\beta = \theta - \alpha(\theta)$, maps
  image-plane coordinates to source-plane coordinates.
- Evaluating a source **light profile** at those mapped coordinates
  gives the brightness of each pixel in the lensed image.
- **Galaxy** objects hold the profiles and redshifts of the lens
  and source.
- A **Tracer** brings these galaxies together and performs the
  calculation for us.

The manual calculation and the `Tracer` describe the same lens
system. Working through both shows what happens when we ask a
tracer to calculate an image.

In Tutorial 3, we will build on this example and explore more
of the ray-tracing calculation.
"""
