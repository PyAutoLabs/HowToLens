r"""
HowToLens: Introduction
======================

A distant galaxy can appear as multiple images, stretched arcs or a ring when its light passes a sufficiently
well-aligned foreground galaxy. The foreground galaxy’s gravity deflects the light, allowing it to reach us
along different paths. This is strong gravitational lensing.

We call the foreground galaxy the **lens** and the background galaxy the **source**. The side-on illustration
below shows two paths by which light from the source can reach the observer. It illustrates the geometry;
it is not drawn to scale.

![Side-on illustration of light from a source galaxy passing a lens galaxy and reaching JWST](https://raw.githubusercontent.com/PyAutoLabs/autolens_assistant/main/docs/images/strong_lens_schematic.png)

Looking towards the lens, we see the source’s distorted images around it. The JWST image below shows the
COSMOS-Web Ring: the bright central galaxy is the lens, and the surrounding ring is light from the more
distant source galaxy.

![The COSMOS-Web Ring, labelled with the lens galaxy, lensed source galaxy and extra galaxy](https://raw.githubusercontent.com/PyAutoLabs/autolens_assistant/main/docs/images/cosmos_web_ring_rgb_labelled.png)

Both figures label an **Extra Galaxy**. This is another galaxy visible near the ring in the image, distinct
from the main lens and the lensed source. Its light can overlap the region we want to model. Appearing nearby
on the sky does not by itself tell us whether it is at the same distance as the lens.

For now, the strong lenses we consider in this chapter have one lens galaxy. Chapter 4 introduces systems
with additional galaxies and explains how to account for their light and, where relevant, their gravitational
effect on the source images.

These figures are also used in the [COSMOS-Web Ring guided example](https://github.com/PyAutoLabs/autolens_assistant/blob/main/docs/colab/cosmos_web_ring_colab.py).

By modelling these images, astronomers can study the distribution of mass in the lens, including dark matter,
and reconstruct the source galaxy’s light. Some lens systems also allow measurements of the Universe’s expansion.

**HowToLens** teaches you to build and fit these models using **PyAutoLens**. We begin with simple descriptions
of galaxy light, then introduce gravitational lensing and the statistical methods used to compare models with data.

Tutorial 1: Grids and Galaxies
=============================

In this tutorial, we will build an image of a galaxy from light profiles and explore how its appearance changes when we
adjust its position, orientation, size and brightness.

We start with a grid of positions on the sky, written in `(y, x)` order. A **light profile** is a mathematical
function that describes how a galaxy’s light varies with position. It is a model: a simplified mathematical
description of the galaxy. Evaluating it on the grid gives us a **light-profile image**. Later, we will
include the telescope’s blurring to construct a **model image** that can be compared with observed data.

To describe a profile centred and oriented anywhere in the image, we express the grid coordinates relative
to that profile. We will work through these coordinate transformations before using the library’s light
profiles and galaxy objects.

The examples evaluate models with parameters that we choose. They do not yet fit a model to observed data.

We introduce the deflection of light by gravity in the next tutorial.

__Contents__

- **Grids:** Create image coordinates and inspect their native and slim representations.
- **Geometry:** Express coordinates relative to a profile’s centre, orientation and ellipticity.
- **Light Profiles:** Calculate Sérsic intensities and build light-profile images.
- **One Dimension Projection:** Sample intensity along a profile’s major axis.
- **Galaxies:** Combine light profiles and galaxies, and compare their contributions.
- **Units:** Convert angular sizes into physical sizes using a cosmology.
- **Wrap Up:** Review the main ideas.
- **Advanced Topics:** Optional notes on units and numerical accuracy.
- **Other Unit Conversion:** Understand what brightness calibration requires.
- **Over Sampling:** Improve the numerical calculation within each pixel.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
import autolens as al
import autolens.plot as aplt

"""
__Grids__

To build a galaxy image, we need positions at which to evaluate its light profile. A `Grid2D` stores these
positions in `(y, x)` order. For a uniform image grid, each coordinate marks the centre of a pixel.

We express positions in **arcseconds**, a unit of angle equal to 1/3600 of a degree. This describes distances
across the sky rather than physical distances within the galaxy.

The grid below has 101 rows and 101 columns. Setting `pixel_scales=0.1` gives square pixels that span
0.1 arcseconds on each side.

The image covers 10.1 by 10.1 arcseconds, with edges at −5.05 and +5.05 arcseconds. The pixel centres run
from −5.0 to +5.0 arcseconds along each axis, with the central pixel at `(0.0, 0.0)`.
"""
grid = al.Grid2D.uniform(
    shape_native=(101, 101),
    pixel_scales=0.1,
)

"""
Plot the grid to see where we will evaluate the light profile. Each dot marks a pixel centre, with neighbouring
dots separated by 0.1 arcseconds horizontally and vertically.

This figure shows positions only. We have not yet assigned a brightness to any pixel.
"""
aplt.plot_grid(grid=grid, title="Uniform Grid of Coordinates")

"""
We can inspect individual coordinates using `grid.native[row, column]`. Rows and columns are numbered from
zero, and each entry contains a `(y, x)` pair in arcseconds.

The first three entries below lie along the top row: their y coordinate stays at +5.0 while x increases
from −5.0 in steps of 0.1. The final entry starts the next row, where y decreases to +4.9.
"""
print("Row 0, column 0 (y, x):", grid.native[0, 0])
print("Row 0, column 1 (y, x):", grid.native[0, 1])
print("Row 0, column 2 (y, x):", grid.native[0, 2])
print("Row 1, column 0 (y, x):", grid.native[1, 0])

"""
The same grid coordinates can be arranged in two ways:

- **`native`** preserves the rows and columns of the image. Use `grid.native[row, column]` to select a pixel’s `(y, x)` coordinates.
- **`slim`** lists the coordinate pairs in row order. Use `grid.slim[index]` to select a pair from that list.

These are different arrangements of the same positions, not different grids. The two expressions below
return the coordinates of the same pixel.
"""
print("First coordinate via native:", grid.native[0, 0])
print("First coordinate via slim:", grid.slim[0])

"""
For this unmasked grid, `native` has shape `(101, 101, 2)`: 101 rows, 101 columns and two values per pixel—the
y and x coordinates.

The `slim` representation has shape `(10201, 2)`: one coordinate pair for each of the 101 × 101 pixels.
The final dimension remains two because each position still needs both y and x.

Print the shapes below to check how the coordinates are arranged.
"""
print("Native shape:", grid.native.shape)
print("Slim shape:", grid.slim.shape)

"""
Try changing the grid definition above, then rerun the plotting and coordinate-inspection cells:

- Keep `shape_native=(101, 101)` and change `pixel_scales` from `0.1` to `0.2`. The number of points stays the same, but their spacing doubles and the image covers 20.2 by 20.2 arcseconds.
- Restore `pixel_scales=0.1`, then change `shape_native` to `(51, 51)`. The spacing stays the same, but fewer pixels cover a smaller region: 5.1 by 5.1 arcseconds.

Compare the axis ranges in the plots and the printed array shapes. Changing pixel scale changes the angular
spacing; changing the number of pixels changes how many positions we sample.

Before continuing, restore `shape_native=(101, 101)` and `pixel_scales=0.1`, then rerun the grid-definition cell.

__Geometry__

Our grid describes positions relative to the image centre, `(0.0, 0.0)`. To evaluate a galaxy’s light profile,
we first express those positions relative to the galaxy’s centre.

For a galaxy centred at `(0.3, 0.5)` arcseconds, subtract 0.3 from every y coordinate and 0.5 from every
x coordinate. The galaxy’s centre then has coordinates `(0.0, 0.0)` in this new reference frame.

This changes how we describe the positions; it does not move the pixels on the sky.
"""
centre = (0.3, 0.5)

grid_shifted = grid.copy()  # Preserve the original image coordinates.
grid_shifted[:, 0] = grid_shifted[:, 0] - centre[0]
grid_shifted[:, 1] = grid_shifted[:, 1] - centre[1]

print("First coordinate relative to the galaxy centre:", grid_shifted.native[0, 0])

"""
The first coordinate changes from `(5.0, -5.0)` to `(4.7, -5.5)` arcseconds. The grid’s spacing and shape
are unchanged, but its coordinates now describe offsets from the galaxy’s centre.

In the plot below, the original image centre lies at `(-0.3, -0.5)`. The origin of this coordinate frame
represents the galaxy’s centre.

Try setting `centre=(0.0, 0.0)` and rerunning the shift and plot cells: the coordinates should match the
original grid. Restore `centre=(0.3, 0.5)` and rerun both cells before continuing.
"""
aplt.plot_grid(
    grid=grid_shifted,
    title="Coordinates Relative to the Galaxy Centre",
)

"""
An elliptical light profile also has an orientation. Its **position angle** is measured anticlockwise
from the positive x-axis to its major axis.

To express our coordinates in a frame aligned with that profile, we apply the inverse rotation. For a
position angle of 60 degrees, we rotate the coordinate values clockwise by 60 degrees.

The code below finds each point’s distance and angle from the galaxy’s centre, subtracts the profile’s
position angle, and converts back to `(y, x)` coordinates. `np.arctan2(y, x)` determines the angle while
accounting for which quadrant contains the point.
"""
angle_degrees = 60.0

y = grid_shifted[:, 0]
x = grid_shifted[:, 1]

radius = np.sqrt(y**2 + x**2)
theta = np.arctan2(y, x) - np.radians(angle_degrees)

grid_rotated = grid_shifted.copy()  # Preserve the unrotated offsets.
grid_rotated[:, 0] = radius * np.sin(theta)
grid_rotated[:, 1] = radius * np.cos(theta)

print("First coordinate in the profile frame:", grid_rotated.native[0, 0])

"""
The coordinates now describe positions in the profile’s reference frame, with its major axis along x.
The transformation preserves each point’s distance from the galaxy’s centre.

Plot the transformed grid below. Its orientation has changed because we rotated the coordinate values
clockwise by 60 degrees.

Try setting `angle_degrees=0.0` and rerunning the rotation and plot cells. The result should match
`grid_shifted`. Then try `30.0` to see a smaller rotation. Restore `60.0` and rerun both cells before continuing.
"""
aplt.plot_grid(
    grid=grid_rotated,
    title="Coordinates in the Profile Frame",
)

r"""
We can now assign an **elliptical radius** to each position. Points with the same elliptical radius lie
on an ellipse centred on the galaxy.

Let $q=b/a$ be the ratio of the ellipse’s minor axis to its major axis. A circle has $q=1$; smaller values
describe more elongated ellipses. In the profile frame, the major axis lies along x, and the elliptical radius is

$\eta = \sqrt{x_r^2 + \left(\frac{y_r}{q}\right)^2}$.

Here, $y_r$ and $x_r$ are the shifted and rotated coordinates in arcseconds, so $\eta$ is also in arcseconds.

For $q=0.5$, a point at `(y, x) = (0, 1)` has elliptical radius 1 arcsecond, while `(1, 0)` has elliptical
radius 2 arcseconds. Offsets along the minor axis therefore correspond to larger elliptical radii.

The code below calculates one radius per grid position. It does not move the grid points. Try
`axis_ratio=1.0` to recover ordinary distance from the centre, then restore `0.5` before continuing.
"""
axis_ratio = 0.5

y_rotated = grid_rotated[:, 0]
x_rotated = grid_rotated[:, 1]

eta = np.sqrt(x_rotated**2 + (y_rotated / axis_ratio) ** 2)

print("First ten elliptical radii (arcseconds):", eta[:10])

r"""
The axis ratio $q$ and position angle $\phi$ together describe the ellipse’s shape and orientation.
The library packages the same information into two **ellipticity components**, passed as `ell_comps`:

$\epsilon_y = \frac{1-q}{1+q}\sin(2\phi)$,
$\epsilon_x = \frac{1-q}{1+q}\cos(2\phi)$.

The component order is `(epsilon_y, epsilon_x)`. These are dimensionless shape parameters, not spatial coordinates.

The doubled angle reflects the symmetry of an ellipse: rotating it by 180 degrees leaves its appearance
unchanged. For a circle, $q=1$, both components are zero and the orientation has no effect.

The code below converts our axis ratio of 0.5 and position angle of 60 degrees into approximately
`(0.289, -0.167)`. We convert degrees to radians because NumPy’s trigonometric functions expect radians.
"""
ellipticity = (1 - axis_ratio) / (1 + axis_ratio)
angle_radians = np.radians(angle_degrees)

epsilon_y = ellipticity * np.sin(2 * angle_radians)
epsilon_x = ellipticity * np.cos(2 * angle_radians)

ell_comps = (epsilon_y, epsilon_x)

print("Ellipticity components:", ell_comps)

r"""
__Light Profiles__

We now have the geometry needed to describe an elliptical galaxy. A light profile assigns a brightness to
each position using its distance from the galaxy’s centre.

A commonly used model is the **Sérsic profile**:

$I(r)=I_e\exp\left\{-k\left[\left(\frac{r}{R_e}\right)^{1/n}-1\right]\right\}$.

Its parameters are:

- $I_e$: the intensity at the effective radius, which sets the brightness scale.
- $R_e$: the effective radius, which describes the galaxy’s size.
- $n$: the Sérsic index, which controls the radial shape of the profile.
- $k$: a value determined by $n$, chosen so that the effective radius encloses half the model’s total light.

To match the library’s convention, we use the **circularised radius** $r=\sqrt{q}\,\eta$. A circle of radius
$r$ has the same area as the ellipse with major-axis radius $\eta$ and axis ratio $q$. Accordingly, $R_e$
is the circularised size of the half-light ellipse; it does not generally describe a circular aperture
containing half the light.

For this worked calculation, we use the simple approximation $k\approx2n-1/3$. The half-light interpretation
is therefore approximate here; the library uses a more accurate expression.

The code below evaluates the profile at a circularised radius of 0.5 arcseconds, using $R_e=2.0$ arcseconds,
$n=1.0$, and an intensity scale of 1.0 in arbitrary units.
"""
radius = 0.5  # Circularised radius in arcseconds.
intensity = 1.0
effective_radius = 2.0
sersic_index = 1.0

k = 2 * sersic_index - 1 / 3

sersic_value = intensity * np.exp(
    -k * ((radius / effective_radius) ** (1 / sersic_index) - 1)
)

print("Intensity at radius 0.5 arcseconds:", sersic_value)

"""
The result is approximately 3.49 in our chosen intensity units. It exceeds `intensity=1.0` because this
position lies inside the effective radius, where the profile is brighter.

To build an image, apply the same calculation at every grid position. First convert the elliptical radii
`eta` to circularised radii, then evaluate the Sérsic function for the whole array.
"""
circularised_radii = np.sqrt(axis_ratio) * eta

sersic_image = intensity * np.exp(
    -k * ((circularised_radii / effective_radius) ** (1 / sersic_index) - 1)
)

"""
Plot the resulting brightness values at their original image positions. Look for a galaxy centred at
`(0.3, 0.5)` arcseconds, with its major axis oriented 60 degrees anticlockwise from the positive x-axis
and an axis ratio of 0.5.

The clockwise coordinate transformation lets us evaluate this anticlockwise-oriented profile. We then
display the calculated brightnesses on the original image grid.

Try changing `intensity` from `1.0` to `2.0`, then rerun the intensity calculations and plot. Every calculated
intensity should double, although automatic colour scaling may make the image look unchanged. Check the
printed value and colour-bar labels.

Restore `intensity=1.0` and rerun those cells before continuing.
"""
aplt.plot_array(
    array=aa.Array2D(values=sersic_image, mask=grid.mask),
    title="Sérsic Image",
)

"""
We have worked through the geometry and intensity calculation explicitly. A `Sersic` light-profile object
brings these steps together: we supply its parameters, and the library evaluates the profile.

The `lp` namespace contains the library’s light profiles. Below, we create a Sérsic profile centred at
`(0.0, 0.0)`, with ellipticity components `(0.0, 0.1)`. These describe a mildly elliptical profile whose
major axis lies along x.

This is a different example from our manual calculation: its centre and ellipticity have changed.
The intensity, effective radius and Sérsic index retain their previous values.
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=2.0,
    sersic_index=1.0,
)

print(sersic_light_profile)

"""
Pass the original image grid to `image_2d_from()`. The profile handles the shift into its own reference frame,
its orientation and the intensity calculation.

Use `grid`, rather than `grid_shifted` or `grid_rotated`: those transformed coordinates were part of our
manual demonstration.

Look for a galaxy centred on the origin and elongated along x. Its appearance differs from the previous
image because we changed the profile parameters. The library also uses a more accurate Sérsic constant
and samples within pixels to improve the image calculation; the optional over-sampling section explains
this later.

This evaluates the model we specified. It does not fit the model to data.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

aplt.plot_array(array=image, title="Sérsic Image via Light Profile")

"""
The result is an `Array2D`, which stores one intensity value per image pixel. Like the grid, it supports
`native` and `slim` representations:

- `image.native` arranges the values into image rows and columns, with shape `(101, 101)`.
- `image.slim` lists the values in row order, with shape `(10201,)`.

Unlike grid coordinates, image intensities do not need a final dimension of size two: each pixel contains
a single value.

The two expressions below access the same pixel, so they should print the same intensity.
"""
print("First pixel via native:", image.native[0, 0])
print("First pixel via slim:", image.slim[0])

"""
__One Dimension Projection__

A light-profile image shows the galaxy’s two-dimensional appearance. We can also inspect how its intensity
changes along a line through its centre.

We still supply `(y, x)` positions to `image_2d_from()`, but those positions now lie along a line rather than
filling an image. The four points below lie along the positive x-axis, at distances of 0, 1, 2 and 3 arcseconds
from the origin.

`Grid2DIrregular` stores a list of positions without requiring a rectangular pixel grid.
"""
grid_2d_projected = al.Grid2DIrregular(
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (0.0, 3.0),
    ]
)

"""
For a smoother curve, sample more positions along the profile’s major axis. The code below places 101 points
between the galaxy’s centre and a distance of 5 arcseconds.

The position angle determines the direction of this line. Multiplying each distance by its sine and cosine
gives the y and x offsets; adding the profile centre places the line correctly in the image.

We then evaluate the light profile at these positions. This measures intensity along the major axis,
rather than averaging the image in circular rings.
"""
radii = np.linspace(0.0, 5.0, 101)
angle_radians = np.radians(sersic_light_profile.angle())

grid_2d_projected = al.Grid2DIrregular(
    np.column_stack(
        (
            sersic_light_profile.centre[0] + radii * np.sin(angle_radians),
            sersic_light_profile.centre[1] + radii * np.cos(angle_radians),
        )
    )
)

image_1d = sersic_light_profile.image_2d_from(grid=grid_2d_projected)

"""
Plot intensity against distance from the galaxy’s centre. The curve should fall outwards as the profile
becomes fainter.

The horizontal axis is distance along the major axis in arcseconds, not the circularised radius used earlier.
The vertical axis is intensity in our chosen arbitrary units—not luminosity integrated over the galaxy.

Try changing `effective_radius` in the `Sersic` definition from `2.0` to `1.0`. Recreate the profile, then rerun
the sampling and plotting cells. The profile should become more compact. Restore `effective_radius=2.0`
and rerun those cells before continuing.
"""
plt.plot(radii, image_1d)
plt.xlabel("Distance along major axis (arcseconds)")
plt.ylabel("Intensity (arbitrary units)")
plt.show()
plt.close()

"""
A galaxy’s bright centre and faint outskirts can span a wide range of intensities. On a linear colour scale,
the bright centre can dominate the image, making the outskirts difficult to see.

Logarithmic colour scaling gives more of the colour range to faint emission. Equal colour intervals represent
equal ratios of intensity, rather than equal differences.

Plot the same light-profile image with `use_log10=True` and compare it with the earlier linear-scale plot.
Look for light extending beyond the bright centre that was difficult to distinguish before. The intensity
values have not changed; we have changed how they are displayed.
"""
aplt.plot_array(
    array=image,
    title="Sérsic Light-Profile Image (Log Scale)",
    use_log10=True,
)

"""
__Galaxies__

Now, let’s introduce `Galaxy` objects. A light profile can represent a component of a galaxy, such as its
bulge or disc. By combining light profiles in a `Galaxy`, we can describe their light together.

A `Galaxy` also has a **redshift**, which describes how much the wavelengths of its light have increased.
Together with a cosmological model, this lets us convert angular sizes in arcseconds into physical sizes
in kiloparsecs. A kiloparsec is approximately 3,260 light-years.

Redshifts are especially important in strong lensing, where the foreground lens galaxy and background source galaxy
lie at two different redshifts. We are not yet performing any lensing calculations in this tutorial, so for now we
simply use a single galaxy to build up intuition for the `Galaxy` object.

Let’s create a galaxy with two Sérsic light profiles representing a bulge and a disc. In this example,
the bulge is smaller and rounder, while the disc is more extended and flattened.
"""
bulge = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

disk = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.3),
    intensity=0.3,
    effective_radius=3.0,
    sersic_index=1.0,
)

galaxy = al.Galaxy(redshift=0.5, bulge=bulge, disk=disk)

print(galaxy)

"""
We used `image_2d_from()` to calculate the image of a light profile. The same method works for a `Galaxy`:
it evaluates each light profile on the grid and adds their intensities together.

The result is an `Array2D` containing the galaxy image. Below, we print the combined intensity in each of
the first three pixels of the top row.
"""
image = galaxy.image_2d_from(grid=grid)

print("Row 0, column 0 intensity:", image.native[0, 0])
print("Row 0, column 1 intensity:", image.native[0, 1])
print("Row 0, column 2 intensity:", image.native[0, 2])

"""
Let’s plot the galaxy image. Each pixel contains light from both the bulge and the disc.

The bright central region should stand out most clearly. The disc contributes light too, but its emission
is spread over a larger area and can be difficult to distinguish on this linear colour scale.
"""
aplt.plot_array(
    array=image,
    title="Galaxy Bulge and Disc Image",
)

"""
To see what each component contributes, plot the bulge and disc separately using
`aplt.subplot_galaxy_light_profiles()`.

Look for the bulge’s concentrated centre and the disc’s broader, more flattened appearance. These are the
two light-profile images that add together to produce the galaxy image above.

Each panel chooses its own colour scale, so compare the colour-bar values as well as the colours when
judging their relative brightness.
"""
aplt.subplot_galaxy_light_profiles(galaxy=galaxy, grid=grid)

"""
Now display the combined galaxy image with logarithmic colour scaling. This makes the extended, faint
light easier to see alongside the bright centre.

Compare the shape of the outskirts with the separate disc image. The disc’s contribution should be easier
to recognise here, although each pixel still contains the sum of both components. Changing the colour
scale does not separate their light.
"""
aplt.plot_array(
    array=image,
    title="Galaxy Bulge and Disc Image (Log Scale)",
    use_log10=True,
)

"""
A radial plot lets us compare the bulge and disc intensities directly, using the same axes.

Our two profiles share a centre at `(0.0, 0.0)` and have their major axes along x. We can therefore evaluate
both at the same positions along the positive x-axis.

Look for the bulge’s high central intensity and how its contribution changes relative to the disc as you
move outwards. These curves show intensity at each position, not the total light enclosed within that radius.
"""
radii = np.linspace(0.0, 5.0, 101)

grid_2d_projected = al.Grid2DIrregular(
    np.column_stack((np.zeros_like(radii), radii))
)

bulge_image_1d = galaxy.bulge.image_2d_from(grid=grid_2d_projected)
disk_image_1d = galaxy.disk.image_2d_from(grid=grid_2d_projected)

plt.plot(radii, bulge_image_1d, label="Bulge")
plt.plot(radii, disk_image_1d, label="Disc")
plt.xlabel("Distance along major axis (arcseconds)")
plt.ylabel("Intensity (arbitrary units)")
plt.legend()
plt.show()
plt.close()

"""
Just as a `Galaxy` groups light profiles, a `Galaxies` object groups individual galaxies.

Let’s add a second galaxy, centred at `(0.2, 0.3)` arcseconds, and combine it with our bulge-and-disc galaxy.
Both have `redshift=0.5` in this example.

Their light overlaps in the image. Here we are simply adding their light profiles; we are not calculating
gravitational lensing or simulating an interaction between the galaxies.
"""
extra_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.2, 0.3),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

galaxies = al.Galaxies(galaxies=[galaxy, extra_galaxy])

"""
The `Galaxies` object provides the same `image_2d_from()` method we used for an individual galaxy.

It calculates each galaxy’s image on the supplied grid, then adds their intensities at every position.
The resulting image contains the light from the original bulge and disc together with the second galaxy’s
Sérsic profile.
"""
image = galaxies.image_2d_from(grid=grid)

"""
Plot the combined image of our two galaxies. Their centres are close together, so their light overlaps strongly.

Compare this with the original bulge-and-disc galaxy image. The additional galaxy changes the brightness
distribution near the centre.
"""
aplt.plot_array(
    array=image,
    title="Combined Galaxy Image",
)

"""
To see the contribution from each galaxy, use `aplt.subplot_galaxy_images()`. Each panel shows one galaxy
from the collection, in the order we supplied them.

The first contains our bulge and disc; the second contains the additional galaxy’s single Sérsic profile.
Compare their positions and shapes. Each panel uses its own colour scale, so check the colour bars when
comparing brightness.
"""
aplt.subplot_galaxy_images(galaxies=galaxies, grid=grid)

"""
Display the combined image with logarithmic colour scaling to make the faint, extended light easier to see.

The galaxies’ light overlaps in the same pixels. Compare this view with the individual images to understand
how each contributes to the combined shape.
"""
aplt.plot_array(
    array=image,
    title="Combined Galaxy Image (Log Scale)",
    use_log10=True,
)

"""
__Units__

So far, we have described positions and sizes in arcseconds: angles measured on the sky. To express a
galaxy’s size in kiloparsecs, we need its distance.

A cosmological model relates the galaxy’s redshift to the distance used for this conversion. Here we use
`Planck15`, a cosmology based on measurements from the Planck satellite.

The code below calculates how many kiloparsecs correspond to one arcsecond at our galaxy’s redshift of 0.5.
"""
cosmology = al.cosmo.Planck15()

kpc_per_arcsec = cosmology.kpc_per_arcsec_from(
    redshift=galaxy.redshift
)

print("Kiloparsecs per arcsecond:", kpc_per_arcsec)

"""
Multiply an angular size by `kpc_per_arcsec` to express it in kiloparsecs.

Our bulge has an effective radius of 1.0 arcsecond. The calculation below gives the physical size corresponding
to that radius at the galaxy’s redshift, using the chosen cosmology.

This is the same circularised effective radius expressed in different units. The conversion does not change
the galaxy image or the profile parameters stored in `galaxy`.
"""
effective_radius_kpc = (
    galaxy.bulge.effective_radius * kpc_per_arcsec
)

print("Bulge effective radius (arcseconds):", galaxy.bulge.effective_radius)
print("Bulge effective radius (kiloparsecs):", effective_radius_kpc)

"""
__Wrap Up__

You have built galaxy images from light profiles and explored how their appearance depends on position,
orientation, size and brightness.

The main ideas are:

- **Grids:** `(y, x)` coordinates specify where we evaluate the light.
- **Geometry:** shifting and rotating coordinates lets us describe a profile relative to its own centre and orientation.
- **Light profiles:** a Sérsic profile assigns an intensity to each position.
- **Galaxies:** light profiles combine into a `Galaxy`, and individual galaxies combine into `Galaxies`.
- **Visualisation:** images show spatial structure, while radial plots let us compare intensity as a function of distance.
- **Units:** a redshift and a cosmological model let us convert angular sizes into physical sizes.

**🤖 Ask AI:** In this tutorial, why do we rotate coordinates clockwise to evaluate a galaxy whose major axis
is oriented anticlockwise?

__Advanced Topics__

The notes below give some additional context on brightness units and numerical accuracy. They are optional
on a first reading.

__Other Unit Conversion__

We used redshift and a cosmological model to convert angular sizes into physical sizes. Converting image
brightness into calibrated astronomical units is a different task.

Astronomers often express brightness using magnitudes, including the AB magnitude system. To calculate these
from an observed image, we need its flux calibration or an appropriate magnitude zero point. Redshift alone
is not enough, and the arbitrary intensity units used in this tutorial do not define an AB magnitude.

The workspace’s [flux guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/guides/units/flux.py)
gives examples of brightness conversions, while its
[cosmology guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/guides/units/cosmology.py)
covers cosmological distance conversions.

__Over Sampling__

A light profile can vary substantially across a pixel, especially near a bright galaxy centre. Evaluating
it only at the pixel centre may not represent the light across the whole pixel accurately.

Over-sampling evaluates the profile at several positions within each pixel and combines those values.
This improves the numerical estimate without changing the image’s pixel size.

The library supports both uniform over-sampling and adaptive schemes that use finer sampling where it is
needed. Our manual calculation sampled pixel centres, whereas the library’s image calculation uses the
grid’s over-sampling settings.

Accuracy still depends on the profile and sampling choices. A useful check is to increase the sampling
and see whether the calculated image changes appreciably.

The workspace’s [over-sampling guide](https://github.com/PyAutoLabs/autolens_workspace/blob/main/scripts/guides/advanced/over_sampling.py)
explains these settings and how to check their accuracy.
"""
