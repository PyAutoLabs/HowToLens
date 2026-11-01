"""
Tutorial 7: Summary
===================

In this chapter, we have built up the steps needed to describe a gravitational lens and compare its predicted image with an observation:

1. A `Grid2D` provides the $(y,x)$ coordinates where we evaluate light and mass profiles.
2. Light profiles describe a galaxy's surface brightness, while mass profiles describe its gravitational lensing effects.
3. A `Galaxy` groups these profiles together at a specified redshift.
4. A `Tracer` groups galaxies into redshift planes and traces light between them.
5. The redshifts and cosmology connect angular quantities with physical distances.
6. A simulated imaging observation combines the predicted image with a telescope's PSF, exposure time and noise.
7. Comparing a predicted image with observed data gives residuals, a chi-squared and a log likelihood.
8. Putting this calculation into a likelihood function lets us evaluate different choices of model parameters.

This final tutorial brings those ideas together. We will revisit how profiles, galaxies and planes fit inside a tracer,
and use that structure to plot individual components of a lensed source.

__Contents__

- **Start:** Build a lens and a source with two light profiles.
- **Object Composition:** Access the galaxies and profiles inside a tracer.
- **Visualization:** Add critical curves to the lensed image.
- **Code Design:** Connect the structure of the code with the physical system.
- **Source Code:** Find the projects that provide these tools.
- **Wrap Up:** Prepare for automated lens modelling in Chapter 2.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import autolens as al
import autolens.plot as aplt

"""
__Start__

We begin with a lens galaxy containing an elliptical mass profile and a source galaxy containing two light profiles,
named `bulge` and `disk`.

The source profiles share a centre and ellipticity, but have different Sérsic indices. This gives them different radial
brightness distributions. We omit the lens galaxy's light, so the image produced by this tracer contains only the lensed
source.

These are specified profiles and parameter values, just as in our earlier examples.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
    ),
    disk=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Object Composition__

The tracer retains the galaxies and profiles we used to construct it. We can access them by stepping through this
structure.

`tracer.planes[0]` contains the galaxies at the lens redshift, while `tracer.planes[1]` contains those at the source
redshift. Each plane contains one galaxy in this example, so `[0]` selects that galaxy.

We can then use the component names we supplied, such as `.mass`, `.bulge` and `.disk`, to access a particular profile.
The printed output below follows this sequence from the complete tracer down to its individual profiles.
"""
print("Tracer:")
print(tracer)

print("Lens plane:")
print(tracer.planes[0])

print("Source plane:")
print(tracer.planes[1])

print("Lens galaxy:")
print(tracer.planes[0][0])

print("Source galaxy:")
print(tracer.planes[1][0])

print("Lens mass profile:")
print(tracer.planes[0][0].mass)

print("Source bulge:")
print(tracer.planes[1][0].bulge)

print("Source disk:")
print(tracer.planes[1][0].disk)

"""
First, plot the complete lensed image. The tracer evaluates both source light profiles at the ray-traced coordinates
and adds their contributions.

We also keep the source-plane coordinates for the next plots. Each coordinate tells us where a ray from an image-plane
pixel reaches the source plane.
"""
aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Lensed Source: Bulge + Disk",
)

source_plane_grid = tracer.traced_grid_2d_list_from(grid=grid)[1]

"""
We can now evaluate each source profile separately at those same ray-traced coordinates.

These plots show the individual **lensed** contributions of the bulge and disk. Although we evaluate their brightness
at source-plane coordinates, the resulting values remain arranged according to the original image-plane pixels.

Compare the distribution of light in the two images. Their different Sérsic indices change how concentrated the light
is and how it extends away from the centre. Adding these two images gives the complete lensed source image above.

This is a decomposition of our specified source description. We have not inferred separate bulge and disk components
from observed data.
"""
aplt.plot_array(
    array=tracer.planes[1][0].bulge.image_2d_from(grid=source_plane_grid),
    title="Lensed Bulge",
)

aplt.plot_array(
    array=tracer.planes[1][0].disk.image_2d_from(grid=source_plane_grid),
    title="Lensed Disk",
)

"""
**🤖 Ask AI:** Why does evaluating a source light profile on ray-traced source-plane coordinates produce a lensed image,
whereas evaluating it on a uniform source-plane grid produces an unlensed image?
"""

"""
__Visualization__

We can combine the lensed image with other quantities calculated from the same tracer.

Here, we overlay the tangential critical curves introduced earlier in the chapter. Compare their position with the
bright arcs in the lensed image. The curves depend on the lensing mass distribution and redshifts, while the appearance
of the arcs also depends on the source's position and light profiles.

The `lines=` argument adds these curves to the image. The plotting examples in `autolens_workspace/plot` show how to
customise figures and add other overlays as you need them.
"""
tangential_critical_curve_list = al.LensCalc.from_tracer(
    tracer=tracer
).tangential_critical_curve_list_from(grid=grid)

aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Lensed Source\nTangential Critical Curves",
    lines=tangential_critical_curve_list,
)

"""
__Code Design__

The structure we have just explored reflects the physical description of a lens system: profiles belong to galaxies,
galaxies occupy redshift planes, and a tracer calculates how light travels between those planes.

This is an example of *object-oriented design*. Each object brings together information about something, such as a
galaxy, and operations we can perform with it, such as calculating its image.

We have used Python variables, parameters and functions throughout this chapter, but their names and organisation let
us express those operations in familiar scientific terms. For example, `source.bulge` identifies a particular light
component, while `tracer.image_2d_from(grid=grid)` calculates the image of the complete system.

As your analyses become more complex, this same structure lets you add components and inspect their individual
contributions.
"""

"""
__Source Code__

The tools used in **HowToLens** are provided by four related projects:

- **[PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit)** provides tools for defining models, searching their parameter spaces and analysing inference results. We will use these ideas in Chapter 2.
- **[PyAutoArray](https://github.com/PyAutoLabs/PyAutoArray)** provides arrays, grids, masks and tools for handling astronomical datasets.
- **[PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy)** provides light profiles, mass profiles and galaxies.
- **[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens)** brings these components together for gravitational lensing, including ray tracing and comparisons with lensing observations.

You access much of this functionality through `import autolens as al`, so you do not need to work with each repository
separately to follow the tutorials.

If you want to understand how a calculation is implemented, you can explore the source code and its accompanying tests.
Start with a familiar operation, such as calculating a light-profile image, and follow how its inputs become an output.

[Installation from source](https://pyautolens.readthedocs.io/en/latest/installation/source.html) is described in the
documentation. This is useful if you want to modify or contribute to the libraries; it is optional for continuing
**HowToLens**.
"""

"""
__Wrap Up__

You have now followed the path from a mathematical description of a lens system to a predicted observation and a
likelihood function that measures its agreement with data.

In our simulations, we chose the parameters that generated the observation. For a real lens, those parameters are
unknown. We must explore which choices describe the data well and what uncertainty remains.

In Tutorial 6, you began this process by changing parameters by hand. Chapter 2 introduces automated lens modelling:
a search repeatedly evaluates the likelihood function for different parameter choices, helping us infer the properties
of the lens and source.

The profiles, galaxies, tracers and dataset calculations you have learnt here remain the foundation of that process.

If you have found **HowToLens** useful, you can support the project by starring the
[PyAutoLens GitHub repository](https://github.com/PyAutoLabs/PyAutoLens).
"""
