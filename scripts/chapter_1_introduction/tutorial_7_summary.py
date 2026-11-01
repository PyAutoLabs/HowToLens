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

Furthermore, using the `MatPLot2D` and `lines=`/`positions=` overlays objects we can visualize any aspect we're interested 
in and fully customize the figure. 

Before beginning chapter 2 of **HowToLens**, you should checkout the package `autolens_workspace/plot`. This provides a 
full API reference of every plotting option in **PyAutoLens**, allowing you to create your own fully customized 
figures of strong lenses with minimal effort!
"""

tangential_critical_curve_list = al.LensCalc.from_tracer(
    tracer=tracer
).tangential_critical_curve_list_from(grid=grid)
radial_critical_curve_list = al.LensCalc.from_tracer(
    tracer=tracer
).radial_critical_curve_list_from(grid=grid)


aplt.plot_array(
    array=tracer.image_2d_from(grid=grid),
    title="Tracer Image with Critical Curves",
    lines=tangential_critical_curve_list,
)

"""
And, we're done, not just with the tutorial, but the chapter!

__Code Design__

To end, I want to quickly talk about the **PyAutoLens** code-design and structure, which was really the main topic of
this tutorial.

Throughout this chapter, we never talk about anything like it was code. We didn`t refer to  'variables', 'parameters`' 
'functions' or 'dictionaries', did we? Instead, we talked about 'galaxies', 'planes' a 'Tracer', etc. We discussed 
the objects that we, as scientists, think about when we consider a strong lens system.

Software that abstracts the underlying code in this way follows an `object-oriented design`, and it is our hope 
with **PyAutoLens** that we've made its interface (often called the API for short) very intuitive, whether you were
previous familiar with gravitational lensing or a complete newcomer!

__Source Code__

If you do enjoy code, variables, functions, and parameters, you may want to dig deeper into the **PyAutoLens** source 
code at some point in the future. Firstly, you should note that all of the code we discuss throughout the **HowToLens** 
lectures is not contained in just one project (e.g. the **PyAutoLens** GitHub repository) but in fact four repositories:

**PyAutoFit** - Everything required for lens modeling (the topic of chapter 2): https://github.com/PyAutoLabs/PyAutoFit

**PyAutoArray** - Handles all data structures and Astronomy dataset objects: https://github.com/PyAutoLabs/PyAutoArray

**PyAutoGalaxy** - Contains the light profiles, mass profiles and galaxies: https://github.com/PyAutoLabs/PyAutoGalaxy

**PyAutoLens** - Everything strong lensing: https://github.com/PyAutoLabs/PyAutoLens

Instructions on how to build these projects from source are provided here:

https://pyautolens.readthedocs.io/en/latest/installation/source.html

We take a lot of pride in our source code, so I can promise you its well written, well documented and thoroughly 
tested (check out the `test` directory if you're curious how to test code well!).

__Wrap Up__

You`ve learn a lot in this chapter, but what you have not learnt is how to 'model' a real strong gravitational lens.

In the real world, we have no idea what the 'correct' combination of light profiles, mass profiles and galaxies are 
that will give a good fit to a lens. Lens modeling is the process of finding the lens model which provides a good fit 
and it is the topic of chapter 2 of **HowToLens**.

Finally, if you enjoyed doing the **HowToLens** tutorials please git us a star on the **PyAutoLens** GitHub
repository: 

 https://github.com/PyAutoLabs/PyAutoLens

Even the smallest bit of exposure via a GitHub star can help our project grow!
"""
