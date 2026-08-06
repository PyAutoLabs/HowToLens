"""
Tutorial 1: Extra Galaxies
==========================

Welcome to chapter 4 of **HowToLens**, where we scale up lens modeling beyond a single lens galaxy.

In chapters 1 to 3, every strong lens we studied had the same simple anatomy: one lens galaxy, whose light and mass
we modeled, and one background source galaxy, whose lensed emission we fitted. This is the cleanest possible lensing
configuration, and it was the right place to learn the fundamentals of ray-tracing, non-linear searches and
pixelized source reconstructions.

Real strong lenses are rarely this tidy. Galaxies live in crowded fields: a lens galaxy may have a faint companion a
few arc-seconds away, sit in a small group of comparably massive galaxies, or be embedded in a cluster containing
hundreds of members and a vast dark matter halo. The science of strong lensing scales up through this hierarchy, and
so must our modeling:

- **Extra galaxies (this tutorial)**: a single dominant lens galaxy, with one or more nearby interloper galaxies
  whose light and mass may contaminate the analysis.

- **Multi-galaxy lenses (next tutorial)**: systems where two or more galaxies are co-equal lenses, and no single
  galaxy can be called "the" lens.

- **Scaling relations**: modeling many galaxies at once by tying their properties to their luminosity, so that
  adding galaxies does not add free parameters.

- **Group and cluster scales**: lenses whose deflectors are entire groups or clusters of galaxies, including a
  shared dark matter halo.

- **Weak lensing**: the statistical distortion of many background galaxies by foreground mass, extending lensing
  beyond the strong regime entirely.

Much of the methodology in this chapter is not specific to lensing. Fitting the light of extra galaxies, modeling
blended multi-galaxy fields and composing models via scaling relations are equally important for non-lensing galaxy
studies, and **PyAutoGalaxy** provides the same API for these tasks applied to galaxy morphology (without the
ray-tracing).

In this first tutorial, we take the first step up in scale: a galaxy-scale lens with one extra galaxy nearby. We
will learn how to decide whether the extra galaxy matters, and the two approaches to dealing with it when it does:
removing its light from the data, or including it in the lens model.

__Contents__

- **Initial Setup:** Load the imaging dataset of a lens with an extra galaxy and inspect the interloper.
- **Light Versus Mass:** The core decision: do the extra galaxy's light and / or mass affect the analysis?
- **Mask:** Define a circular mask large enough to include the extra galaxy's emission.
- **Approach 1 Noise Scaling:** Remove the extra galaxy's light from the fit by scaling its data and noise values.
- **Noise Scaling Fit:** Fit a lens model to the noise-scaled data, without the extra galaxy in the model.
- **Approach 2 Extra Galaxies Model:** Include the extra galaxy's light and mass in the lens model explicitly.
- **Extra Galaxy Centres:** Why the extra galaxy's centre is fixed to its observed light centre.
- **Extra Galaxies Fit:** Fit the lens model which includes the extra galaxy.
- **Which Approach When:** Guidance on choosing between noise scaling and explicit modeling.
- **Wrap Up:** Summary of the script and next steps.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
__Initial Setup__

Lets load the `Imaging` dataset we'll fit in this tutorial. It is similar to the `lens_sersic` dataset used
throughout chapter 2, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `SersicCore`.

However, there is one addition: an extra galaxy, with its own light (an `ExponentialSph`) and its own
mass (an `IsothermalSph`), located a few arc-seconds from the lens galaxy.
"""
dataset_name = "lens_extra_galaxy"
dataset_path = Path("dataset") / "imaging" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/lens_extra_galaxy.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

"""
When we plot the dataset, the familiar sight of a lens galaxy surrounded by a lensed source's ring of emission is
joined by a blob of light in the upper-right of the image.

This is the extra galaxy. It is not part of the strong lens system we are studying: it is an interloper, a galaxy
that happens to lie close to the lens on the sky. Interlopers like this are extremely common in real imaging of
strong lenses, and every lens modeler has to decide what to do about them.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Light Versus Mass__

The decision of what to do about an extra galaxy hinges on one question, asked twice:

**Does its light affect the analysis?** The extra galaxy's emission may extend into the region of the image we fit.
If it does, and we fit a model containing only the lens and source, the model has no component that can produce
this emission. The fit will respond by distorting the lens light, source light or mass model to soak it up,
biasing every parameter we infer.

**Does its mass affect the analysis?** The extra galaxy's mass deflects light rays, just like the lens galaxy's
mass does. If the galaxy is close (in projection) to the lensed source's light, its deflections measurably perturb
the ray-tracing. A model without this mass will compensate by biasing the main lens galaxy's mass parameters, for
example its `einstein_radius` or `ell_comps`.

These two effects are dealt with separately, which gives us two approaches:

- **Approach 1 (remove its light)**: If only the light matters, we can remove the extra galaxy's emission from the
  fit entirely, without adding anything to the model. The mass is ignored.

- **Approach 2 (model it explicitly)**: If the mass matters (or the light blends too closely with the lensed source
  to cleanly remove), we include the extra galaxy in the lens model, with its own light and / or mass profiles.

We will now perform both, and at the end of the tutorial discuss when each is appropriate.

__Mask__

We first define the circular mask used to fit the data. In chapter 2 we typically used a 2.6" - 3.0" mask, which
tightly contained the lens and lensed source.

Here, we use a larger 4.0" mask, so that the region containing the extra galaxy is included in the fit. If we
simply shrank the mask to exclude the extra galaxy, we would also throw away pixels containing lensed source
emission, and the mask's hard edge could still cut through the extra galaxy's light.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=4.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Plotting the masked dataset confirms the extra galaxy's emission is inside the mask, and will therefore impact the
model-fit unless we do something about it.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Approach 1 Noise Scaling__

Our first approach removes the extra galaxy's light from the fit.

The most obvious way to do this would be to mask the extra galaxy's pixels, removing them from the fit entirely.
However, removing pixels changes the fit in subtle ways: their coordinates are no longer used in the ray-tracing,
and for certain models (e.g. the pixelized source reconstructions of chapter 3) removing interior pixels creates
discontinuities in the pixelization that introduce unexpected systematics.

Instead, we use **noise scaling**: the pixels stay in the fit, but their data values are set to zero and their
noise-map values are increased to very large values. A pixel with enormous noise contributes negligibly to the
likelihood, so the extra galaxy's light cannot influence the model, while the pixels themselves remain part of the
fit's geometry.

To do this we need a mask of the extra galaxy's region. For real data, you would create this yourself by
inspecting the image (the workspace's `data_preparation` package includes a GUI for drawing it); for this simulated
dataset the simulator script has already output a `mask_extra_galaxies.fits` circle covering the extra galaxy.

We reload the dataset first, because noise scaling must be applied before the circular mask.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path / "mask_extra_galaxies.fits",
    pixel_scales=0.1,
    invert=True,  # Note that we invert the mask here as `True` means a pixel is scaled.
)

dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=4.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Plotting the dataset shows the extra galaxy's emission has vanished: its data values are zero and the
signal-to-noise of its pixels is effectively zero, so the fit will simply ignore that region of the image.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Noise Scaling Fit__

We now fit this noise-scaled dataset with a lens model that does **not** include the extra galaxy. The model is the
same one we used in chapter 2's linear profiles tutorial: a linear `Sersic` lens bulge, an `Isothermal` plus
`ExternalShear` mass model and a linear `SersicCore` source.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=al.mp.ExternalShear
)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_4"),
    name="tutorial_1_extra_galaxies_noise_scaling",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/tutorial_1_extra_galaxies_noise_scaling"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_noise_scaling = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
Plotting the maximum log likelihood fit shows the lens and source are fitted well, with the noise-scaled region
contributing nothing to the residuals.

The extra galaxy's light has been dealt with. However, note what this approach did **not** do: the extra galaxy's
mass is completely absent from the model. Its deflection of the source's light rays is unaccounted for, and the
inferred lens mass model will have absorbed that perturbation as best it can. For an extra galaxy that is small
and / or far from the lensed emission this bias is negligible; for one that is massive and close, it is not.
"""
aplt.subplot_fit_imaging(fit=result_noise_scaling.max_log_likelihood_fit)

"""
__Approach 2 Extra Galaxies Model__

Our second approach includes the extra galaxy in the lens model, fitting its light (so we no longer need to remove
it from the data) and its mass (so its deflections are included in the ray-tracing).

We reload the dataset and apply the 4.0" circular mask, but this time we do **not** apply noise scaling, because
the extra galaxy's emission is now something the model itself will fit.
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=4.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Extra Galaxy Centres__

To include the extra galaxy in the model, we input the centre of its light, as observed in the image.

In principle, we could add the extra galaxy to the model with a completely free centre, and let the non-linear
search figure out where it is. In practice this is a bad idea: the extra parameters make parameter space more
complex, and fits commonly go wrong in characteristic ways. For example, the extra galaxy's light profile may
wander off and fit one of the lensed source's multiple images instead of the interloper, or its mass profile may
recentre itself onto the main lens galaxy and act as part of its mass distribution.

Fixing each extra galaxy's light and mass centres to its observed light centre removes these failure modes and
keeps the model as simple as possible. The observed centre is an excellent estimate of the true centre, because
(unlike the lensed source) the extra galaxy's light is not distorted by lensing.

For real data you would measure these centres from the image (the workspace's `data_preparation` package shows
how, including a GUI for marking them); for this simulated dataset the simulator has output them to a .json file,
which we load below.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(extra_galaxies_centres)

"""
__Extra Galaxies Model Composition__

We compose the main lens and source model exactly as before.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=al.mp.ExternalShear
)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

"""
We now compose the model of the extra galaxy, using the dedicated `extra_galaxies` modeling API.

For each extra galaxy centre (there is only one here, but the loop below scales to any number) we create a `Galaxy`
model with:

 - A linear `ExponentialSph` light profile, with its `centre` fixed to the observed centre [1 free
   parameter: `effective_radius`, as the `intensity` is solved for by the linear inversion].

 - An `IsothermalSph` mass profile, with its `centre` fixed to the observed centre [1 free
   parameter: `einstein_radius`].

Extra galaxy mass profiles can wander to unphysically high `einstein_radius` values, degrading the fit, so we set
a `UniformPrior` with an upper limit of 0.5" to prevent this. The extra galaxy is much less massive than the main
lens (whose `einstein_radius` is around 1.6"), so this prior comfortably contains all plausible solutions.

The extra galaxies are grouped into their own `af.Collection`, which is passed to the overall model via its
`extra_galaxies` input, alongside the `galaxies` collection containing the lens and source. This is the same
API used throughout the `autolens_workspace` for extra galaxies, and it is how **PyAutoLens** knows to include
these galaxies in the ray-tracing without treating them as the main lens or source.
"""
# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in extra_galaxies_centres:

    # Extra Galaxy Light

    light = af.Model(al.lp_linear.ExponentialSph)
    light.centre = extra_galaxy_centre

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.5)

    # Extra Galaxy

    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, light=light, mass=mass)

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

# Overall Lens Model:

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), extra_galaxies=extra_galaxies
)

"""
The `info` attribute confirms the model includes the extra galaxy, with its fixed centres and its
free `effective_radius` and `einstein_radius` parameters.
"""
print(model.info)

"""
__Extra Galaxies Fit__

We fit this model with the same search set up as before. The model has only two more free parameters than the
noise-scaling fit, thanks to the fixed centres and linear light profile, so the fit remains fast.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_4"),
    name="tutorial_1_extra_galaxies_model",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,  # GPU batching and VRAM use explained in chapter 2 tutorial 2.
    iterations_per_quick_update=2500,  # Outputs Notebook visualization of max likelihood model every N iterations
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the workspace/output/howtolens/chapter_4/tutorial_1_extra_galaxies_model"
    " folder for live output of the results, images and model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_extra_galaxies = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

"""
Plotting the maximum log likelihood fit shows the extra galaxy's emission is now fitted by its own light profile,
and its mass has contributed deflections to the ray-tracing of the source.
"""
aplt.subplot_fit_imaging(fit=result_extra_galaxies.max_log_likelihood_fit)

"""
The result's `info` shows the inferred `einstein_radius` of the extra galaxy, quantifying how much lensing power
the interloper contributes to the system.
"""
print(result_extra_galaxies.info)

"""
__Which Approach When__

We have seen the two extremes of dealing with an extra galaxy. Choosing between them comes back to the
light-versus-mass question at the start of this tutorial, which in practice is settled by three properties of the
interloper:

- **Distance from the lensed emission**: An extra galaxy whose light is well separated from the lensed source (as
  in this tutorial) can be cleanly noise-scaled away. If its light blends into the source's arcs, noise scaling
  would also delete source emission we need, and the light must be modeled instead.

- **Brightness**: A faint interloper whose emission barely rises above the noise can often be ignored entirely, or
  noise-scaled with no consequence. A bright one must be removed or modeled, or it will bias the fit.

- **Mass and proximity to the lens**: The mass matters when the extra galaxy is massive enough, and close enough
  (in projection) to the lensed source's light, for its deflections to measurably perturb the ray-tracing. A rough
  rule of thumb is to compare its expected Einstein radius to the astrometric precision of the data: a small galaxy
  several arc-seconds from the arcs can have its mass safely ignored, whereas one abutting the Einstein ring
  cannot. When in doubt, fit both approaches and compare the inferred lens models: if the main lens's mass
  parameters shift appreciably, the extra galaxy's mass matters.

Intermediate options also exist and are fully supported by the API: you can noise-scale the light but still include
the mass profile in the model, or model the light but omit the mass. The `extra_galaxies` collection simply
contains whatever profiles you give it.

__Wrap Up__

In this tutorial, we took the first step up in scale from the single lens galaxy of chapters 1 to 3, and learnt:

1. Real lenses live in crowded fields, and interloping extra galaxies are the first complication real data throws
   at us.

2. Whether an extra galaxy matters hinges on two separate questions: does its **light** contaminate the region of
   the image we fit, and does its **mass** perturb the ray-tracing of the source?

3. Noise scaling removes an extra galaxy's light from the fit without adding model complexity, by zeroing its data
   and inflating its noise, but leaves its mass unaccounted for.

4. The `extra_galaxies` modeling API includes extra galaxies in the model with their own light and mass profiles,
   with their centres fixed to the observed light centres to keep parameter space simple and well behaved.

5. Which approach is appropriate depends on the interloper's distance from the lensed emission, its brightness,
   and its mass's proximity to the lens.

Throughout, the extra galaxy was a nuisance: something to remove or account for, so that our analysis of the
main lens remained accurate. In the next tutorial we meet systems where that framing breaks down entirely, because
a second galaxy is not a nuisance but a co-equal lens, with light and mass comparable to the first. There, no
single galaxy is "the" lens, and the model must treat them all on an equal footing
"""
