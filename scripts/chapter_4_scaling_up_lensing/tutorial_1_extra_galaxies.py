"""
Tutorial 1: Extra Galaxies
==========================

Welcome to chapter 4 of **HowToLens**, where we scale up lens modeling beyond a single dominant lens galaxy.

In chapters 2 and 3, we focused on modeling systems with one main lens galaxy and one background source galaxy.
These examples introduced non-linear searches, lens modeling and pixelized source reconstructions.

In chapter 1, we also saw that lensing occurs in more complex systems, including groups and clusters of galaxies.
We now explore how to build and fit models of these systems, starting with a nearby galaxy that complicates an
otherwise familiar galaxy-scale lens.

This chapter introduces:

- **Extra galaxies:** nearby galaxies whose light or mass may affect our analysis.
- **Multi-galaxy lenses:** systems where two or more galaxies contribute substantially to the lensing.
- **Scaling relations:** linking the mass properties of many galaxies to their luminosities, reducing the number
  of free parameters.
- **Group and cluster scales:** modeling member galaxies together with a shared dark matter halo.
- **Weak lensing:** using the small distortions of background galaxies to constrain foreground mass beyond the
  strongly lensed region.

Many of these modeling techniques also apply to studies of galaxy morphology without lensing. **PyAutoGalaxy**
provides the same tools for fitting galaxy light and composing models of multiple galaxies, without ray-tracing.

In this tutorial, we consider one extra galaxy near a dominant lens galaxy. We will distinguish the effects of its
light and mass, then compare excluding its light from the fit with explicitly including the galaxy in our model.

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

Let's load the `Imaging` dataset we will fit in this tutorial. It is similar to the `lens_sersic` dataset used in
chapter 2:

- The lens galaxy's light follows a `Sersic` profile.
- Its mass follows an `Isothermal` profile, with a separate `ExternalShear` field.
- The source galaxy's light follows a `SersicCore` profile.

This dataset also contains an extra galaxy a few arcseconds from the main lens. Its light follows an
`ExponentialSph` profile and its mass follows an `IsothermalSph` profile.

Both its emission and its gravitational deflections are therefore present in the simulated data.
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
Plotting the dataset shows the familiar lens galaxy and lensed source emission, together with an extra galaxy
nearby.

We call this an **extra galaxy** because the main lens remains the dominant deflector. This does not mean the extra
galaxy has no lensing effect: its mass also deflects the source's light.

Nearby galaxies are common in observations of strong lenses. Before fitting the data, we must decide whether their
light, their mass, or both need to be accounted for.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Light Versus Mass__

An extra galaxy raises two separate questions.

**Does its light affect the analysis?** If its emission falls within the region we fit, a model containing only the
main lens and source may try to reproduce that light using the wrong components. This can distort the inferred
lens light, source reconstruction or mass model.

**Does its mass affect the analysis?** The extra galaxy also deflects the source's light. Whether we need to model
these deflections depends on its mass, position and redshift, together with the precision of the data and the
measurements we want to make.

Removing the extra galaxy's light from the fit does not remove its gravitational effect. Likewise, including its
mass does not account for its emission.

We will compare two approaches:

- **Noise scaling:** use the method introduced in chapter 2 to prevent the extra galaxy's light from meaningfully
  constraining the fit. In this first example, we also omit its mass, allowing us to explore the limitation of
  that choice.
- **Explicit modeling:** include the extra galaxy's light and mass profiles, fitting its emission and accounting
  for its deflections.

These are not the only combinations. We can also noise-scale the light while retaining a mass profile, or model
the light while omitting the mass when its lensing effect is negligible.

__Mask__

We use a circular mask with a radius of 4.0", which includes the main lens, the lensed source emission and the
extra galaxy.

Keeping the extra galaxy inside the mask lets us compare noise scaling and explicit modeling using the same
overall fitting region.

For other datasets, a smaller mask may exclude an unwanted galaxy. This is appropriate only if it retains the
lensed emission we want to analyze and avoids significant contamination from the extra galaxy's light.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=4.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Plotting the masked dataset confirms that the extra galaxy's emission lies within the fitting region. We must
therefore account for its light, either by noise scaling the affected pixels or by modeling the emission explicitly.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Approach 1 Noise Scaling__

We first use the noise-scaling method introduced in chapter 2. Recall that this sets selected data values to zero
and increases their noise values, giving those pixels negligible weight when constraining the model while
retaining them in the dataset's geometry.

Here, we apply it to the region containing the extra galaxy's light. Any lensed source emission within that region
will also lose its constraining power, so we should avoid this approach where the extra galaxy and source are
strongly blended.

For this simulated dataset, the simulator provides a `mask_extra_galaxies.fits` file identifying the extra galaxy's
region. For real observations, we would define this region by inspecting the image.

We reload the original dataset, apply noise scaling, and then apply the same 4.0" circular mask.
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
The subplot shows that the selected region now has zero data values and effectively zero signal-to-noise. Its
pixels therefore contribute negligibly to constraining the fit.

This excludes the extra galaxy's light from the analysis, but its mass has still deflected the source light
recorded elsewhere in the image. Noise scaling does not undo those deflections.
"""
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Noise Scaling Fit__

We now fit the noise-scaled dataset using the familiar model from chapter 2: a linear `Sersic` lens light profile,
an `Isothermal` lens mass profile, a separate `ExternalShear` field and a linear `SersicCore` source light profile.

This model contains no extra galaxy. Its light has been handled by noise scaling, but its mass is deliberately
omitted.

The fit therefore illustrates an important distinction: excluding contaminating light does not necessarily make a
single-galaxy mass model adequate.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source), fields=field)

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
    "The non-linear search has begun. Check the "
    "output/howtolens/chapter_4/tutorial_1_extra_galaxies_noise_scaling "
    "folder for results and visualizations as the search progresses. "
    "This cell will finish when the search completes."
)

result_noise_scaling = search.fit(model=model, analysis=analysis)

print("The search has finished — you may now continue the notebook.")

"""
Inspect the maximum log likelihood fit below, paying particular attention to the residuals and normalized
residuals around the lensed source.

The noise-scaled pixels have negligible weight in the fit. Their raw residuals need not be zero, because the model
can still predict emission there.

Elsewhere, the model must reproduce source emission that was lensed by both the main lens and the extra galaxy,
even though it includes only the main lens mass and external shear. These components may absorb some of the
missing perturbation, potentially shifting the inferred parameters.

A visually satisfactory fit therefore does not establish that the extra galaxy's mass can safely be ignored.
Next, we include it explicitly so that we can compare the results.
"""
aplt.subplot_fit_imaging(fit=result_noise_scaling.max_log_likelihood_fit)

"""
__Approach 2 Extra Galaxies Model__

We now include the extra galaxy's light and mass in the model. Its light profile accounts for its emission, while
its mass profile contributes to the deflections of the background source.

We reload the original dataset and apply the same 4.0" circular mask. This time, we do not apply noise scaling,
because we want to fit the extra galaxy's light.
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

We first load the extra galaxy's observed light centre.

Allowing its light and mass centres to vary freely would introduce additional parameters and give the search more
opportunities to find unwanted solutions. For example, its light profile might fit part of a lensed arc, or its
mass profile might move towards the main lens and contribute to that galaxy's mass distribution.

Measuring galaxy light centres first and then fixing them in the model is a common workflow throughout
extra-galaxy modeling. It becomes especially useful for the more complex multi-galaxy, group and cluster systems
we encounter later in this chapter, where allowing every centre to vary would introduce many additional parameters.

Typically, we inspect the image and record each galaxy's light centre using a GUI, a source-detection tool or a
preliminary light-profile fit. We then use these measured positions as fixed centres for its light and mass
profiles in the lens model. The workspace's data-preparation tools support this workflow; here, we load the
positions supplied by the simulator.

Fixing centres is therefore a practical modeling choice used in many analyses, rather than just a simplification
for this tutorial. It keeps galaxies anchored to their observed positions and reduces the freedom for their
profiles to fit unrelated emission or move towards the main lens.

This still assumes that each mass profile is centred on the observed light. Where positional uncertainties or
possible light-mass offsets matter scientifically, we can relax that assumption using informative priors.

In this example, the extra galaxy is at the main lens's redshift, so its light is not lensed by the main lens.
This need not be true of every galaxy projected nearby on the sky.
"""
extra_galaxies_centres = al.Grid2DIrregular(
    al.from_json(file_path=dataset_path / "extra_galaxies_centres.json")
)

print(extra_galaxies_centres)

"""
__Extra Galaxies Model Composition__

We compose the main lens and source model exactly as before. Keeping these components unchanged makes it easier
to assess the effect of adding the extra galaxy.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)
field = af.Model(al.MassField, redshift=0.5, shear=al.mp.ExternalShear)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

"""
We now build the extra galaxy using the `extra_galaxies` modeling API.

For each observed centre, we create a galaxy containing:

- A linear `ExponentialSph` light profile. Its centre is fixed, leaving `effective_radius` as one free non-linear
  parameter. Its intensity is solved for during the linear inversion.
- An `IsothermalSph` mass profile. Its centre is also fixed, leaving `einstein_radius` as one free non-linear
  parameter.

The spherical profiles and fixed centres keep this model compact: the extra galaxy adds just two non-linear
parameters.

We assign its Einstein radius a uniform prior between 0.0" and 0.5", appropriate for the small companion in this
simulated example. For real data, this range should reflect what we know about the galaxy, and we should check
whether the inferred value approaches either prior boundary.

We place the extra galaxies in an `af.Collection` and pass it through the overall model's `extra_galaxies` input,
alongside the main lens and source.

Here, the extra galaxy has redshift 0.5, matching the main lens. A galaxy at a different redshift would require
its appropriate redshift in the model.
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
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
    fields=field,
)

"""
The `info` attribute shows the extra galaxy's fixed centres and the priors on its two free non-linear parameters:
`effective_radius` and `einstein_radius`.

Its light-profile intensity is determined by the linear inversion during each likelihood evaluation, rather than
sampled by the non-linear search.
"""
print(model.info)

"""
The same model can also be visualized as a figure, making its structure easier to understand at a glance.

The figure shows how the model is organized: which parameters belong to each component, and whether they are free,
fixed, shared, linked by an expression, solved during the fit, or not configured. `model.info` provides the
corresponding numerical details, including the prior assigned to each free parameter and the value of each fixed
parameter.

The extra galaxy is part of the model and part of the ray-tracing, but it is neither the lens nor the source.
Its centres are fixed to the values we measured from the data and its light is linear, so an entire extra galaxy
costs just two more free parameters than the same lens-and-source model without it.
"""
af.ModelPlotter(model).figure()

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
    "The non-linear search has begun running - checkout the output/howtolens/chapter_4/tutorial_1_extra_galaxies_model"
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
single galaxy is "the" lens, and the model must treat them all on an equal footing.
"""
