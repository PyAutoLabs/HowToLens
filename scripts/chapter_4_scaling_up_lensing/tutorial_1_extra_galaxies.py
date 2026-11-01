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
The model can also be visualized as a figure, showing how its components and parameters are organized.

Notice that the extra galaxy's light and mass centres are fixed to its measured position. This is the workflow
we will reuse as we introduce larger populations of galaxies: measure their positions first, then keep those
positions fixed while fitting their other properties.

Here, spherical profiles and a linearly solved light intensity mean the extra galaxy adds only two non-linear
parameters. The figure shows this structure, while `model.info` gives the numerical values and priors.
"""
af.ModelPlotter(model).figure()

"""
__Extra Galaxies Fit__

We fit the model using the same search settings as before.

The extra galaxy adds only two non-linear parameters, keeping the increase in search dimensionality small.
Nevertheless, fitting time also depends on degeneracies between components and the cost of each likelihood
evaluation.

This fit uses the original, unscaled data, so the extra galaxy's light now contributes to constraining the model
alongside the main lens and source.
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
    "The non-linear search has begun. Check the output/howtolens/chapter_4/tutorial_1_extra_galaxies_model"
    " folder for results and visualizations as the search progresses."
    " This cell will finish when the search completes."
)

result_extra_galaxies = search.fit(model=model, analysis=analysis)

print("The search has finished — you may now continue the notebook.")

"""
Inspect the maximum log likelihood fit below. The extra galaxy's emission is now represented by its own light
profile, and its mass contributes to the ray-tracing of the background source.

Compare the residuals around the extra galaxy and the lensed arcs with the earlier fit. Also compare the inferred
main-lens parameters: accounting for the extra galaxy's mass may change them even if both fits appear visually
satisfactory.

The two fits use different noise maps and different sets of informative pixels. Their raw likelihood or Bayesian
evidence values should therefore not be interpreted as a straightforward model comparison.
"""
aplt.subplot_fit_imaging(fit=result_extra_galaxies.max_log_likelihood_fit)

"""
The result's `info` summarizes the inferred parameters and their uncertainties, including the extra galaxy's
`einstein_radius`.

Inspect its uncertainty and whether it lies close to a prior boundary. Including a mass profile does not
guarantee that the data tightly constrain its normalization.

The inferred Einstein radius describes the lensing strength of the adopted `IsothermalSph` profile; it is not a
direct measurement of the galaxy's total mass.
"""
print(result_extra_galaxies.info)

"""
__Which Approach When__

Choosing how to handle an extra galaxy returns us to two separate questions: does its light affect the fit,
and do its gravitational deflections matter for the measurements we want to make?

**Consider the light first.** If its emission is outside the fitting region or negligible compared with the noise,
we may not need a light profile. If it contaminates pixels we would otherwise fit, we can noise-scale those pixels
or model the emission explicitly.

Noise scaling is useful when we can exclude the contaminated region without losing important source information.
If the extra galaxy overlaps the lensed arcs, explicitly modeling its light lets us retain those pixels, although
the accuracy of its light profile then matters.

**Consider the mass separately.** Its importance depends on its mass, redshift and position relative to the lensed
images, as well as the precision of the data and the scientific question. Neither faintness nor separation alone
guarantees that its deflections are negligible.

To investigate its effect, compare models with and without the extra mass profile while keeping the data
treatment and other modeling choices consistent. Inspect changes in the residuals and inferred parameters
relative to their uncertainties. A small change is useful evidence of robustness within the models tested,
rather than proof that every possible perturbation is negligible.

The API supports each combination: model both light and mass, model only the light, or noise-scale the light while
retaining a mass profile.

__Wrap Up__

In this tutorial, we learned:

1. An extra galaxy's light and mass affect the analysis in different ways and should be considered separately.
2. Noise scaling, introduced in chapter 2, can exclude contaminating light from constraining the fit, but does not
   remove the galaxy's gravitational effect.
3. The `extra_galaxies` collection lets us include the light and mass profiles needed for each galaxy.
4. Measuring light centres first and fixing them in the model is a common practical workflow. It becomes
   increasingly useful as the number of galaxies grows.
5. The appropriate treatment depends on the data and scientific goal, and should be checked through fit
   diagnostics and comparisons between consistent modeling choices.

Here, one galaxy remained the dominant lens, with a smaller nearby galaxy adding a complication.

In the next tutorial, we consider systems where two or more galaxies contribute substantially to the lensing.
We will build models containing multiple main deflectors and explore the degeneracies between their mass
distributions.
"""
