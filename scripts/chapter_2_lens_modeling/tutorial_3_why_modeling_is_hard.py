"""
Tutorial 3: Why Modeling Is Hard
================================

We now have the tools to build a lens model, run a search and inspect its results. However, a search finishing does not
necessarily mean that it has found a satisfactory fit.

As we add model components and free parameters, more combinations can produce similar images. Parameter space becomes
harder to explore, and a search can miss solutions that explain the data better.

In this tutorial, we examine these difficulties using the lens-light, mass, shear and source-light model from
*Practicalities*. We then try three approaches: adjusting justified priors, simplifying the model and searching more
thoroughly.

Each approach has benefits and costs. The aim is to understand which change addresses the problem we see, rather than
repeatedly changing settings until a fit looks convincing.

__Contents__

- **Overview:** Introduce the experiment and the results to inspect.
- **Dataset and Mask:** Load the imaging data and select the region to fit.
- **Model and Analysis:** Reconstruct the reference model and its likelihood calculation.
- **A Less Thorough Search:** Explore the model with a deliberately small live-point population.
- **Inspecting the Result:** Inspect the result summary, fitted image and residuals.
- **Comparing Searches:** Compare the small search with the reference fit.
- **Why Modeling Is Hard:** Understand why different parameter combinations can explain similar images.
- **Three Ways Forward:** Introduce prior tuning, simpler models and more thorough searches.
- **Prior Tuning:** Use the image and justified assumptions to narrow the priors.
- **Profile Centres:** Use the observed light centre to motivate centre priors.
- **Lens-Light Shape:** Estimate the lens-light ellipticity from the image.
- **Mass Shape:** Distinguish an assumed mass shape from the directly observed light.
- **Lens-Light Size:** Use the visible extent of the light to guide its size prior.
- **Light Concentration:** Motivate a prior for the Sersic index.
- **Einstein Radius:** Use the arc separation to guide the Einstein-radius prior.
- **Source and Shear Priors:** Explain why the source and shear priors remain unchanged.
- **Inspecting the Tuned Model:** Check the revised priors before fitting.
- **Prior-Tuned Fit:** Fit the model with the revised priors.
- **Assessing the Improvement:** Compare the tuned fit with the initial result.
- **Prior Tuning: Benefits and Limitations:** Assess the benefits and risks of narrowing priors.
- **Reducing Complexity:** Consider which model parameters could be linked.
- **Linking Parameters:** Link light and mass centres and elliptical components.
- **Inspecting the Simpler Model:** Check the reduced number of independent parameters.
- **Fitting the Simpler Model:** Fit the reduced-complexity model.
- **Reducing Complexity: Benefits and Limitations:** Balance easier exploration against restrictive assumptions.
- **Search More Thoroughly:** Return to the original model and give the search more resources.
- **Original Model:** Restore independent model components and their default priors.
- **More Thorough Search:** Run the original model with more live points.
- **Comparing the Approaches:** Compare the fitted images and residuals across experiments.
- **Benefits and Weaknesses:** Compare the practical trade-offs of the three approaches.
- **Run Times:** Understand the costs that determine how long a fit takes.
- **Model Mismatch:** Distinguish insufficient exploration from an inadequate model.
- **Wrap Up:** Review the lesson and its connection to the next topic.


__Overview__

We will first run a less thorough search and inspect its result. It may find a satisfactory solution, or it may miss a
better-fitting region of parameter space. The outcome is not guaranteed because the search includes random sampling.

We will use this experiment to understand why fits can struggle and how to investigate them. We will also distinguish a
search that has missed a good solution from a model that cannot adequately describe the data.

__Dataset and Mask__

We use the same `lens_sersic` imaging dataset as in *Practicalities*, with its Sérsic lens light, elliptical isothermal
mass, external shear and exponential source light.

We also retain the circular mask with a radius of 2.5 arcseconds. Keeping the dataset and mask unchanged lets us
compare searches using the same measurements and likelihood calculation.

Inspect the image again before fitting. Identify the central lens light and the lensed source emission: the model must
explain both.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

al.util.dataset.simulate_if_needed(
    dataset_path, simulator="scripts/simulator/lens_sersic.py"
)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

mask_radius = 2.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model and Analysis__

We reconstruct the same model used in *Practicalities*, with the same profiles, fixed redshifts and default priors.

Print `model.info` and inspect its diagram to remind yourself which parameters are free. The lens light, lens mass,
shear and source light all contribute to the image, so changes in one component can sometimes be partly compensated by
changes in another.

We again use `AnalysisImaging` to predict the image and evaluate its likelihood. At this stage, neither the model nor
the analysis has changed. We will change only how thoroughly Nautilus explores parameter space.
"""

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=al.lp.Sersic,
            mass=al.mp.Isothermal,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge=al.lp.ExponentialCore,
        ),
    ),
    fields=field,
)

print(model.info)
af.ModelPlotter(model).figure()

analysis = al.AnalysisImaging(dataset=dataset)

"""
__A Less Thorough Search__

We deliberately reduce `n_live` from 200 to just 21. This gives Nautilus very few live points with which to explore
this complex model, making it more likely to miss a good solution.

Nautilus requires at least one more live point than the number of non-linear parameters. This model has 20, so 21 is
the smallest valid setting; a value of 10 would raise an error rather than produce a fit.

Inspect the resulting fit and residuals. Your run may still succeed because the search includes random sampling; a
successful run does not make this a reliable setting.

We give this fit its own name so its output is kept separate from the *Practicalities* result.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_3_less_thorough",
    unique_tag=dataset_name,
    n_live=21,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result = search.fit(model=model, analysis=analysis)

"""
__Inspecting the Result__

Print the result summary and plot the fitted image.

First, look at whether the model reproduces the central lens light and the lensed source. Then examine the normalised
residuals for coherent structures that remain unexplained.

If the residuals contain a recognisable arc or a substantial mismatch around the lens galaxy, the fit needs further
investigation. Check the inferred parameters too, but do not judge the fit from plausible-looking values alone.

Your run may produce a satisfactory fit despite the low number of live points. In either case, the next step is to
compare it with the more thorough search.
"""

print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Comparing Searches__

Compare this fit with the 200-live-point fit from *Practicalities*. We use the same dataset, mask and physical model,
so their maximum log likelihoods can be compared directly.

A higher log likelihood means that the corresponding model explains these measurements better under the same noise
assumptions. Inspect the residuals alongside the numbers to see where the predictions differ.

If the more thorough search finds a substantially higher likelihood, it demonstrates that the less thorough search
missed a better solution. It does not prove that the better result is the best possible solution.

If the fits look similar and reach similar likelihoods, this run has not demonstrated a failure. Retain that outcome
rather than assuming the low-live-point fit must be wrong.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_2_practicalities",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_reference = search.fit(model=model, analysis=analysis)

print("Less thorough maximum log likelihood:")
print(result.max_log_likelihood_fit.log_likelihood)
print("Practicalities maximum log likelihood:")
print(result_reference.max_log_likelihood_fit.log_likelihood)
aplt.subplot_fit_imaging(fit=result_reference.max_log_likelihood_fit)

"""
__Why Modeling Is Hard__

Our example deliberately makes the search less thorough by reducing the number of live points from 200 to 21. This is a somewhat contrived way to expose a difficulty within a short tutorial: your particular run may still find a good solution.

The underlying problem, however, is common in modeling real lenses. Different combinations of lens light, mass and source light can reproduce parts of an image, creating several promising regions of parameter space. A search may explore one region thoroughly while missing a better solution elsewhere.

A **local maximum** has a likelihood at least as high as nearby parameter values. There may still be another peak with a higher likelihood elsewhere. The **global maximum** is the highest likelihood available within the chosen model and parameter ranges.

In a real analysis, it can be difficult to establish whether the best solution we have found is a local or global maximum. We do not know the true parameters, and a plausible-looking fit does not prove that every important region has been explored.

Finding a higher-likelihood solution demonstrates that an earlier result was not the global maximum. Repeatedly recovering a similar solution increases our confidence, but does not prove that no better solution exists.

The approaches in this tutorial are therefore tools for investigation. By changing justified priors, simplifying selected parts of the model and exploring more thoroughly, we can investigate suspicious fits and understand what makes the modeling difficult.

Residuals also require interpretation: they can arise because the search missed a good solution, or because the model cannot describe the data adequately. We will return to this distinction under *Model Mismatch*.

__Three Ways Forward__

We will try three approaches to making the fit more reliable:

1. **Prior tuning:** use justified information to give more probability to plausible parameter values or exclude
inappropriate ranges. 2. **Reducing complexity:** fix or link selected parameters, giving the search fewer independent
quantities to explore. 3. **Searching more thoroughly:** increase the search effort while retaining the original model
and priors.

The first two approaches change our modeling assumptions. The third changes the effort spent exploring them.

We will start each experiment from the original setup, rather than accumulating changes from the previous experiment.
This helps us understand what each approach contributes.

**🤖 Ask AI:** My lens fit has significant residuals. What observations would help me decide whether to adjust the
priors, simplify the model or search more thoroughly?

__Prior Tuning__

Our default priors cover a broad range of possible systems. For a particular image, we may have information that
justifies more specific choices.

For example, we can inspect where the lens light is centred, how extended it appears and the approximate scale of the
lensed arcs. These observations can help us identify sensible parameter ranges.

We must distinguish what the image tells us directly from additional assumptions. The lens-light centre is visible, but
assuming the mass is centred at exactly the same position is a physical assumption. Likewise, similar light and mass
shapes are not guaranteed.

In this exercise, we will make these choices explicit and examine their effects. We will not use the known simulation
parameters as though they were information available for an unknown real lens.

Priors chosen after inspecting the same image are practical modeling choices, not independent new measurements. Overly
restrictive choices can exclude good solutions and make the reported uncertainties misleadingly small.

__Profile Centres__

Look at the bright central lens galaxy. Its light is centred close to `(0.0, 0.0)` arcseconds, so we do not need to
give the search equal freedom to place it anywhere across the image.

We restrict each coordinate of the light-profile centre to between −0.05 and +0.05 arcseconds. This lets the search
refine the position around our visual estimate.

We also expect the lens mass to be centred near the light, so we apply the same limits to its centre. This is an
additional physical assumption: we see the light directly, whereas the mass distribution is inferred through its
lensing.

The two centres remain independent within these bounds. We have not forced them to coincide.
"""

bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)

mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)

"""
__Lens-Light Shape__

Now inspect the shape of the lens light. It appears fairly round, with a slight elongation running approximately 45°
counter-clockwise from the positive x-axis.

We can express this impression using an axis ratio and position angle. An axis ratio near one describes a nearly
circular profile; smaller values describe stronger flattening.

As an approximate starting guess, we use an axis ratio of 0.8 and an angle of 45°. These are estimates to guide the
search, not precise measurements from the image.

PyAutoLens uses elliptical components to describe the shape, so we convert our guess into those components and centre
Gaussian priors on them. Their widths allow the search to move away from our initial impression. This matters
especially for a nearly round galaxy, whose orientation can be difficult to judge.
"""

ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)

bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=ell_comps[0], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)
bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=ell_comps[1], sigma=0.1, lower_limit=-1.0, upper_limit=1.0
)

"""
__Mass Shape__

The lens light also gives us a possible starting point for the mass shape. We might expect the two to have broadly
similar orientations, but the image does not directly establish this.

We therefore centre the mass-shape priors on the same elliptical components, while making them wider than the
light-profile priors.

This guides the search towards mass distributions resembling the visible galaxy, but allows it to infer a different
flattening or orientation. The mass includes dark matter as well as stars, so agreement with the light should remain an
assumption we assess.
"""

mass.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
    mean=ell_comps[0], sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)
mass.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
    mean=ell_comps[1], sigma=0.3, lower_limit=-1.0, upper_limit=1.0
)

"""
__Lens-Light Size__

Look at how far the lens light extends from its centre. Its appearance suggests a characteristic scale of around an
arcsecond, rather than tens of arcseconds.

We use this impression to place a broad prior on the effective radius, centred on 1.0 arcsecond with a standard
deviation of 0.8 arcseconds.

The effective radius describes the profile’s half-light scale; it is not the visible outer edge of the galaxy. Faint
light extends beyond the bright region we can easily inspect, so our prior leaves substantial room for the search to
refine the size.

Truncating the prior at zero excludes negative radii.
"""

bulge.effective_radius = af.TruncatedGaussianPrior(
    mean=1.0, sigma=0.8, lower_limit=0.0, upper_limit=np.inf
)

"""
__Light Concentration__

The lens galaxy has a bright central concentration surrounded by fainter, extended light. This appearance motivates
trying a concentrated Sérsic profile.

A Sérsic index near one gives an exponential profile, often used for disks. Higher indices produce a more concentrated
centre and more extended wings; a value near four is a familiar starting description for elliptical-galaxy light.

We therefore centre the prior on 4.0, with a standard deviation of 1.0. We are using the galaxy’s appearance to choose
a plausible range, rather than claiming that its Sérsic index can be measured accurately by eye.

After fitting, we should inspect whether the model reproduces both the central light and the surrounding emission.
Persistent structure in the residuals may indicate that a single Sérsic profile is insufficient.
"""

bulge.sersic_index = af.TruncatedGaussianPrior(
    mean=4.0, sigma=1.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Einstein Radius__

Look at the distance between the lens centre and the bright lensed arcs. They lie roughly 1.6 arcseconds from the
centre, giving us an initial estimate of the lens’s Einstein radius.

For an elliptical lens with external shear, the arcs need not trace a circle of exactly this radius. Their distance
from the centre nevertheless gives a useful indication of the mass scale.

We centre a Gaussian prior on 1.6 arcseconds, with a standard deviation of 0.2 arcseconds. This directs the search
towards mass models capable of producing lensed images at approximately the observed separation, while allowing it to
refine our estimate.
"""

mass.einstein_radius = af.TruncatedGaussianPrior(
    mean=1.6, sigma=0.2, lower_limit=0.0, upper_limit=np.inf
)

"""
__Source and Shear Priors__

We could estimate the lens-light properties by looking directly at the foreground galaxy. The source is less
straightforward: we see its stretched and distorted images, rather than its original appearance.

Inferring its unlensed position, size and shape requires interpreting those images through a mass model. With
experience, the image configuration can guide these choices, but we will leave the source priors unchanged here.

We also retain the default shear priors. The appearance of the arcs reflects both the lens mass and external shear,
making their separate contributions difficult to estimate by eye.

Our changes therefore concentrate on quantities for which the image provides a useful initial impression, with the
assumptions about the mass stated explicitly.
"""

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp.ExponentialCore,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

"""
__Inspecting the Tuned Model__

Print the model summary and inspect its diagram before starting the search.

Connect each changed prior back to the image: the central lens position, its approximate orientation and flattening,
its light distribution, and the separation of the arcs from the centre.

The model still contains the same physical components and the same number of free parameters. We have changed the
probability assigned to their values, rather than removed quantities from the fit.

Check that the source and shear priors remain unchanged. This helps us keep track of which assumptions we are testing.
"""

print(model.info)

af.ModelPlotter(model).figure()

"""
__Prior-Tuned Fit__

We now repeat the fit using our revised priors.

We use 75 live points for the prior-tuned fit. Our aim is to see whether image-informed priors help us obtain a
satisfactory fit with less search effort than the 200-live-point setup in Practicalities.

This changes both the priors and the live-point count relative to the 21-live-point experiment, so any improvement
cannot be attributed to prior tuning alone.

Give this search a separate name so its output can be compared with the original broad-prior fit.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_3_prior_tuning",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_custom_priors = search.fit(model=model, analysis=analysis)

"""
__Assessing the Improvement__

Plot the new fit and compare it with the original 21-live-point result.

Look first at the lens centre and the lensed arcs. Has the model reproduced their brightness and structure more
closely? Have coherent patterns in the residuals become smaller?

Compare the maximum log likelihoods too. Because the dataset, mask and likelihood calculation are unchanged, a higher
value shows that the new search found a model that fits these measurements better.

If the fit improves, we have demonstrated how visual inspection can help guide a search. We have not proved that the
new priors are uniquely correct or that every relevant solution has been explored.

If it does not improve, inspect where the mismatch remains. The priors may be inappropriate, or the search may still
need more thorough exploration. Both outcomes are useful evidence for deciding what to try next.
"""

print(result_custom_priors.info)

aplt.subplot_fit_imaging(
    fit=result_custom_priors.max_log_likelihood_fit
)

"""
__Prior Tuning: Benefits and Limitations__

Looking at the image helped us identify plausible centres, shapes and scales. Encoding those impressions in the priors
can reduce the effort spent exploring models that bear little resemblance to the observed lens.

This can improve the reliability and speed of a fit. However, the benefit depends on the quality of our choices: a
prior that excludes a good solution prevents the search from finding it, however thoroughly it explores.

Some properties are also easier to estimate visually than others. The foreground light is directly visible, whereas the
source structure and mass distribution must be inferred through the lensing.

Finally, these choices are specific to this image. Inspecting and tuning every lens may be practical for a small study,
but becomes time-consuming for hundreds or thousands of systems.

We therefore need other ways to make complex models easier to fit.

__Reducing Complexity__

Return to the image and consider how much freedom we gave the lens model. Its light and mass can each have their own
centre, flattening and orientation.

That flexibility allows them to differ, but it also gives the search more combinations to explore. As a simplifying
experiment, we will require the mass and light to share their centre and elliptical components.

The visible lens light will then help constrain quantities that also determine the lensing. This reduces the number of
independent parameters, potentially making the model easier to fit.

Sharing these quantities is a physical assumption, not simply a faster way to calculate the same model. It excludes
lenses whose light and mass have different centres or shapes.

__Linking Parameters__

We create fresh light and mass models with their default priors. This removes the prior tuning from the previous
experiment, so we can examine parameter linking separately.

Setting `mass.centre = bulge.centre` makes the two profiles use the same centre parameters. Setting `mass.ell_comps =
bulge.ell_comps` does the same for their shapes.

These quantities are still fitted. The search chooses their values, but each choice now applies to both profiles.

This links four pairs of parameters: two centre coordinates and two elliptical components. It therefore removes four
independent dimensions from the search.

The Sérsic light profile and isothermal mass profile still have different radial forms. Sharing their centre and shape
does not make the full mass distribution proportional to the light.
"""

bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.Isothermal)

mass.centre = bulge.centre
mass.ell_comps = bulge.ell_comps

"""
__Inspecting the Simpler Model__

We assemble the lens, source and shear as before, then inspect the model summary and diagram.

Find the shared centre and elliptical components. They should appear as linked quantities rather than separate choices
for the light and mass.

The source and shear remain free to vary as in the original model. We have simplified one relationship within the lens
model, rather than fixing the whole lens or changing the observations.

Compare the parameter count with the original model and confirm that it has fallen by four. The diagram is particularly
useful here: an unintended link can be difficult to spot in a long list of parameters.
"""

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp.ExponentialCore,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

print(model.info)

af.ModelPlotter(model).figure()

"""
__Fitting the Simpler Model__

We use 75 live points for the reduced-complexity fit. This changes both the model and the live-point count relative to
the original broad-prior experiment, so any improvement cannot be attributed to parameter linking alone.

After the fit, inspect both the foreground light and the lensed arcs. A satisfactory model must explain both while
using the shared centre and shape.

In the current simulation, the lens light and mass were given matching centres and elliptical components. This makes it
a useful controlled example of simplifying a model when the assumption is compatible with the system.

For a real lens, we would not know that in advance. If linking improves a difficult fit, it provides a useful starting
model; it does not establish that the light and mass truly share the same geometry.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_3_reducing_complexity",
    unique_tag=dataset_name,
    n_live=75,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_linked = search.fit(model=model, analysis=analysis)

print(result_linked.info)

aplt.subplot_fit_imaging(fit=result_linked.max_log_likelihood_fit)

"""
__Reducing Complexity: Benefits and Limitations__

Linking the light and mass centres and shapes reduces the number of quantities the search must determine independently.
When these assumptions are appropriate, this can make a fit easier and faster.

Unlike priors chosen by inspecting one image, the same linking rule can be applied to many lenses. However, being easy
to apply does not make it appropriate for every system.

Imagine a lens whose light is elongated in one direction while its mass is oriented differently. Our linked model
cannot represent that difference. It may reproduce the foreground light but leave structured residuals around the arcs,
or distort other parameters to compensate.

Inspecting both parts of the image is therefore essential. Even apparently good residuals do not establish that the
linking assumption is harmless: some biases can be absorbed by other model components.

A simplified model can provide a useful first fit. For scientific measurements, we should also assess what changes when
its restrictions are relaxed.

__Search More Thoroughly__

Our first two approaches changed the priors or reduced the freedom of the model. We now return to the original model and ask Nautilus to explore it more thoroughly.

We increase the number of live points to 300. A larger population gives the search more opportunity to represent different promising regions of parameter space, although this generally requires more computation and does not guarantee that every important region will be found.

Increasing the number of live points is only one option. There are many different non-linear search methods, and they explore parameter space in different ways. Trying another method can make a substantial difference when a particular search struggles with a model.

The optional tutorial, `chapter_optional/tutorial_searches.py`, introduces alternative searches, including optimisation and MCMC. Their outputs serve different purposes: finding a high-likelihood solution and characterising parameter uncertainty are related but distinct tasks.

When investigating a difficult fit, compare repeated runs and, where appropriate, different search methods using the same model, priors and likelihood. Examine the fitted images and residuals alongside the likelihood values. Agreement provides useful evidence; disagreement tells us that further investigation is needed.

__Original Model__

We create fresh model components so that neither the tuned priors nor the parameter links carry over from the previous
experiments.

Check `model.info` and the diagram carefully. The light and mass should again have independent centres and elliptical
components, and the priors should match the original broad-prior model.

The dataset, mask and Analysis object remain unchanged. We are giving the search more opportunity to explore the same
physical model used at the start of this tutorial.
"""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp.Sersic,
    mass=al.mp.Isothermal,
)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp.ExponentialCore,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

print(model.info)

af.ModelPlotter(model).figure()

"""
__More Thorough Search__

Run Nautilus with 300 live points, keeping the batch size and output-update frequency unchanged.

Watch the saved fit images as the search progresses. Early models may explain only part of the image; allow the search
to finish before comparing its final result with the earlier fits.

More live points improve the opportunity for exploration, but do not guarantee that every important solution is found.
We still judge the result through the image comparison, inferred parameters and consistency with other searches.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_3_search_more_thoroughly",
    unique_tag=dataset_name,
    n_live=300,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result_thorough = search.fit(model=model, analysis=analysis)

print(result_thorough.info)

aplt.subplot_fit_imaging(fit=result_thorough.max_log_likelihood_fit)

"""
__Comparing the Approaches__

Compare the fitted images and normalised residuals from the different searches. Look at the central lens light and the
arcs separately: an improvement in one region can reveal a remaining mismatch elsewhere.

Then compare the maximum log likelihoods and inferred parameters. The dataset and likelihood calculation are unchanged,
so the likelihood values tell us which returned models fit these measurements better. They do not, by themselves,
establish which modeling assumptions are scientifically preferable.

Keep the changes made in each experiment in mind:

| Fit | Priors and model | Live points |
|---|---|---:|
| Deliberately under-resourced | Original priors and independent parameters | 21 |
| Prior tuning | Image-informed priors; parameters remain independent | 75 |
| Reduced complexity | Default priors; linked light and mass geometry | 75 |
| More thorough search | Original priors and independent parameters | 300 |

The prior-tuned and reduced-complexity fits change both the modeling choices and the number of live points relative to
the first experiment. Any improvement therefore cannot be attributed to prior tuning or parameter linking alone.

Agreement between satisfactory fits increases our confidence, while differences tell us what needs investigation. For
example, a shift in the inferred mass parameters after removing the light–mass links would make those assumptions worth
examining more closely.

**🤖 Ask AI:** My linked and independent light–mass models both produce small residuals, but infer different mass
parameters. What comparisons would help me assess whether the linking assumption affects my scientific conclusions?
"""

for label, fit_result in [
    ("Less thorough", result),
    ("Practicalities", result_reference),
    ("Prior tuning", result_custom_priors),
    ("Reduced complexity", result_linked),
    ("More thorough", result_thorough),
]:
    print(label, fit_result.max_log_likelihood_fit.log_likelihood)

"""
__Benefits and Weaknesses__

Each approach helps the search in a different way, and each has a cost.

| Approach | Benefits | Weaknesses |
|---|---|---|
| **Prior tuning** | Uses our interpretation of the image to guide exploration towards plausible solutions. Can make fitting faster and more reliable. | Incorrect priors can exclude good solutions or bias inference. Some parameters are difficult to estimate visually, and tailoring priors for every lens becomes time-consuming for large samples. |
| **Reducing complexity** | Fewer independent parameters can make the model easier to fit. The same simplifying rule can be applied across many lenses. | The assumptions may exclude real differences between components. A simpler model can produce biased parameters, even when its residuals look acceptable. |
| **Searching more thoroughly** | Easy to configure and retains the original model’s flexibility and priors. Does not require additional assumptions about this particular lens. | Requires more computation and may become impractical for demanding models. More searching cannot compensate for a model that cannot describe the data. |

The appropriate choice depends on what the image and previous fits tell us. A clear visual estimate may justify tuning
a prior; a useful physical approximation may justify simplifying the model. When the existing model and priors are
appropriate, more thorough exploration may be the best next step.

__Run Times__

Compare the time taken by the fits alongside their results. A quick search is useful only if its output is adequate for
the question we want to answer.

Prior tuning leaves the model’s image calculation largely unchanged, but can reduce the exploration needed to find good
solutions. Linking parameters reduces the number of independent quantities being searched, while retaining the same
light and mass profiles. Increasing the number of live points generally requires more evaluations.

The first two approaches therefore aim to make the search’s task easier; the third gives it more resources.

Our tuned-prior and reduced-complexity fits each used 75 live points, compared with 300 for the more thorough search.
Their elapsed times reflect both the modeling choices and these different search settings.

Compilation and output also contribute to the elapsed time, as discussed in *Practicalities*. Treat these runs as
examples of the trade-offs on your computer, rather than universal speed comparisons.

The following values come from each search's saved timer, so loading a completed result reports its recorded duration
rather than the time spent loading it. A resumed search includes the time since its original start, including any
interruption. Runs without recorded timing information are labelled accordingly.
"""

for label, fit_result in [
    ("Less thorough", result),
    ("Practicalities", result_reference),
    ("Prior tuning", result_custom_priors),
    ("Reduced complexity", result_linked),
    ("More thorough", result_thorough),
]:
    elapsed_seconds = fit_result.samples.time
    if elapsed_seconds is None:
        print(f"{label}: elapsed time not recorded")
    else:
        print(f"{label}: recorded elapsed time = {float(elapsed_seconds):.1f} seconds")

"""
__Model Mismatch__

So far, our simulated data have been described by the same profile families used in the unrestricted model. This gives
the search an opportunity to reproduce the image to the noise level.

Real galaxies are less accommodating. Imagine that the source contains a bright star-forming knot, but our model allows
only one smooth exponential light profile. The fitted arc might reproduce the overall brightness distribution while
leaving a compact feature in the residuals.

Searching longer cannot give that smooth profile a feature it does not possess. Even the highest-likelihood solution
available within this model may leave significant residuals.

Similar problems arise if the lens light contains a disk or other structure missing from its Sérsic description.
Incorrect assumptions about the PSF, noise or contaminating objects can also produce mismatches.

We must therefore distinguish **an inadequately explored model** from **an inadequate description of the
observations**. The residual image helps us investigate this distinction, although it rarely identifies the cause on
its own.

__Wrap Up__

More realistic lens models give us greater freedom to explain an image, but that freedom makes fitting harder. We have
explored three responses: use visual information to guide priors, simplify the model through explicit assumptions, or
search more thoroughly.

Each can help, but none removes the need to inspect the fit and question its assumptions.

These trade-offs are a major reason why **search chaining is central to practical lens modeling**. It helps us balance
model realism, reliable fitting and manageable run times, drawing on the strengths of the approaches above.

Later in this chapter, we will see how search chaining brings these ideas together for more complex lens models.

The next tutorial introduces another way to simplify the search while describing more detailed light distributions.
"""
