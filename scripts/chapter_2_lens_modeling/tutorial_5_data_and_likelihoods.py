"""
Tutorial 5: Data and Likelihoods
================================

Throughout this chapter, we have fitted imaging data using light and mass profiles. However, the modeling tools we have
learnt are not tied to this particular type of observation. We can use the same approach to fit interferometer
measurements, the positions of multiply imaged point sources, or the shapes of weakly lensed galaxies.

This flexibility comes from separating the different parts of a model fit. The **dataset** contains the measurements
and their uncertainties. The **model** describes the physical system and its free parameters. The **Analysis** object
uses that model to predict the measurements and evaluates a **likelihood**, which tells us how well those predictions
agree with the data. The **search** explores the model's parameters using the likelihood and priors.

The search does not need to know whether it is fitting image pixels, interferometer visibilities or image positions. It
supplies parameter values and receives a likelihood. By changing the dataset and the Analysis object, we can adapt the
tools we already know to a different observation, while retaining the model components appropriate to that system.

We will build up this idea through familiar imaging examples. First, we explain why our circular masks retain the lens
galaxy, the arcs and their faint surroundings. We then consider how positions constraints provide additional
information about the mass model, and how noise scaling changes the contribution of contaminating light to the fit.

These examples show how we can change the information entering an analysis without rebuilding the entire modeling
procedure. By the end of the tutorial, we will connect this separation to other types of lensing data and to analyses
that combine several datasets, each with its own likelihood, to constrain a shared physical model.

__Contents__

- **Dataset:** Return to the imaging example containing lens and source light.
- **Circular Masks:** Understand why we retain the central lens and surrounding arcs.
- **Beyond the Bright Arcs:** Recognise the information in faint emission and apparently empty regions.
- **Mask:** Define the 2D mask applied to the dataset for the model-fit.
- **Positions Constraints:** Use corresponding image features to constrain the mass model.
- **Choosing Positions:** Identify and plot features with a common source origin.
- **The Positions Penalty:** Understand the source-plane threshold and fast consistency check.
- **Model and Analysis:** Add the positions constraint to the imaging analysis.
- **Search and Result:** Fit the image with the positions constraint.
- **Contaminating Light:** Recognise light not described by the main lens and source.
- **Extra Galaxy Dataset:** Inspect an image containing a neighbouring galaxy.
- **Extra Galaxy Region:** Load the region selected for noise scaling.
- **Noise Scaling:** Reduce the weight of contaminated pixels.
- **Interpreting the Scaled Dataset:** Distinguish lost light information from gravitational deflection.
- **Interferometer Data:** Adapt the shared model composition API to visibilities.
- **Point-Source Data:** Fit measured positions and their uncertainties.
- **Weak-Lensing Data:** Constrain mass using background galaxy shapes.
- **Multiple Datasets:** Combine suitable likelihoods to constrain shared parameters.
- **Wrap Up:** Connect datasets, models, analyses and searches.

"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
__Dataset__

Return to the image from *Practicalities*, containing both the foreground lens galaxy and the lensed source.

Look first at the bright central galaxy, then follow its light outwards. Its brightness decreases gradually; it does
not end at a sharp boundary. Around it, the source forms bright arcs with fainter emission extending beyond their most
obvious features.

Our model describes both galaxies. We therefore want the fitted region to contain their useful light, including the
faint outskirts that are less immediately apparent in the image."""

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

"""
__Circular Masks__

A circular mask gives us one continuous region containing the central lens galaxy, the arcs and their surroundings. We
fit the lens and source light together within this region.

The figure below compares this with an annular mask, which excludes the centre. Although the annulus retains the bright
arcs, it discards the central lens light that helps constrain the foreground galaxy's profile.

Removing the centre does not remove the lens galaxy's contribution to the arcs. Its extended light still overlaps the
source emission farther out, so we still need to describe it in our model.

Throughout these PyAutoLens analyses, we therefore use circular masks. The annulus is shown here to explain the
information it would discard, rather than as another masking procedure to follow.

![Circular and annular masks compared on the same simulated lens image; grey shading marks excluded pixels](../../images/chapter_2/circular_vs_annular.png)

__Beyond the Bright Arcs__

The circle should extend beyond the brightest features. Faint source emission can help constrain the source's extent,
while the outskirts of the lens galaxy help constrain how its brightness falls with radius.

Individual pixels in these regions may contain only a weak signal. Taken together, they can still distinguish between
models that look similar around the bright centre and arcs.

Regions without detectable emission are useful too. A mass model might reproduce the observed arcs but predict an
additional image elsewhere. If that location lies inside our circle, the extra predicted light disagrees with the data
and reduces the likelihood.

If we exclude that location, the likelihood cannot see the disagreement. A mask drawn tightly around the observed arcs
could therefore hide evidence that the lens model is wrong.

This is why we retain a generous circular region around the system, including the space between and around its bright
features.

__Mask__

We use a circular mask with a radius of 3.0 arcseconds for this example. It contains the central galaxy, the lensed
arcs and surrounding pixels that constrain their fainter emission.

The aim is to include the system as a whole, rather than trace the outlines of individual bright features. Galaxy
profiles have extended faint wings, so this does not mean capturing every last photon.

Including more pixels generally adds computation, but making the fitted region unnecessarily small can discard
information that helps distinguish competing lens models.

In real observations, the surrounding region may also contain neighbouring galaxies. We will address their light later
in this tutorial, while retaining the large circular region used to fit the lens."""

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Positions Constraints__

Look again at the lensed source. Its multiple images are different views of the same background galaxy. If we identify
the same feature in two or more images, a suitable mass model should trace those positions back to approximately the
same location in the source plane.

An unsuitable mass model may instead trace them to widely separated locations. It could reproduce some of the observed
light while failing this requirement of the lensing geometry.

We can supply these image positions to the analysis and check their source-plane separation for every proposed mass
model. When that separation exceeds a chosen threshold, a likelihood penalty discourages the search from exploring the
model.

This gives the search a useful constraint: the marked features must be explainable as multiple images of the same
source feature. It does not tell the search the correct mass parameters in advance.

__Choosing Positions__

Positions must refer to the same source feature seen in different lensed images. For example, a compact bright knot may
appear in several arcs. Marking its location in each image gives us positions that should trace back together.

Simply choosing the brightest pixel in each arc is not always reliable. An extended source may contain several bright
knots, and different knots may dominate different images. Foreground lens light can also make a feature's position
difficult to identify.

For an individual lens, we can choose positions by inspecting the image or using the workspace's data-preparation GUI.
Always plot the selected positions over the image and check which features they mark.

In *Search Chaining*, we will introduce a more automated approach: using an initial fitted model to estimate positions
for a subsequent analysis.

For this example, we supply four positions corresponding to the centre of the simulated source, seen in different parts
of the arcs. These coordinates were checked against the simulation's mass model and rounded to three decimal places.
They are in `(y, x)` order, measured in arcseconds. Plot them over the image before fitting."""

positions = al.Grid2DIrregular(
    values=[(0.271, 1.673), (-1.499, 0.195), (1.499, -0.195), (-0.271, -1.673)]
)

aplt.subplot_imaging_dataset(dataset=dataset, positions=positions)

"""
__The Positions Penalty__

For each proposed mass model, PyAutoLens ray-traces the marked image positions to the source plane and calculates the
largest separation between them.

The threshold specifies how far apart we allow those source-plane positions to be, in arcseconds. If their maximum
separation exceeds it, the analysis applies a large likelihood penalty. If they fall within it, no positions penalty is
applied.

This check is extremely fast compared with evaluating the full imaging likelihood. It only requires ray-tracing a
handful of positions and calculating their separations, rather than predicting and comparing the light across thousands
of image pixels. It therefore adds very little computational cost while providing useful guidance to the search.

We use this as a broad consistency check. The marked positions have uncertainties, and features in extended arcs may
not identify exactly the same source location. A threshold that is too restrictive can therefore penalise otherwise
suitable mass models.

A threshold that is too generous has the opposite weakness: substantially different source locations may pass the
check, providing little guidance to the search.

Here we use a broad threshold of 1.0 arcsecond. For comparison, our rounded coordinates trace within 0.001 arcseconds
of one another under the simulation's true model. The much larger threshold deliberately allows approximate agreement
while discouraging widely separated solutions; it is not a measurement of positional uncertainty or a universal setting
for other lenses.

The positions come from the image we already fit. Here, the penalty is a practical constraint for guiding exploration,
rather than an additional independent measurement with its own positional-error likelihood.

Passing this check does not mean that a model fits the image. It must still reproduce the observed light through the
imaging likelihood."""

positions_threshold = 1.0

"""
__Model and Analysis__

We reuse the model from *Linear Profiles*: Sérsic lens light, an elliptical isothermal mass profile, external shear and
exponential source light. The two light-profile intensities are solved during each likelihood evaluation.

The new ingredient belongs to the analysis. We create a `PositionsLH` object containing our marked positions and their
allowed source-plane separation, then pass it alongside the imaging dataset to `AnalysisImaging`.

This illustrates the separation introduced at the start of the tutorial. We have added a constraint to the analysis
without changing the physical components of the model or the interface used by the search."""

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp_linear.Sersic,
    mass=al.mp.Isothermal,
)

source = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=al.lp_linear.ExponentialCore,
)

field = af.Model(
    al.MassField,
    redshift=0.5,
    shear=al.mp.ExternalShear,
)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=field,
)

positions_likelihood = al.PositionsLH(
    positions=positions,
    threshold=positions_threshold,
)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
)

"""
__Search and Result__

Run Nautilus using this model and analysis. The search explores the same physical parameters as before, but the
analysis now discourages mass models that trace our marked features to widely separated source locations.

This can help the search avoid unsuitable regions of parameter space. It does not guarantee that the search finds the
global maximum, and incorrectly chosen positions or an overly restrictive threshold can steer it away from a good
solution.

Inspect the fitted image and normalised residuals after the search finishes. Check the central lens galaxy and the
surrounding arcs, including their fainter emission. Satisfying the positions constraint is only one requirement; the
model must also explain the light throughout the circular mask.

This fit is stored with HowToLens, as explained in *Practicalities*, so its result loads in seconds. Delete its
`tutorial_5_with_positions` folder in `output/howtolens/chapter_2/lens_sersic` to run it yourself."""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_5_with_positions",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)

result = search.fit(model=model, analysis=analysis)

print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Contaminating Light__

A large circular mask can contain light from objects other than the lens galaxy and its source. A neighbouring galaxy
may appear beside an arc, with its fainter outskirts overlapping the emission we want to fit.

Our current light model contains only the main lens and source. If we ask it to explain the neighbouring galaxy too,
the search may distort those components to reproduce light that belongs to a different object.

One approach is to include the neighbour explicitly in the model. We will explore this in Chapter 4. Here, we introduce
**noise scaling**, which reduces the influence of selected pixels on the fit while retaining our circular mask.

This is useful when we want to prevent contaminating light from driving the fit, without introducing another light
profile and its free parameters.

__Extra Galaxy Dataset__

Load a second simulated image containing a neighbouring galaxy. Look towards the upper-right of the main lens, where
the extra galaxy's bright centre and extended light are visible.

Follow its light towards the lensed arcs. The contamination is not confined to the galaxy's brightest pixels, and some
pixels contain light from more than one object.

This example will show how we change the contribution of those pixels to the likelihood. We begin with the full image,
before applying either noise scaling or the circular mask."""

dataset_name_extra = "lens_extra_galaxy"
dataset_path_extra = Path("dataset") / "imaging" / dataset_name_extra

al.util.dataset.simulate_if_needed(
    dataset_path_extra,
    simulator="scripts/simulator/lens_extra_galaxy.py",
)

dataset_extra = al.Imaging.from_fits(
    data_path=dataset_path_extra / "data.fits",
    noise_map_path=dataset_path_extra / "noise_map.fits",
    psf_path=dataset_path_extra / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset_extra)

"""
__Extra Galaxy Region__

The simulator supplies a file identifying a region around the extra galaxy. For real observations, we would define this
region ourselves after inspecting the image, for example using the workspace's data-preparation GUI.

This selection has a different purpose from our main circular mask. The circular mask defines the overall region we
fit. The extra-galaxy selection identifies pixels within that region whose influence we want to reduce.

The supplied region surrounds the neighbour's centre and extends into its fainter light. It is an approximation to the
contaminated area, rather than a boundary at which the galaxy's light suddenly ends.

We load the file with `invert=True` so that its selected pixels follow the convention expected by the noise-scaling
method."""

mask_extra_galaxies = al.Mask2D.from_fits(
    file_path=dataset_path_extra / "mask_extra_galaxies.fits",
    pixel_scales=dataset_extra.pixel_scales,
    invert=True,
)

"""
__Noise Scaling__

Recall that the imaging likelihood compares the difference between the data and model with the uncertainty in each
pixel. A difference has less influence when the assigned uncertainty is larger.

Noise scaling uses this relationship to give contaminated pixels almost no weight. By default, the method below sets
their noise values to a very large number and their data values to zero.

This is a deliberate modification of the data used by the analysis. It does not mean that the original measurement had
such a large uncertainty, and it does not subtract a physical model of the neighbouring galaxy.

Apply the noise scaling first, then apply the same 3.0-arcsecond circular mask. The selected pixels remain inside the
fitted region, but differences between their data and predicted light contribute negligibly to the fit."""

dataset_scaled = dataset_extra.apply_noise_scaling(
    mask=mask_extra_galaxies,
)

mask = al.Mask2D.circular(
    shape_native=dataset_scaled.shape_native,
    pixel_scales=dataset_scaled.pixel_scales,
    radius=3.0,
)

dataset_scaled = dataset_scaled.apply_mask(mask=mask)

aplt.subplot_imaging_dataset(dataset=dataset_scaled)

"""
__Interpreting the Scaled Dataset__

Compare the data, noise-map and signal-to-noise panels. In the selected region, the data have been set to zero and the
noise has been increased. The galaxy has not physically disappeared; we have told the analysis to obtain almost no
information from those pixels.

This also means losing information about any lens or source light in the same pixels. Noise scaling cannot distinguish
which galaxy emitted a photon. Where the neighbour overlaps important parts of an arc, explicitly modeling its light
may preserve more useful information.

Light outside the selected region can still affect the fit, including faint wings spread by the PSF. Inspect the
surrounding pixels rather than assuming the selection removes every contribution from the neighbour.

There is a further distinction for lensing: reducing the weight of a galaxy's light does not remove its gravitational
deflection. The extra galaxy in this simulation also has mass. Whether that mass must be included in the lens model is
a separate question, which we develop in Chapter 4.

To use this prepared dataset, we would pass it to `AnalysisImaging`. The analysis then uses the modified data and noise
map when calculating the likelihood. The search interface remains unchanged."""

analysis_scaled = al.AnalysisImaging(dataset=dataset_scaled)

r"""
Changing the noise map changes the likelihood itself. We should therefore not interpret a change in the absolute likelihood before and after scaling as evidence that we have found a better physical model.

__Interferometer Data__

In Chapter 1, we saw that an interferometer measures **visibilities**, rather than the image pixels used throughout
this chapter.

We can still describe the lens using mass profiles and the source using light profiles. The model predicts a lensed
image, but that image must be transformed into predicted visibilities before we compare it with the observations.

`AnalysisInterferometer` performs this comparison using the measured visibilities and their uncertainties. We replace
the imaging dataset and analysis with their interferometer equivalents, while retaining the familiar process of
defining a model and running a search.

The model is built using the same composition API for every dataset type: `af.Model` defines profiles and galaxies, and
`af.Collection` groups them into the complete model. We use the same syntax to assign priors, fix parameters and link
parameters between components. The appropriate physical components may change, but we do not need to learn a different
way of constructing models.

The mask has a different role here. It defines the real-space grid on which we predict the lensed emission; it does not
select observed image pixels for the likelihood. The measurements being fitted are the visibilities.

The workspace's `scripts/interferometer/modeling.py` demonstrates the complete procedure. The light and mass profiles,
priors and search tools should already feel familiar—the main change is how the model predicts the measured quantities.

__Point-Source Data__

For a multiply imaged compact source, such as a quasar, we may measure the positions of its separate images and their
positional uncertainties. These measurements can constrain the lens mass without requiring us to fit an extended arc
pixel by pixel.

A `PointDataset` stores the measurements, and `AnalysisPoint` connects them to the lens model. The analysis evaluates
how consistently the model explains the observed image positions, using the chosen positional likelihood.

This differs from the positions penalty introduced earlier. That penalty provides a broad consistency constraint
alongside an imaging fit. A point-source analysis treats the measured positions and their uncertainties as the
observations to be fitted.

The physical model also changes where appropriate: a point source replaces the extended source-light profile. However,
we still define mass profiles, assign priors and use a search to explore their parameters.

See `autolens_workspace/scripts/point_source/modeling.py` for a complete example.

__Weak-Lensing Data__

Weak lensing uses the small distortions of many background galaxies to constrain a foreground mass distribution. Rather
than fitting bright arcs from one strongly lensed source, we work with a catalogue of galaxy positions and shape
measurements.

Each galaxy has an unknown intrinsic shape as well as measurement uncertainty. A single elongated galaxy therefore does
not tell us the lensing distortion directly. The information emerges from the pattern across many galaxies.

A `WeakDataset` contains the measurements used for this comparison. `AnalysisWeak` evaluates the lens model's predicted
distortion at the galaxy positions and compares it with the measured shapes, accounting for the adopted uncertainties.

Once again, the search does not need a new interface. It explores the mass-model parameters using the likelihood
returned by the analysis.

Chapter 4 develops weak-lensing modeling further. The workspace's `scripts/weak/modeling.py` also shows how these
components fit together.

__Multiple Datasets__

Separating datasets from models also allows several observations to constrain the same physical system.

Imagine observing one lens in two wavebands. Both images are produced by the same foreground mass distribution, but the
lens and source brightness can differ between wavebands. Each observation also has its own PSF and noise map.

We can give each dataset its own Analysis object and connect both to a model that shares the mass parameters while
allowing appropriate light-profile parameters to differ.

For datasets that are independent given the model, their likelihoods multiply. Equivalently, their log likelihoods add:

$$
\ln \mathcal{L}_{\mathrm{total}} = \ln \mathcal{L}_1 + \ln \mathcal{L}_2.
$$

The search then explores parameter values using this combined likelihood. A proposed mass model must explain both
observations, rather than fitting either one in isolation.

This can also combine different kinds of measurements, such as imaging and point-source positions. We must take care
with information shared between datasets: treating the same measurement twice as independent would count its
information twice.

Joint modeling is an advanced topic. The workspace's `scripts/multi_dataset/` examples explain how to construct these
analyses and decide which parameters to share. For now, the important idea is that each dataset can have its own
prediction and likelihood calculation while contributing to a common model fit.

__Wrap Up__

We began with familiar imaging data and examined how different information enters the fit. The circular mask retains
the lens, arcs and their surroundings. Positions constraints help identify suitable mass models. Noise scaling reduces
the influence of selected contaminated pixels.

These changes belong to different parts of the analysis, but they all fit into the same structure:

- The **dataset** contains the measurements and their uncertainties.
- The **model** describes the physical system and its free parameters.
- The **Analysis** object connects the model to the observations and calculates their likelihood.
- The **search** explores the parameters using that likelihood and the priors.

This separation is why the tools learnt in this chapter extend beyond fitting one image. Whether we fit imaging,
visibilities, point-source positions or weak-lensing shapes, we compose the model using the same `af.Model` and
`af.Collection` API. We adapt its physical components to the problem, select the appropriate dataset and Analysis
objects, and pass them to the search through the familiar `search.fit(model=model, analysis=analysis)` interface.

In the next tutorial, we will use this flexibility to connect several fits in a sequence. An initial, simpler analysis
will provide information that helps us construct and fit a more complex model.

**🤖 Ask AI:** I have two images of the same gravitational lens taken in different wavebands. Which parts of the physical model might be shared, which might differ, and what must each Analysis object know about its own observation?
"""
