r"""
Tutorial 5: Cluster Scale
=========================

We now extend the tools developed for galaxies and groups to **galaxy clusters**.

Galaxy clusters typically inhabit dark matter halos with masses of roughly $10^{14}$-$10^{15}$ solar masses,
with the most massive systems exceeding $10^{15}$ solar masses. Within this shared halo, a cluster can contain
**hundreds of individual lens galaxies**, including a brightest cluster galaxy, other prominent galaxies and a
large population of lower-mass members.

Their combined mass can produce multiple images of many background sources at different redshifts.

These systems offer two important scientific opportunities:

- **Studying magnified background galaxies:** lensing can make faint sources brighter and stretch their images,
  revealing details that would otherwise be difficult to observe.
- **Mapping the cluster's mass distribution:** multiple-image systems at different positions and redshifts
  constrain the combined contributions of the cluster halo and its galaxies.

High magnification occurs near critical curves in both galaxy- and cluster-scale lenses. Clusters provide
extended regions in which many background sources can be lensed, but the magnification of an individual source
depends on its position, size and the lens model.

In this tutorial, we fit **multiple-image positions** rather than the full surface-brightness distribution of
every arc. This provides a compact set of constraints for exploring the cluster mass model.

We also introduce CSV catalogues for the image positions, individually modeled components and member population.
These make larger systems easier to organize while retaining the model-composition tools used earlier.

__Contents__

- **Multi-Plane Ray Tracing:** Distinguish multiple source planes from multiple deflecting planes.
- **Dataset:** Load the simulated cluster's CCD imaging and inspect the true multi-plane tracer.
- **Point Source Modeling:** Why cluster modeling fits multiple-image positions rather than extended arcs.
- **Point Datasets:** The per-source multiple-image positions, loaded from a single hand-editable CSV.
- **The CSV Interface:** The named-galaxy model CSVs and the scaling-member catalogue CSV.
- **Point Solver:** The solver which finds the image-plane multiple images of a point source.
- **Model:** Compose the four-tier cluster model (main lenses, scaling members, host halo, sources).
- **Analysis + Factor Graph:** One analysis per source dataset, combined into a single global fit.
- **Search:** Configure the Nautilus non-linear search.
- **Model Fit:** Run the fit.
- **Results:** A thorough tour of results access for a cluster fit — tracer, per-member profiles,
  solved source-plane positions, magnifications and image-plane validation.
- **Customization:** Where to go for the full point-source customization options.
- **Wrap Up:** Summary and the hand-off to weak lensing.

__Multi-Plane Ray Tracing__

Our simulated cluster places all deflecting mass at redshift $z=0.5$, with background sources at $z=1.0$ and
$z=2.0$.

This gives us **one deflecting plane and two source planes**. Each source is mapped through the same cluster
mass distribution, but with the lensing-distance factors appropriate to its redshift.

Different source redshifts therefore require different geometric scaling; they do not automatically imply that
light is deflected by mass on several planes.

Genuine multi-plane deflection arises when significant mass occupies more than one redshift plane. For example,
a galaxy at $z=1.0$ could both be lensed by the cluster and deflect light from a more distant source at $z=2.0$.

In that case, the ray's position at a later plane depends on the deflections accumulated at earlier planes.
The multi-plane lens equation accounts for this sequentially.

The `Tracer` supports both situations. It organizes galaxies by redshift and computes the appropriate mappings
to each source plane. Our example uses that machinery with mass on the cluster plane only.

Source redshifts are therefore essential inputs. Here they are known from the simulation; observational
redshift uncertainties must be considered when interpreting a real cluster model.
"""

import importlib.util
import sys

if importlib.util.find_spec("jax") is None:
    print(
        "Skipping this tutorial: it requires the `jax` package (used to "
        "accelerate the multi-plane point-source solves), which is not "
        "installed (install with `pip install autolens[optional]`)."
    )
    sys.exit(0)

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

We use a simplified simulated cluster containing:

- Two individually modeled lens galaxies: a brightest cluster galaxy, or BCG, and a satellite.
- Ten additional member galaxies whose masses follow a luminosity-based scaling relation.
- A shared dark matter halo with input `mass_at_200` of $10^{15.3}$ solar masses.
- Two multiply imaged sources at redshifts 1.0 and 2.0.

The cluster's deflecting components are all at redshift 0.5.

This example demonstrates the catalogue and fitting workflow with a small member population. Adding more members
need not add non-linear mass parameters when their positions and profile properties are fixed or tied through
the same relation, although evaluating their deflections still adds computational work.

Real clusters can require additional halos, more flexible galaxy models and further treatment of redshift or
membership uncertainties.
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "cluster" / dataset_name

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/simulator/cluster.py"],
        check=True,
    )

"""
We first load and plot the simulated cluster image. Its pixel scale is 0.2" and its field spans 100" x 100",
allowing us to see the cluster galaxies and widely separated lensed images.

Inspect where the multiple images lie relative to the central galaxies and member population.

The angular extent of strong lensing depends on the mass distribution and source redshift, not simply the
cluster's total halo mass.

We use this image for visualization. The fit below uses measured multiple-image positions rather than these
image pixels.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.2)

aplt.plot_array(array=data, title="Cluster CCD Imaging")

"""
We load the `Tracer` saved by the simulator to inspect its redshift structure.

It contains three planes: the cluster plane at $z=0.5$, followed by source planes at $z=1.0$ and $z=2.0$.

The cluster plane contains 13 `Galaxy` objects: two main galaxies, ten scaling-relation members and one mass-only
object carrying the shared dark matter halo. The halo's use of a `Galaxy` container is an API convention; it does
not represent an additional luminous galaxy.

All deflecting mass in this example lies on the cluster plane. The two source planes specify where we evaluate
the mappings for their respective background sources.
"""
tracer_true = al.from_json(file_path=dataset_path / "tracer.json")

print(f"Number of planes: {len(tracer_true.planes)}")
print(f"Plane redshifts: {[float(plane.redshift) for plane in tracer_true.planes]}")
print(f"Galaxies in lens plane: {len(tracer_true.planes[0])}")

"""
__Point Source Modeling__

We now describe each multiply imaged source using a set of image positions rather than fitting its full
surface-brightness distribution.

For extended galaxies, these positions should identify **corresponding features** in the different images -- for
example, recognizable knots that represent the same source structure. Selecting each image's brightest pixel
independently does not necessarily identify matching features.

This reduces the amount of data evaluated during the fit and avoids constructing a light-profile or pixelized
reconstruction for every source. It is a widely used approach for building cluster mass models.

The trade-off is that position fitting does not use the full information in the extended arcs. Surface-brightness
modeling can provide complementary constraints and remains possible for selected systems or more detailed analyses.

For genuine point sources such as quasars or supernovae, image positions are natural measurements, although
fluxes, time delays and other observables can provide additional information.

__Point Datasets__

We load the multiple-image measurements from `point_datasets.csv`, with one row per observed image:

- `name`: identifies the source or matched source feature. Rows sharing a name form one multiple-image system.
- `y`, `x`: its observed image-plane coordinates in arcseconds.
- `positions_noise`: the positional uncertainty adopted by the likelihood, in arcseconds.
- `redshift`: the source redshift, shared by the rows belonging to that system.

The positional uncertainty should reflect the measurement and modeling assumptions. Precise centroid measurements
do not imply that a simplified cluster mass model predicts positions to the same precision; unresolved structure
and other modeling limitations can also matter.

For this simulation, we use the supplied uncertainties. For real observations, choose and justify them rather
than adopting a universal value from the pixel scale or telescope resolution.

`al.list_from_csv` loads the file into a list of `PointDataset` objects, one per named system.
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

for dataset in dataset_list:
    print("Point Dataset Info:")
    print(dataset.info)
    print(f"Redshift: {dataset.redshift}")

"""
Plot each dataset to inspect its multiple-image positions.

Each system supplies a small set of coordinates that must be consistent with a common source position under the
lens model. Systems distributed across the cluster probe different parts of its deflection field, while their
redshifts determine the appropriate lensing geometry.

The information available depends on the number and configuration of images, their uncertainties and the model's
flexibility. A small number of precise positions does not automatically determine a unique cluster mass
distribution.
"""
for dataset in dataset_list:
    aplt.plot_grid(
        grid=al.Grid2DIrregular(np.atleast_2d(dataset.positions)),
        title=dataset.name,
    )

"""
__The CSV Interface__

CSV files provide a convenient way to organize a model containing many components. They separate the measured
catalogue and profile choices from the Python code that assembles the model.

The simulator supplies:

- `mass.csv`: the individually described mass profiles, including the main galaxies and shared dark matter halo.
- `light.csv`: light-profile descriptions, which are not needed for this positions-only likelihood.
- `point.csv`: the source point components.

Each profile row specifies a galaxy name, attribute name, profile class, redshift and relevant profile parameters.
Rows with the same galaxy name are assembled into one `Galaxy`.

The lower-mass member population is listed separately in `scaling_galaxies.csv`, with measured positions and
luminosities. We assign its mass profiles and shared scaling relation in Python.

Adding members to this catalogue does not add independent mass parameters **under our fixed-centre,
spherical-profile and shared-relation assumptions**. Adding an individually modeled galaxy or halo can add
parameters, depending on which quantities we allow to vary.

The catalogue format makes a larger model easier to assemble; the modeling assumptions determine its
dimensionality.

We now load the mass and point-profile tables, followed by the member catalogue.
"""
mass_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "mass.csv", family="mass"
)
point_table = al.galaxy_models_from_csv(
    file_path=dataset_path / "point.csv", family="point"
)

scaling_galaxies_table = al.galaxy_table_from_csv(
    file_path=dataset_path / "scaling_galaxies.csv"
)
scaling_galaxies_centres = scaling_galaxies_table.centres
scaling_galaxies_luminosity_list = scaling_galaxies_table.luminosities

print(f"Scaling galaxies in catalogue: {len(scaling_galaxies_luminosity_list)}")

"""
In real observations, member positions and luminosities can come from preliminary light fits or a suitable
catalogue. As in earlier tutorials, fixing member mass centres to measured light positions is a useful way to
limit the model's freedom.

These measurements are not exact, and selected galaxies can be given positional freedom where the data and
scientific question justify it.

__Point Solver__

We met the `PointSolver` earlier when finding the multiple images of a point source, such as a lensed quasar,
behind a single lens galaxy.

We use exactly the same tool here. Given a mass model and a source position, it solves the lens equation
numerically to locate the multiple images. The difference is that our mass model now contains a cluster halo
and many galaxies, and we repeat the calculation for sources at their respective redshifts.

Recall that the solver divides the image plane into triangles, maps them to the source plane and refines
candidate regions until the requested positional resolution is reached.

The grid should cover the region in which we want to search for images. Here, it spans the same approximate 100"
field as the simulated imaging. Images outside that search region will not be found.

`pixel_scale_precision` sets the solver's numerical resolution in arcseconds; it is separate from the
observational uncertainties in `positions_noise`.

`magnification_threshold` filters out predicted images with low absolute magnification. This is a numerical
selection rule, not a complete model of detectability, which also depends on source brightness, noise and blending.

Unlike an image-plane position likelihood that uses the solver during fitting, our chosen cluster likelihood
back-traces the observed positions without solving for predicted images at every evaluation. We will use the
familiar `PointSolver` after fitting to check the predicted image configuration for each source.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=1.0,  # The pixel-scale converts pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__

Our example combines four types of component:

- **Two main lens galaxies:** each has a spherical dPIE mass profile with fixed centre and core radius, but free
  `sigma` and `r_cut`. Together they contribute **four parameters**.
- **Ten scaling-relation members:** their positions and core radii are fixed. Their velocity-dispersion
  normalizations depend on one shared `sigma_ref`; their truncation radii follow a fixed luminosity prescription.
  This population contributes **one parameter**.
- **One shared dark matter halo:** an `NFWMCRLudlowSph` profile with fixed centre and free `mass_at_200`.
  Its mass-concentration prescription supplies additional structure without an independent concentration
  parameter. It contributes **one parameter**.
- **Two background sources:** `PointSolved` components whose source positions are solved within the selected
  likelihood rather than sampled independently. They contribute **zero non-linear sampling parameters**.

The total is therefore **six free non-linear parameters**.

This small count reflects strong restrictions on the model: spherical profiles, fixed positions and redshifts,
a prescribed halo mass-concentration relation, and a deterministic member scaling relation. It is not a claim
that real cluster mass distributions generally require only six parameters.

The model contains twelve lens galaxies, one mass-only halo component and two source objects. Adding more tied
members preserves the parameter count only while those restrictions remain in place.

`al.galaxy_af_models_from_csv_tables` builds the named galaxy models from the CSV tables, initially fixing their
supplied values. We then assign priors to the particular parameters we want the search to vary.
"""
redshift_lens = 0.5
source_redshifts = [dataset.redshift for dataset in dataset_list]

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

"""
We give each main lens galaxy two free dPIE parameters: `sigma`, which controls its mass normalization, and
`r_cut`, which controls its outer truncation scale.

Their centres, redshifts and zero core radii retain the values supplied by the catalogue. As in the group
tutorial, `sigma` uses the Lenstool profile convention and is not automatically identical to a measured stellar
velocity dispersion.

We explicitly fix `H0` and `Om0`, because this example assumes a cosmology rather than attempting to infer it.

The prior ranges below are choices for this simulated example. For real data, they should reflect the system
being studied, and results near their boundaries warrant further inspection.
"""
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(
        lower_limit=50.0, upper_limit=600.0
    )
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=2.0, upper_limit=40.0)
    galaxy_models[name].mass.H0 = 67.66
    galaxy_models[name].mass.Om0 = 0.30966

"""
The shared dark matter halo receives a free `mass_at_200` parameter, expressed in solar masses.

This is the mass parameter of the NFW host component, not the sum of every mass component in our cluster model.
The main galaxies and scaling-relation members contribute additional mass.

The `NFWMCRLudlowSph` profile uses a prescribed mass-concentration relation, so changing the halo mass also changes
its associated profile structure. We do not fit concentration independently here.

We use a log-uniform prior from $10^{14.5}$ to $10^{16}$ solar masses, assigning equal prior weight to equal
logarithmic intervals. This is a broad example range, rather than a universal prior for cluster lenses.
"""
galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

"""
We replace each source's point component with `PointSolved`.

With the source-plane likelihood used below, the source position is solved from the back-traced image positions
for each proposed mass model, rather than sampled as two independent coordinates.

This does not mean the source position is known exactly or remains fixed. It changes as the mass model changes,
and its interpretation depends on the positional uncertainties and likelihood assumptions.

Each component name, such as `point_0`, must match the name of its `PointDataset`. We also retain the source
redshift associated with that dataset so that its images are traced to the correct plane.
"""
for i, dataset in enumerate(dataset_list):
    setattr(galaxy_models[f"source_{i}"], f"point_{i}", af.Model(al.ps.PointSolved))

r"""
We build the member population by looping over its measured positions and luminosities, using the same approach
as in the previous tutorials.

The shared `scaling_sigma_ref` prior is defined once, outside the loop. Each member's `sigma` is an expression
involving that prior and its fixed luminosity ratio.

Each `r_cut`, however, is calculated entirely from fixed quantities: the reference truncation scale, luminosity
ratio and exponent. It therefore receives a fixed value, rather than depending on `scaling_sigma_ref`.

We adopt $\sigma\propto L^{0.25}$ and $r_{\rm cut}\propto L^{0.7}$, with fixed centres and zero core radii.
These assumptions leave one free mass parameter for the entire member population.

The simulator used `sigma_ref = 85 km/s`. We will later inspect whether the fit constrains this normalization.

The dPIE `redshift_source` input specifies the reference source redshift used to normalize the profile's lensing
quantities. Here, we use the highest source redshift consistently for the scaling members. The tracer then
evaluates the lens mapping to each source's own plane; this does not place every source at that reference
redshift.
"""
scaling_sigma_ref = af.UniformPrior(lower_limit=0.0, upper_limit=200.0)
scaling_sigma_exponent = 0.25  # alpha (Faber-Jackson)
scaling_gamma = 0.2  # mass-to-light tilt, fixed
scaling_rcut_exponent = 1.0 + scaling_gamma - 2.0 * scaling_sigma_exponent  # 0.7

reference_luminosity = 1.0
scaling_r_core_fixed = 0.0
scaling_r_cut_ref_fixed = 5.0

scaling_galaxies_list = []
for centre, luminosity in zip(
    scaling_galaxies_centres, scaling_galaxies_luminosity_list
):
    luminosity_ratio = luminosity / reference_luminosity

    mass = af.Model(al.mp.dPIEMassSph)
    mass.centre = tuple(centre)
    mass.sigma = scaling_sigma_ref * luminosity_ratio**scaling_sigma_exponent
    mass.r_core = scaling_r_core_fixed
    mass.r_cut = scaling_r_cut_ref_fixed * luminosity_ratio**scaling_rcut_exponent
    mass.redshift_object = redshift_lens
    mass.redshift_source = max(source_redshifts)
    mass.H0 = 67.66
    mass.Om0 = 0.30966

    scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

scaling_galaxies = af.Collection(scaling_galaxies_list)

"""
We combine the individually named components and scaling-relation members into the overall model.

The `galaxies` collection contains the two main lens galaxies, the mass-only host-halo component and the two
sources. The `scaling_galaxies` collection contains the ten additional members.

This gives **15 `Galaxy` objects**, representing twelve lens galaxies, one shared dark matter halo and two
background sources.

Inspect `model.info` to check the fixed values, independent priors and linked expressions. Under the assumptions
above, the model has **six free non-linear parameters**: four for the main galaxies, one for the host halo and
one shared by the scaling members.
"""
model = af.Collection(
    galaxies=af.Collection(**galaxy_models),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
The figure shows the model's independent priors, fixed values and linked expressions.

Each main galaxy has its own free `sigma` and `r_cut`. The host halo has one free mass parameter, and the ten
scaling members share one `scaling_sigma_ref` prior.

The source positions are solved within the selected likelihood rather than sampled independently.

These choices give six non-linear parameters across fifteen `Galaxy` objects. The small parameter count comes
from the restrictions and shared relations we imposed, rather than from the catalogue format itself.
"""
af.ModelPlotter(model).figure()

"""
__Analysis + Factor Graph__

We create one `AnalysisPoint` for each source's dataset. Each analysis evaluates how well the same cluster mass
model explains that source's observed multiple-image positions.

We select `FitPositionsSourceSolved`, which:

1. Traces the observed image positions back to the source's redshift plane.
2. Solves for a common source position under the adopted weighting.
3. Evaluates the scatter around that position using the positional uncertainties and the likelihood's
   source-plane weighting.

This avoids solving for predicted image positions at every likelihood evaluation. It is a source-plane fitting
approach, rather than a direct comparison between observed and forward-solved image positions.

We will use the `PointSolver` afterwards to inspect the predicted images. That check is important because a
compact set of back-traced positions does not by itself establish that the model predicts the observed image
configuration.

We combine the analyses using a **factor graph**. Each source contributes a likelihood factor, and all factors
share the same global mass model.

Assuming the datasets are conditionally independent given that model, the total log likelihood is the sum of
their individual log likelihoods. The shared parameters are sampled once, with every source contributing
constraints through its own positions and redshift.
"""
analysis_list = [
    al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsSourceSolved,
        use_jax=True,
    )
    for dataset in dataset_list
]

analysis_factor_list = [
    af.AnalysisFactor(prior_model=model, analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

"""
__Search__

We use `Nautilus` with 100 live points as the starting configuration for this six-parameter example.

A low parameter count helps, but does not guarantee a simple posterior. Correlations between the halo, main
galaxies and member normalization can still make inference challenging.

Inspect the resulting fits and posterior distributions, and assess whether important conclusions remain stable
under repeated runs or more demanding search settings.
"""
search = af.Nautilus(
    path_prefix=Path("howtolens", "chapter_4"),
    name="tutorial_5_cluster_scale",
    unique_tag=dataset_name,
    n_live=100,
    n_batch=50,
    iterations_per_quick_update=10000,
)

"""
__Model Fit__

We pass the factor graph's global model and combined analysis to the search.

Each proposed parameter set defines one cluster mass model, which is evaluated against both source datasets.
Their log likelihoods combine to guide the search.

The position-based likelihood avoids constructing a full model image at each evaluation. Runtime nevertheless
depends on the model, number of images and members, hardware and search settings.

When the search finishes, we will inspect both the global mass parameters and the fit to each source.
"""
print(
    "The non-linear search has begun. "
    "Progress updates are printed during the search. "
    "This cell will finish when the search completes."
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished — you may now continue the notebook.")

"""
__Results__

The factor-graph fit returns one result per source dataset, in the same order as `dataset_list`.

These results come from **one joint inference**, not separate searches. They share the global mass model and
posterior, while each provides access to its own dataset and corresponding `FitPointDataset`.

We first use one result to inspect the shared model parameters and tracer. We then loop over the results to
examine each source's inferred position and fit diagnostics.

This separates two useful questions: what mass distribution describes the cluster, and how well does that
distribution explain each observed multiple-image system?
"""
print(f"Number of results (one per source dataset): {len(result_list)}")

result = result_list[0]

"""
__Results: The Maximum Likelihood Instance__

The `max_log_likelihood_instance` contains the parameter values of the highest-likelihood sample found by
the search.

Its structure follows the model we constructed: named components appear under `.galaxies`, while the tied
members appear under `.scaling_galaxies`.

We can inspect the main galaxies' dPIE parameters and the host halo's `mass_at_200` directly. These describe one
fitted realization; later, we use the posterior samples to examine their uncertainties and correlations.
"""
instance = result.max_log_likelihood_instance

print("Max Log Likelihood Model:")
print(f"BCG sigma: {instance.galaxies.lens_0.mass.sigma:.1f} km/s")
print(f"BCG r_cut: {instance.galaxies.lens_0.mass.r_cut:.2f} arcsec")
print(f"Satellite sigma: {instance.galaxies.lens_1.mass.sigma:.1f} km/s")
print(
    f"Halo mass_at_200: {instance.galaxies.host_halo.dark.mass_at_200:.2e} solar masses"
)

"""
__Results: Per-Member Profiles__

Although the member population shares one free normalization, the fitted instance contains a complete
`dPIEMassSph` profile for every member.

Each member's `sigma` has been calculated from the fitted shared normalization and its luminosity ratio.
Its centre, core radius and luminosity-derived truncation radius retain their fixed values.

The loop below lists these realized profile parameters. They are not independent measurements of every member's
mass: they inherit the scaling relation and its assumptions.

To quantify their uncertainties, evaluate the derived quantities across posterior samples. With fixed
luminosities and exponent, the members' `sigma` values vary together through their shared normalization.
Intrinsic scatter and photometric uncertainty are not included unless explicitly modeled.
"""
for i, member in enumerate(instance.scaling_galaxies):
    print(
        f"Member {i}: centre={member.mass.centre}, "
        f"sigma={member.mass.sigma:.1f} km/s, r_cut={member.mass.r_cut:.2f} arcsec"
    )

"""
__Results: The Tracer__

`max_log_likelihood_tracer` assembles the named components and scaling members into a tracer for the fitted model.

It provides the same lensing calculations used in earlier chapters, now with the full cluster mass distribution
and both source redshifts.

Inspect the plane redshifts and number of objects in the lens plane. Remember that the host halo occupies a
mass-only `Galaxy` object alongside the twelve lens galaxies.

When calculating quantities such as critical curves or magnifications, specify the relevant source plane:
the same cluster produces different lens mappings for sources at different redshifts.

This tracer represents one fitted realization. Uncertainty in derived lensing quantities requires repeating
the calculation over appropriate posterior samples.
"""
tracer = result.max_log_likelihood_tracer

print(f"Tracer planes: {[float(plane.redshift) for plane in tracer.planes]}")
print(f"Galaxies in lens plane: {len(tracer.planes[0])}")

aplt.subplot_tracer(tracer=tracer, grid=grid)

"""
__Results: Solved Source-Plane Positions__

Each result's `max_log_likelihood_fit` describes the fit to its own source dataset.

With our selected likelihood, `fit.positions.source_plane_coordinate` gives the source position solved from
that system's back-traced image positions under the fitted mass model.

The `residual_map` records their offsets from this common position in the source plane. Inspect these together
with the positional uncertainties and likelihood weighting, rather than judging the fit solely by whether
the raw offsets look small.

The printed log likelihood is the position-likelihood contribution for that source. The joint inference combines
the contributions from all source datasets.
"""
for result in result_list:
    fit = result.max_log_likelihood_fit

    print(f"Source '{result.max_log_likelihood_fit.dataset.name}':")
    print(
        f"  Solved source-plane centre: {np.asarray(fit.positions.source_plane_coordinate)}"
    )
    print(f"  Source-plane residuals: {np.asarray(fit.positions.residual_map)}")
    print(f"  Log likelihood: {float(fit.positions.log_likelihood):.2f}")

"""
We can inspect the same mapping directly using the tracer, following the ray-tracing approach introduced in
Chapter 1.

For each dataset, we identify its source plane from its redshift and trace the observed image positions through
the model. We then select the coordinates at that source plane.

Images identified as the same source feature should map to a common location within the accuracy allowed by the
data and model. Differences between their back-traced positions reveal where the proposed mass distribution
does not bring them into agreement.

This calculation displays the traced coordinates. It does not reproduce the full likelihood weighting or
independently calculate the solved source position.
"""
for dataset in dataset_list:
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)
    traced_grids = tracer.traced_grid_2d_list_from(grid=dataset.positions)

    print(f"Source '{dataset.name}' (z={dataset.redshift}, plane {plane_index}):")
    print(f"  Back-traced source-plane positions: {traced_grids[plane_index].in_list}")

"""
__Results: Magnifications__

The magnification of each multiple image tells us how much the cluster brightens the source at that position —
the "natural telescope" number. It is computed from the local distortion of the multi-plane deflection field
(its Hessian) at each observed position, via the `LensCalc` object. `plane_j` selects the source's plane, so
each source's magnifications are evaluated through its own multi-plane chain.

At cluster scale, images near the critical curves can reach magnifications of tens to hundreds — this is
precisely the quantity you would quote when using the cluster to study an intrinsically faint background
galaxy, since the source's true luminosity is the observed luminosity divided by the magnification.
"""
for dataset in dataset_list:
    plane_index = tracer.plane_index_via_redshift_from(redshift=dataset.redshift)

    lens_calc = al.LensCalc.from_tracer(
        tracer=tracer, use_multi_plane=True, plane_j=plane_index
    )
    magnifications = lens_calc.magnification_2d_via_hessian_from(grid=dataset.positions)

    print(
        f"Source '{dataset.name}' image magnifications: {np.abs(np.asarray(magnifications))}"
    )

"""
__Results: Image-Plane Validation__

The solved source-plane likelihood never solves the lens equation forwards, so it cannot tell us whether the
best-fit model predicts the *right number* of multiple images — a model that would produce an extra image, or
lose an observed one, looks the same to it. The standard final check is therefore to forward-solve the
best-fit model with the `PointSolver` and compare the predicted image positions to the observed ones,
per source.
"""
for result, dataset in zip(result_list, dataset_list):
    fit = result.max_log_likelihood_fit

    source_plane_centre = tuple(np.asarray(fit.positions.source_plane_coordinate))

    model_positions = solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_plane_centre,
        plane_redshift=dataset.redshift,
    )

    print(f"Source '{dataset.name}':")
    print(
        f"  Observed positions ({len(dataset.positions)}): {dataset.positions.in_list}"
    )
    print(f"  Model positions ({len(model_positions)}): {model_positions.in_list}")

    aplt.plot_grid(
        grid=model_positions,
        title=f"Model Multiple Images: {dataset.name}",
    )

"""
__Results: Samples and Errors__

Everything above used the maximum likelihood model, but the `Samples` object gives the full posterior — the
same API as chapter 2, unchanged by the cluster's complexity. Of particular scientific interest here are the
inferred halo mass and the scaling-relation normalization `sigma_ref` (whose truth value in the simulator was
85 km/s): the marginalised errors on these are what a cluster paper reports.
"""
samples = result_list[0].samples

median_pdf = samples.median_pdf()

print("Median PDF Model:")
print(f"Halo mass_at_200: {median_pdf.galaxies.host_halo.dark.mass_at_200:.2e}")
print(f"BCG sigma: {median_pdf.galaxies.lens_0.mass.sigma:.1f} km/s")

upper_3_sigma = samples.values_at_upper_sigma(sigma=3.0)
lower_3_sigma = samples.values_at_lower_sigma(sigma=3.0)

print("BCG sigma 3-sigma interval:")
print(
    f"  {lower_3_sigma.galaxies.lens_0.mass.sigma:.1f}"
    f" - {upper_3_sigma.galaxies.lens_0.mass.sigma:.1f} km/s"
)

aplt.corner_anesthetic(samples=samples)

"""
__Customization__

This tutorial used the default point-source setup throughout: `Point` / `PointSolved` source components, the
solved source-plane likelihood, and default `PointSolver` settings. Point-source modeling has a rich set of
options beyond these — image-plane likelihoods and their pairing schemes, free source centres, fitting the
fluxes of the multiple images, fitting time delays (the observable behind lensed-quasar cosmology) — and a
dedicated workspace guide covers them extensively:

 - `autolens_workspace/scripts/point_source`: the galaxy-scale point-source examples (lensed quasars and
   supernovae), including the `fit.py` guide to every fit variant and the `features` folder (fluxes, time
   delays, multiple sources).
 - `autolens_workspace/scripts/guides/point_solver/pairing.py`: the full matrix of position-fitting schemes
   and when to use each.
 - `autolens_workspace/scripts/cluster`: the cluster-scale workflow this tutorial is built on, including a
   fit to real Hubble Space Telescope data of the cluster Abell 2744 with 188 catalogue members and 7
   sources, the CSV-schema guide (`csv_api.py`) and a step-by-step walkthrough of the point-source
   likelihood (`likelihood_function.py`).

__Wrap Up__

This tutorial reached the top of the strong-lensing mass ladder. Lets recap what we learnt:

- **Clusters**: 30+ to hundreds of lens galaxies inside a dark matter halo of more than 10^14 solar masses,
  lensing many background sources at once. Their science is extreme magnification — using the cluster as a
  natural telescope onto the faint early Universe — and mapping the cluster's dark matter distribution.

- **Multi-plane ray tracing**: cluster sources sit at different redshifts, so the lens equation is applied
  recursively through the planes, with each plane's deflections scaled by cosmological distance ratios. The
  `Tracer` handles this automatically, provided every source's redshift is set correctly.

- **Point-source modeling**: fitting the extended light of every source is computationally prohibitive at
  this scale, so the standard workflow fits the positions of each source's multiple images, reducing the
  likelihood to source-plane geometry that evaluates in milliseconds.

- **The CSV interface**: the data (`point_datasets.csv`), the individually-modelled galaxies (`mass.csv` /
  `point.csv`) and the member catalogue (`scaling_galaxies.csv`) are all spreadsheet-editable files. Scaling
  the model to a richer cluster is a row-append, not a code change.

- **Six parameters, fifteen galaxies**: fixed observed centres, a scaling relation for the member population
  and solved source centres decouple the model's physical richness from the dimensionality of the search.

- **Results access**: per-member realised profiles, the multi-plane tracer, solved source-plane centres,
  per-image magnifications and forward-solved image positions — the complete toolkit for interpreting a
  cluster fit.

Strong lensing — multiple images, arcs, Einstein rings — has carried us from single galaxies to the most
massive structures in the Universe. But a cluster's gravity does not stop deflecting light at the radius
where multiple images form. Far beyond it, every background galaxy is still subtly sheared — distorted by a
percent or less — and by measuring those distortions statistically across thousands of galaxies, the
cluster's mass can be mapped out to its edges. That is **weak lensing**, a different regime with different
data and different statistics, and it is where the next tutorial takes us.
"""
