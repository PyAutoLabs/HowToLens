r"""
Tutorial 5: Cluster Scale
=========================

Throughout this chapter we have been climbing a ladder of scale: from a single lens galaxy with an extra
galaxy nearby, to multi-galaxy lenses, to galaxy groups whose members share a common dark matter halo.

This tutorial climbs the final rung: **galaxy clusters**, the most massive gravitationally-bound structures in
the Universe. A cluster-scale strong lens contains:

- **30+ to hundreds of lens galaxies**: a brightest cluster galaxy (BCG) and bright satellites, plus a large
  population of lower-mass member galaxies, all embedded in...

- **A cluster-scale dark matter halo of more than 10^14 solar masses** (the most massive exceed 10^15 solar
  masses) — hundreds of times more massive than the halo of a single lens galaxy.

- **Many multiply-imaged background sources** — often tens of them — and, crucially, these sources typically
  all sit at **different redshifts**. Light from each source therefore traverses a different path through the
  Universe, and modeling them together requires **multi-plane ray tracing**, which we explain below.

The science a cluster lens delivers is different from the galaxy-scale lensing of earlier chapters:

- **Extreme magnification**: near a cluster's critical curves, magnifications of tens to hundreds are common
  (compared to the factors of a few of galaxy-scale lensing). Clusters act as natural telescopes, letting us
  study intrinsically faint galaxies in the early Universe that would otherwise be invisible — many of the
  most distant galaxies known were discovered behind lensing clusters.

- **Mapping the cluster's mass**: every multiply-imaged source constrains the cluster's total mass
  distribution along a different line of sight. With many sources, the mass model pins down the shape of the
  cluster's dark matter halo and the substructure within it — the granularity of dark matter on sub-cluster
  scales — providing some of the sharpest tests of the cold dark matter model.

The modeling toolkit also changes at this scale. Fitting the extended arcs of every source, with a light
profile or pixelization each, is computationally prohibitive when there are tens of sources and hundreds of
member galaxies. The standard approach instead fits only the **positions** of each source's multiple images —
point-source modeling — and drives the mass model from spreadsheet-editable **CSV catalogues**, because
writing Python model code for hundreds of galaxies by hand is not sensible either.

__Contents__

- **Multi-Plane Ray Tracing:** What it means for sources at different redshifts to be lensed together.
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

Every lens system in the tutorials so far had two planes: an image plane (the lens galaxies, all at one
redshift) and a source plane (the lensed galaxy). The lens equation was applied once, mapping image-plane
coordinates to source-plane coordinates via the deflection angles:

$\beta = \theta - \alpha(\theta)$

A cluster breaks this picture, because its sources sit at different redshifts. Consider our simulated cluster,
which has lens galaxies at $z = 0.5$ and two sources, at $z = 1.0$ and $z = 2.0$:

- Light from the $z = 1.0$ source is deflected by the $z = 0.5$ cluster and travels to us: the familiar
  two-plane system.

- Light from the $z = 2.0$ source first passes the $z = 1.0$ plane, then the $z = 0.5$ cluster plane. If any
  galaxy at $z = 1.0$ had mass, it would deflect this light too — the deflections of successive planes
  accumulate, each applied at the position the ray has reached by that plane.

This is **multi-plane ray tracing**: the lens equation is applied recursively, plane by plane in redshift
order. There is a second, subtler effect: the strength of a deflection depends on the distances between the
observer, the deflector and the source. The same cluster bends light from a $z = 2.0$ source through a larger
effective angle than light from a $z = 1.0$ source, because the geometry of the light paths differs. The
recursive lens equation accounts for this by scaling each plane's deflections by ratios of cosmological
(angular diameter) distances.

The maths is more bookkeeping than new physics, and the good news is that the `Tracer` handles all of it
automatically: give it galaxies at three or more redshifts and it groups them into planes, orders them, scales
their deflections and applies the recursion. Every source's redshift must simply be *correct* — hardcoding the
wrong redshift silently produces the wrong multi-plane geometry, which is why the dataset CSVs below carry a
redshift for every source.
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

We use a simulated cluster whose scale is kept deliberately small so it runs quickly, but which contains every
ingredient of the cluster regime: 2 main lens galaxies (a BCG and a satellite), 10 lower-mass member galaxies
on a luminosity-mass scaling relation, a 10^15.3 solar-mass dark matter halo, and 2 multiply-imaged sources at
different redshifts (z = 1.0 and z = 2.0). A real cluster simply has more of everything — and we will see that
scaling this model up to hundreds of members does not add a single free parameter.
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
We first load and plot the cluster's CCD imaging. Note the pixel scale of 0.2" and the sheer size of the
field: 100" x 100", compared to the ~6" fields of the galaxy-scale tutorials. The Einstein radius of a
10^15 solar-mass cluster is ~20-30", so its multiple images and arcs are spread across a region hundreds of
times larger in area than a galaxy-scale lens.

The imaging is loaded for visualization only — as we discuss next, the model is not fitted to these pixels.
"""
data = al.Array2D.from_fits(file_path=dataset_path / "data.fits", pixel_scales=0.2)

aplt.plot_array(array=data, title="Cluster CCD Imaging")

"""
The simulator saved the true `Tracer` used to make this dataset, which we can load to inspect the multi-plane
structure described above. The tracer has three planes: the lens plane at z = 0.5 (holding all 13 lens-plane
galaxies — 2 main galaxies, 10 scaling members and the dark matter halo) and one source plane per source.
"""
tracer_true = al.from_json(file_path=dataset_path / "tracer.json")

print(f"Number of planes: {len(tracer_true.planes)}")
print(f"Plane redshifts: {[float(plane.redshift) for plane in tracer_true.planes]}")
print(f"Galaxies in lens plane: {len(tracer_true.planes[0])}")

"""
__Point Source Modeling__

In every previous tutorial we fitted the data pixel-by-pixel: a model image of the lens and source light was
computed, convolved with the PSF and compared to every image pixel. For a cluster this becomes computationally
prohibitive:

- Every one of the tens of sources needs its own light model (or pixelization), each adding parameters (or an
  expensive linear inversion) to the fit.

- The image is enormous — our modest simulated cluster already spans 500 x 500 pixels — and the deflection
  angles of hundreds of member galaxies would need evaluating at every one of them, for every likelihood
  evaluation.

The standard approach in cluster lensing therefore simplifies the data dramatically: each multiply-imaged
source is reduced to the **positions of its multiple images** — the (y,x) coordinates of the brightest pixel
of each image, measured from the imaging. The model then only has to answer a much cheaper question: does this
mass model ray-trace all of a source's observed image positions back to the same source-plane location?

This is **point-source modeling**. We are deliberately discarding the information in the arcs' extended light
in exchange for a likelihood that is fast enough to evaluate with hundreds of galaxies and tens of sources.
(For lensed quasars and supernovae — genuine point sources — nothing is even discarded.) Extended-source
cluster modeling, where individual arcs are fitted at pixel level with the pixelizations of the previous
chapter, does exist in **PyAutoLens** as a specialised follow-up analysis — see the `autolens_workspace`
cluster examples — but position fitting is the workhorse that published cluster mass models are built on.

__Point Datasets__

The positions of each source's multiple images are stored in a single CSV file, `point_datasets.csv`, with
one row per observed multiple image:

 - `name`: the source identifier (e.g. `point_0`). All rows sharing a `name` belong to the same source.
 - `y`, `x`: the image-plane position of the multiple image, in arc-seconds.
 - `positions_noise`: the positional uncertainty in arc-seconds — how precisely the image's centroid is
   measured (for space-telescope imaging ~0.005", set by PSF-fitting precision, not the pixel scale).
 - `redshift`: the source redshift, which every row of a group must share. This column is what makes the
   multi-plane geometry reproducible from the file alone.

This CSV is the recommended input format for cluster datasets because it is hand-editable: measuring your own
cluster's multiple images means filling in a spreadsheet, not writing Python.

`al.list_from_csv` loads the file into a list of `PointDataset` objects, one per source.
"""
dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

for dataset in dataset_list:
    print("Point Dataset Info:")
    print(dataset.info)
    print(f"Redshift: {dataset.redshift}")

"""
Plotting each dataset shows just how sparse point-source data is: a handful of (y,x) coordinates per source.
That a full cluster mass model can be constrained by so few numbers is because each coordinate is measured to
milli-arcsecond precision, and because every source at its own redshift probes the mass distribution
differently.
"""
for dataset in dataset_list:
    aplt.plot_grid(
        grid=al.Grid2DIrregular(np.atleast_2d(dataset.positions)),
        title=dataset.name,
    )

"""
__The CSV Interface__

With hundreds of member galaxies, composing the lens model in Python — one `af.Model(al.Galaxy)` per galaxy,
as in every previous tutorial — stops being sensible. The cluster workflow therefore defines the model in CSV
files as well, which the simulator wrote alongside the data:

 - `mass.csv`: every individually-modelled mass profile — the two main lens galaxies' `dPIEMassSph` profiles
   and the host halo's `NFWMCRLudlowSph`.
 - `light.csv`: the light profiles (not used in point-source modeling, since light does not lens).
 - `point.csv`: the source galaxies' `Point` components.

Each row of these CSVs carries:

 - `galaxy`: a galaxy name (e.g. `lens_0`, `host_halo`, `source_0`). Rows sharing a name compose into one
   galaxy.
 - `attr_name`: the attribute the profile is bound to on the galaxy (e.g. `mass`, `dark`, `point_0`).
 - `profile_class`: the profile class name (e.g. `dPIEMassSph`), looked up in `al.mp` / `al.lp` / `al.ps`.
 - The profile's constructor parameters as columns (tuples like `centre` split into `y` / `x`); parameters a
   row's class does not use are left blank.
 - `redshift`: the galaxy's redshift.

The lower-mass member population lives in a fourth, simpler catalogue, `scaling_galaxies.csv`, with one row
per member and columns `y, x, luminosity`. No profile class, no mass parameters — because, as the previous
tutorial showed, the members' masses are not free parameters but derive from their luminosities via a scaling
relation. The catalogue is exactly what an observer measures: where each member is and how bright it is.

This is the key scaling property of the CSV interface: modeling a richer cluster means appending rows to
`scaling_galaxies.csv` (and, for another individually-modelled galaxy or halo, to `mass.csv`) — the Python
below does not change, and the number of free parameters does not grow with the member count.

We load the mass and point families (`al.galaxy_models_from_csv`) and the member catalogue
(`al.galaxy_table_from_csv`):
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
In a real analysis the member centres and luminosities come from light fits to the imaging data (or an
external photometric catalogue), and the main-galaxy centres in `mass.csv` are pinned to the observed light
centres. Fixing the centres to observed values is essential at cluster scale: a handful of multiple-image
positions cannot constrain hundreds of free centre coordinates, but the imaging measures them for free.

__Point Solver__

Point-source modeling needs one new piece of machinery: given a mass model and a source-plane (y,x)
coordinate, where in the image plane do its multiple images appear? Answering this means solving the lens
equation *backwards*, which has no analytic solution.

The `PointSolver` solves it numerically: it tiles the image plane with triangles, ray-traces each triangle's
corners to the source plane, keeps the triangles that land on the source-plane coordinate, and subdivides
them repeatedly until the image positions are located to a precision of `pixel_scale_precision`.

The solver's grid must span the full 100" cluster field — the multiple images sit tens of arc-seconds from
the cluster centre. The `magnification_threshold` discards heavily demagnified images (e.g. the faint central
image of a typical lens configuration, which real observations do not detect).
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

We now compose the cluster model, which has four tiers of component — the same four tiers every published
cluster mass model is built from:

 - **Main lens galaxies (2)**: the BCG and satellite, modelled individually with `dPIEMassSph` profiles. As
   the previous tutorial on group-scale lenses discussed, the dPIE is the standard truncated profile of this
   regime: members orbiting in the cluster's shared potential have their outer dark matter tidally stripped,
   so their mass genuinely ends at a finite truncation radius `r_cut`. Each main galaxy has free `sigma` (its
   fiducial velocity dispersion, setting the lens strength) and free `r_cut`, with its centre fixed to the
   observed light centre [4 free parameters].

 - **Scaling-tier members (10)**: `dPIEMassSph` profiles with centres fixed from the catalogue and `sigma` /
   `r_cut` derived from each member's luminosity via the scaling relation introduced earlier in this chapter:
   sigma = sigma_ref * (L / L_ref)^0.25 and r_cut = r_cut_ref * (L / L_ref)^0.7. Only the normalization
   `sigma_ref` — the velocity dispersion of a member at the reference luminosity — is free. Whether the
   catalogue holds 10 members or 300, the tier costs [1 free parameter].

 - **Host dark matter halo (1)**: a standalone galaxy carrying the cluster's `NFWMCRLudlowSph` halo, centred
   on the BCG, with a free total mass `mass_at_200` [1 free parameter].

 - **Source galaxies (2)**: parameter-free `PointSolved` components. Rather than sampling each source's
   (y,x) source-plane centre as free parameters, the fit solves for the centre analytically — the
   precision-weighted mean of the back-traced image positions. With tens of sources this matters: it removes
   two parameters per source from the search [0 free parameters].

**Total: N = 6 free parameters** — for a model containing 13 galaxies and 2 sources. This is the punchline of
the cluster workflow: catalogues and scaling relations decouple the model's physical richness from the
dimensionality of the non-linear search.

`al.galaxy_af_models_from_csv_tables` converts the CSV tables into a dictionary of `af.Model(Galaxy)` objects
keyed by galaxy name, with every CSV value fixed as a default. We then selectively promote parameters to free
priors — exactly the `af.Model` customization API used throughout the earlier chapters, just applied to
models built from files instead of Python.
"""
redshift_lens = 0.5
source_redshifts = [dataset.redshift for dataset in dataset_list]

galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

"""
The main lens galaxies get free dPIE `sigma` / `r_cut`; their centres and redshifts stay fixed at the CSV
values, and `r_core` stays fixed at the CSV's 0.0 (the vanishing-core convention discussed in the previous
tutorial). The cosmology constants `H0` / `Om0` are pinned explicitly: they are model *constants*, not
parameters to sample — left unset they would inherit priors from the configuration files and float.
"""
for name in ("lens_0", "lens_1"):
    galaxy_models[name].mass.sigma = af.UniformPrior(
        lower_limit=50.0, upper_limit=600.0
    )
    galaxy_models[name].mass.r_cut = af.UniformPrior(lower_limit=2.0, upper_limit=40.0)
    galaxy_models[name].mass.H0 = 67.66
    galaxy_models[name].mass.Om0 = 0.30966

"""
The host halo gets a free `mass_at_200`, spanning the full range of cluster masses with a log-uniform prior
(mass scales spanning orders of magnitude are sampled in log space, as we saw for intensities in earlier
chapters).
"""
galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
    lower_limit=10**14.5, upper_limit=10**16.0
)

"""
Each source's `point_i` component is swapped for the parameter-free `al.ps.PointSolved`. The name pairing
(`point_0`, `point_1`) is what links each model component to the `PointDataset` with the same name — in a
multi-source cluster this pairing is what guarantees every source's positions are fitted by the correct
component.
"""
for i, dataset in enumerate(dataset_list):
    setattr(galaxy_models[f"source_{i}"], f"point_{i}", af.Model(al.ps.PointSolved))

"""
The scaling tier is composed in a loop over the catalogue, exactly as in the previous tutorial: the single
shared prior `scaling_sigma_ref` is defined once outside the loop, and each member's `sigma` / `r_cut` derive
from it via that member's luminosity ratio. The truth value used by the simulator is sigma_ref = 85 km/s.
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
The named galaxies and the scaling tier combine into the overall model. Printing `model.info` confirms the
parameter count: for all its 15 galaxies, the search explores just 6 dimensions.
"""
model = af.Collection(
    galaxies=af.Collection(**galaxy_models),
    scaling_galaxies=scaling_galaxies,
)

print(model.info)

"""
__Analysis + Factor Graph__

Each source's dataset gets its own `AnalysisPoint`, which defines the log likelihood for that source: how
well do the model's predictions match this source's observed image positions?

`fit_positions_cls=al.FitPositionsSourceSolved` selects the **solved source-plane** likelihood — the default
cluster configuration. Rather than forward-solving the lens equation for the model's image positions on every
likelihood evaluation (accurate but expensive), it ray-traces the *observed* positions back to the source
plane and measures how tightly they converge around the analytically-solved source centre, with a weighting
that maps source-plane scatter back to image-plane precision. No lens-equation solve per evaluation makes
cluster-scale inference fast; the `PointSolver` is reserved for validating the final model below.

The analyses are then combined through a **factor graph**, the same multi-dataset machinery used for
multi-wavelength imaging: each analysis becomes a factor sharing the one global model, and the total log
likelihood is the sum over the per-source likelihoods. This is what lets every source — each at its own
redshift, each with its own multi-plane geometry — constrain the same cluster mass model simultaneously.
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

We fit the model with Nautilus. The parameter space is only 6-dimensional, so a modest number of live points
suffices; a richer cluster with several individually-modelled galaxies would warrant more.
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

We pass the factor graph's global model and the factor graph itself (as the analysis) to the search. Because
the likelihood involves no pixel-level computation, a full cluster point-source fit takes minutes — this is
the payoff of point-source modeling, given that a pixel-level fit of the same system would take days.
"""
print(
    """
    The non-linear search has begun running.

    This Jupyter notebook cell will progress once the search has completed - this could take a few minutes!

    On-the-fly updates every iterations_per_quick_update are printed to the notebook.
    """
)

result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

print("The search has finished run - you may now continue the notebook.")

"""
__Results__

A cluster fit produces the most complex results object we have met: many galaxies across multiple planes,
tiers whose parameters are derived rather than sampled, and one result per source dataset. This section tours
how to pull out each quantity you will actually need — it is worth working through slowly, because navigating
results is half the practical skill of cluster modeling.

A factor-graph fit returns one `Result` per dataset (in the same order as `dataset_list`). All results share
the same global model — and therefore the same samples and maximum likelihood model — but each carries its own
dataset and its own maximum likelihood `FitPointDataset`.
"""
print(f"Number of results (one per source dataset): {len(result_list)}")

result = result_list[0]

"""
__Results: The Maximum Likelihood Instance__

The `max_log_likelihood_instance` is the model instance with the highest likelihood found. Its structure
mirrors the model composition above: named galaxies under `.galaxies`, the scaling tier under
`.scaling_galaxies`. We can read off the best-fit parameters of every tier.
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

The scaling tier's single free parameter was `sigma_ref`, but the instance contains the fully-realised
`dPIEMassSph` of every member — the scaling relation has been applied, so each member carries the `sigma` and
`r_cut` its luminosity implies. This is how you would tabulate the inferred mass of every member galaxy in a
real cluster analysis.
"""
for i, member in enumerate(instance.scaling_galaxies):
    print(
        f"Member {i}: centre={member.mass.centre}, "
        f"sigma={member.mass.sigma:.1f} km/s, r_cut={member.mass.r_cut:.2f} arcsec"
    )

"""
__Results: The Tracer__

`max_log_likelihood_tracer` assembles every galaxy of the instance — named tiers and scaling members alike —
into the best-fit multi-plane `Tracer`. This is the object to use for any lensing calculation with the
best-fit mass model: convergence maps, deflection fields, critical curves and caustics, all computed exactly
as in the earlier chapters, but now for the full cluster.
"""
tracer = result.max_log_likelihood_tracer

print(f"Tracer planes: {[float(plane.redshift) for plane in tracer.planes]}")
print(f"Galaxies in lens plane: {len(tracer.planes[0])}")

aplt.subplot_tracer(tracer=tracer, grid=grid)

"""
__Results: Solved Source-Plane Positions__

Each result's `max_log_likelihood_fit` is the `FitPointDataset` of that source. Because we fitted with the
solved source-plane likelihood, the fit's `positions` object carries the analytically-solved source-plane
centre — the model's inferred true position of the source, before lensing. Its `residual_map` lists how far
each back-traced image lands from that centre (in the source plane): for a good model these residuals are
tiny, since all images of a source originate from the same point.
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
We can perform the same back-tracing manually with the tracer, which makes explicit what the fit just did:
ray-trace each source's observed image positions through the multi-plane lens equation to that source's own
plane. `plane_index_via_redshift_from` maps a source redshift to its plane, and `traced_grid_2d_list_from`
returns the positions traced to every plane — the multi-plane generalisation of the two-plane ray tracing of
chapter 1.
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
 - `autolens_workspace/scripts/guides/point_source_pairing.py`: the full matrix of position-fitting schemes
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
