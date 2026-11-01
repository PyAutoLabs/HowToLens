"""
Tutorial 2: Practicalities
==========================

In the previous tutorial, we fitted a simple lens model using Nautilus. We now move to a more realistic image,
containing both the foreground lens galaxy’s light and the lensed source.

Fitting this image requires additional model components and more free parameters. Alongside building that model, we
will learn how to configure the search, follow its progress, find its saved output and inspect the resulting fit.

The overall approach remains the same: the dataset holds the observations, the model describes the physical components,
and the Analysis object connects them. Nautilus explores the model parameters using the likelihood and priors.

By the end, you should be comfortable running a fit and finding the information needed to assess it. The next tutorial
examines why more complex models can be difficult to fit and what to do when a search struggles.

__Contents__

- **Dataset**
- **Mask**
- **Model**
- **Analysis**
- **Search**
- **Search Settings**
- **Output Updates**
- **VRAM Use**
- **Run Times**
- **Notebooks and Scripts**
- **Model Fit**
- **Result Info**
- **Output Folder**
- **Unique Identifier**
- **Resuming and Loading**
- **Output Folder Contents**
- **Parameter Uncertainties**
- **Plot Labels**
- **Configuration**
- **Further Results**
- **Wrap Up**
"""

"""
__Dataset__

Our new simulated image contains a bright foreground lens galaxy as well as the lensed source.

The simulation uses a Sérsic light profile for the lens galaxy, an elliptical isothermal mass profile, external shear
and an exponential light profile for the source. We will build a model with these components shortly.

As before, the dataset includes the image, its noise map and the telescope’s point-spread function (PSF). If the files
are missing, the following cell runs the simulator before loading them.

In the plotted image, look for the central lens galaxy and the extended lensed emission around it. Both contribute to
the measured light, so our model must account for both.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

# from autolens import setup_notebook; setup_notebook()

from pathlib import Path
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run([sys.executable, "scripts/simulator/lens_sersic.py"], check=True)

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
)

aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Mask__

We use a circular mask with a radius of 2.5 arcseconds to select the central region for this fit.

Inspect the masked image: the selected region should contain the lensed emission and enough of the lens galaxy’s light
to constrain its profile. The mask changes which measurements contribute to the fit; it does not change the physical
model.

Later in this chapter, *Data and Likelihoods* examines mask choices and how to handle contaminating objects.
"""

mask_radius = 2.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
aplt.subplot_imaging_dataset(dataset=dataset)

"""
__Model__

The previous model contained a spherical lens mass profile and a spherical source-light profile, with no foreground
lens light. We now include four components:

- A **Sérsic light profile** to describe the foreground lens galaxy.
- An **elliptical isothermal mass profile** to describe its gravitational lensing.
- An **external shear field** to describe an additional tidal contribution to the lensing.
- An **exponential light profile** to describe the source galaxy.

The lens galaxy’s light and mass are separate components. Fitting its brightness does not, by itself, determine its
mass distribution.

These additions let the model describe more structure in the image, but introduce more free parameters for the search
to explore. This is a recurring balance in modeling: we need enough flexibility to explain the data, while recognising
that a more complex model can be harder to fit.

We use the model construction from the previous tutorial, adding the lens-light profile and placing external shear in a
separate `fields` component.
"""

"""
We construct the model using the same approach as Tutorial 1. Passing a profile class into `af.Model` makes its
parameters available for fitting; supplying a value, such as a galaxy’s redshift, keeps that quantity fixed.

The lens galaxy contains both a light profile and a mass profile. The external shear is a separate field at the lens
redshift, and the source galaxy contains its own light profile.

Here, we use ordinary light profiles from `al.lp`, so their intensities are also free parameters explored by Nautilus.
Tutorial 1 used a linear source-light profile, whose intensity was solved separately. In *Linear Profiles*, we will
return to that approach and see how it simplifies more complex models.
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

"""
Print `model.info` and examine the model diagram below.

Find the lens-light, lens-mass, source-light and shear components. Notice how each contributes free parameters, while
the galaxy and field redshifts remain fixed.

The parameter count tells us how many quantities the non-linear search must explore. More parameters generally make
exploration harder, but their relationships matter too: different combinations can produce similar images.

These displays describe the model we are about to fit. They do not yet tell us whether it explains the data.
"""

print(model.info)

af.ModelPlotter(model).figure()

"""
__Analysis__

We again use `AnalysisImaging` to connect the model to the dataset.

For each proposed model, the analysis predicts the combined lens and lensed-source light, accounts for PSF blurring and
evaluates the log likelihood using the selected image pixels and their noise values.

Although our model has become more complex, the interface is unchanged: the analysis receives a model instance and
returns its log likelihood.
"""

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Search__

We configure Nautilus with a location for its saved output as well as settings controlling the search.

- `path_prefix` groups the results under `output/howtolens/chapter_2`.
- `name` identifies this tutorial’s fit.
- `unique_tag` distinguishes the dataset being fitted.

Together, these organise our results so that fits to different datasets do not become mixed together. PyAutoFit also
adds an automatically generated identifier, explained in the **Unique Identifier** section below.
"""

search = af.Nautilus(
    path_prefix=Path("howtolens") / "chapter_2",
    name="tutorial_2_practicalities",
    unique_tag=dataset_name,
    n_live=200,
    n_batch=50,
    iterations_per_quick_update=2500,
)

"""
__Search Settings__

We increase `n_live` from 100 in Tutorial 1 to 200 for this more complex model. More live points generally allow a more
thorough exploration, at the cost of additional computation. We must still inspect the result rather than assume this
setting guarantees a successful fit.

`n_batch=50` allows likelihood evaluations to be processed in batches. Evaluating several models together can improve
performance, particularly on a GPU, but also requires more memory.

These settings serve different purposes: `n_live` controls the live-point population used to explore parameter space,
whereas `n_batch` controls how evaluations are grouped for computation.

Keep these values for the initial run. The following **VRAM Use** section explains how to check memory requirements and
when to reduce the batch size.
"""

"""
__Output Updates__

While the search runs, PyAutoLens can save updated figures showing the best-fitting model found so far. These let you
follow its progress without waiting for the fit to finish.

We set `iterations_per_quick_update=2500` to request a quick update approximately every 2,500 likelihood evaluations.
This counts evaluated models, not seconds.

Producing figures takes time. If updates occur almost continuously, increasing this value lets the search spend more
time exploring models between updates. Decreasing it gives more frequent views of progress.

Quick updates are intended for monitoring. More extensive output is controlled separately by
`iterations_per_full_update`, and final results are produced when the search finishes.

Keep the current setting for the first run. If you experiment with a different update frequency, restore it to 2,500
afterwards.
"""

"""
__VRAM Use__

If you run the fit on a GPU, its calculations must fit within the GPU’s memory, usually called **VRAM**. Insufficient
memory can cause an out-of-memory error during compilation or evaluation.

Memory requirements depend on the dataset, the model and how many models are evaluated together. Larger images and more
demanding calculations can require more memory. Increasing Nautilus’s `n_batch` also increases the amount of work held
in memory at once.

The following call estimates the memory used by the batched likelihood calculation. `search.batch_size` reports the
batch size set through Nautilus’s `n_batch` argument.

Compare the estimate with the GPU memory available, leaving room for other allocations and applications. This is an
estimate for the likelihood calculation, rather than a guarantee of the total memory used by the process.

If memory is insufficient, try reducing `n_batch` from 50 to 10 and recreate the search before checking again. This
keeps the same model and live-point setting, but evaluates fewer models together.

The estimate itself requires some preparation and may take time. Once you understand the requirements of an unchanged
setup, you can skip this diagnostic on subsequent runs. On a CPU, the relevant limit is system memory rather than GPU
VRAM.
"""

analysis.print_vram_use(
    model=model,
    batch_size=search.batch_size,
)

"""
__Run Times__

Two main factors determine how much computation a fit requires:

- How expensive it is to predict the data and calculate the likelihood for a proposed model.
- How many models the search must evaluate before it finishes.

Adding lens light and allowing elliptical profiles increases the work involved in each prediction. The additional free
parameters can also make parameter space harder to explore, requiring more evaluations.

With JAX, the initial compilation takes time before repeated evaluations become faster. On a GPU, several evaluations
can run together, so multiplying a single-model evaluation time by the number of models does not necessarily predict
the total run time.

Saving figures and results adds further time. The elapsed time therefore includes compilation, searching and output—not
just the likelihood calculation.

Watch the progress of your first run to develop an expectation for your computer. The optional *Need for Speed*
tutorial explains how to investigate performance in more detail.
"""

"""
__Notebooks and Scripts__

In a notebook, the cell containing `search.fit` remains busy until the search finishes. You can inspect files already
written to the output folder while it runs, but later cells in the same notebook kernel must wait.

For a longer fit, you may prefer to run the corresponding Python script from the HowToLens repository root:

```bash
python3 scripts/chapter_2_lens_modeling/tutorial_2_practicalities.py
```

The notebook and script perform the same analysis, and both save results using the output settings we supplied. Running
a script does not inherently make the search faster; it lets the fit run separately from your interactive notebook.

Notebooks remain useful for inspecting data, plotting results and exploring their interpretation. Scripts are
convenient for unattended runs and computing clusters.

Once the fit has completed, rerunning the same setup can load its saved result. We will explain how PyAutoFit
recognises that setup in **Unique Identifier**.
"""

"""
__Model Fit__

We now pass the model and analysis to Nautilus and begin the fit.

Expect an initial compilation stage, followed by search progress and periodic output updates. Use those updates to
follow how the predicted image improves, but wait for completion before interpreting the final parameter uncertainties.

The output is organised beneath `output/howtolens/chapter_2/lens_sersic/tutorial_2_practicalities`. The dataset tag and
automatically generated identifier distinguish this particular fit within that folder.
"""

result = search.fit(model=model, analysis=analysis)

"""
__Result Info__

Once the search finishes, print `result.info` to inspect the inferred parameter values and their uncertainties.

Compare the lens-light, mass, shear and source-light parameters with the model we defined earlier. Then inspect the fit
image and residuals, as in Tutorial 1. Sensible-looking parameter values alone do not establish that the model explains
the data.

The same fit summary is saved as `model.results`, so you can read it without rerunning the notebook.
"""

print(result.info)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
__Output Folder__

Open the `output` directory inside your HowToLens repository. Our search settings organise this fit beneath:

```text
output/howtolens/chapter_2/lens_sersic/tutorial_2_practicalities/
```

Inside is a folder with a long identifier as its name. This contains the saved model, results, figures and information
needed to load or resume the fit.

The tree below shows the main output structure. Some products appear only after completion or when enabled in the
output and plotting configuration.

```text
output/
`-- howtolens/
    `-- chapter_2/
        `-- lens_sersic/                   # unique_tag: the dataset name
            `-- tutorial_2_practicalities/ # name: this analysis
                `-- <identifier>/
                    |-- model.info         # Model components and priors
                    |-- model.results      # Inferred parameters and uncertainties
                    |-- search.summary     # Search progress and timing summary
                    |-- search.log         # Messages written during the run
                    |
                    |-- image/
                    |   |-- dataset.fits   # Saved imaging data, noise map and PSF
                    |   |-- dataset.png    # Dataset visualisation
                    |   |-- fit.fits       # Model and residual-related arrays
                    |   |-- fit.png        # Visual comparison of model and data
                    |   `-- ...            # Additional configured figures and FITS
                    |
                    `-- files/
                        |-- model.json            # Saved model definition
                        |-- tracer.json           # Highest-likelihood tracer
                        |-- search.json           # Saved search configuration
                        |-- samples.csv           # Parameter samples and weights
                        |-- samples_info.json     # Information about the samples
                        |-- samples_summary.json  # Summary of inferred parameters
                        |-- covariance.csv        # Parameter covariance matrix
                        |-- dataset_identity.json # Record of the input dataset
                        `-- search_internal/      # Search checkpoints, when retained
```
"""

"""
__Unique Identifier__

The identifier looks like a random string, but it is calculated from the model, the identifying search settings and the
`unique_tag`.

Rerunning an unchanged setup produces the same identifier and points PyAutoFit to the existing output. Changing the
model or a search setting that contributes to the identifier produces a different identifier.

The `unique_tag` lets us distinguish datasets fitted with the same model and search. Here, we set it to `dataset_name`,
giving the tag `lens_sersic`.

Do not assume that editing a data file, changing a mask or changing every analysis option automatically creates a new
identifier. When deliberately fitting a revised dataset or analysis, use a descriptive new tag or search name to keep
its results separate.

Display settings, such as how often progress figures are produced, need not define a scientifically different fit and
do not necessarily change the identifier.
"""

"""
__Resuming and Loading__

If a Nautilus run is interrupted, rerun the same setup with its output and checkpoint files intact. The search can
resume from its saved state, rather than starting from scratch. Work performed since the latest checkpoint may need to
be repeated.

If the fit has already completed, rerunning the same setup loads the saved result instead of repeating the search. This
is useful when reopening a notebook to inspect or plot a completed analysis.

Keep the output files together. The human-readable summaries are useful for inspection, but they do not replace the
saved samples and other files used to reconstruct results.

The `search_internal` directory contains search-specific state. This tutorial’s output configuration removes it after
completion to save space; that does not prevent the completed result from being loaded.
"""

"""
__Output Folder Contents__

Begin with `image/fit.png`. Inspect whether the predicted image reproduces the lens and source light, and whether
coherent structures remain in the residuals.

Then open `model.results` for the inferred parameters and uncertainties. Use `model.info` alongside it to remind
yourself which components, fixed values and priors were specified.

`search.summary` and `search.log` help you understand the run’s progress and timing. While the search is still running,
treat saved fits and summaries as provisional.

The `files` directory supports more detailed analysis in Python. You do not need to manipulate these files to continue
HowToLens; later we will point you to the workspace examples for loading and analysing completed results.

Output is configurable. `config/output.yaml` controls saved products such as samples and retained search state, while
files under `config/visualize/` control plotting. Keep the tutorial defaults while learning the workflow.
"""

"""
__Parameter Uncertainties__

A corner plot shows the parameter uncertainties inferred by the search and the relationships between them.

The plots along the diagonal show the distribution of each parameter individually. The other panels show pairs of
parameters: an elongated or curved distribution indicates that changing one parameter can be compensated by changing
another.

For example, a galaxy’s light-profile intensity and effective radius can trade off against one another, producing
similar images over the fitted region.

Compare these distributions with the uncertainties in `result.info`. A narrow distribution indicates a tightly
constrained parameter under the assumptions of this fit; it does not account for every possible source of error, such
as an inappropriate model.

**🤖 Ask AI:** My lens model has small image residuals, but two parameters form a long curved distribution in the corner
plot. What does this tell me about what the data constrain?
"""

aplt.corner_anesthetic(samples=result.samples)

r"""
__Plot Labels__

Corner plots use abbreviated parameter names to keep their labels readable. For example, the Sérsic index may be
labelled \(n\). These labels can be customised in `config/notation.yaml`.

Superscripts distinguish components using their names in the model, such as `lens.mass` and `source.bulge`. Use these
alongside `model.info` to identify which parameter each panel represents.

The plotting configuration can also produce separate corner plots for individual components, which are easier to
inspect when the full model contains many parameters.

Some quantities are easier to interpret after conversion. For example, elliptical components can be expressed as an
axis ratio and position angle. The following plot displays supported derived quantities using `latent=True`; it
interprets the same fit rather than running another search.
"""

aplt.corner_anesthetic(samples=result.samples, latent=True)

"""
__Configuration__

The `config` folder controls defaults used throughout HowToLens. You do not need to customise everything before fitting
your own data, but it helps to know where the main settings live:

- `config/priors/` supplies default priors for model components. Assigning priors explicitly in your script makes those choices visible beside the model.
- `config/output.yaml` controls saved products, including sample tables and whether internal search files are retained after completion.
- `config/visualize/` controls which figures are produced. Search plots, including corner plots, are configured in `plots_search.yaml`.
- `config/notation.yaml` controls parameter labels used in plots.

Keep the tutorial defaults initially. Saving fewer products can reduce disk usage, but may limit what you can calculate
later without rerunning the fit.

In particular, a saved summary is not a substitute for the full sample table when you want to perform new calculations
using the parameter distributions.
"""

"""
__Further Results__

You now know how to inspect a fit, read its parameter estimates and find its saved output. That is enough to continue
HowToLens.

The results also support more detailed work: loading fitted galaxies and tracers, calculating derived quantities,
making customised figures and comparing many fits.

Once you are comfortable building models and obtaining satisfactory fits to real data, explore the `autolens_workspace`
results guide, beginning with `scripts/guides/results/start_here.py` or its notebook counterpart.

HowToLens focuses on learning the analysis tools needed to model lenses. The workspace provides the next steps for
using and manipulating the results of those analyses.
"""

"""
__Wrap Up__

We have fitted a model containing lens light, lens mass, external shear and source light, while retaining the same
data–model–Analysis–search structure introduced in Tutorial 1.

You have also learned how to configure search and output settings, check GPU memory requirements, run a notebook or
script, and navigate the saved results. The unique identifier allows an unchanged fit to reuse its output, while
descriptive names and dataset tags help keep different analyses organised.

These practical tools make it easier to run and inspect a fit, but they do not guarantee that the search finds a
satisfactory solution.

In the next tutorial, *Why Modeling Is Hard*, we examine what can go wrong as models become more complex, how to
recognise the problem and which changes can help.
"""
