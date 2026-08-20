The `howtolens` folder contains **HowToLens** lectures, which teach a new user what strong lensing is and how to model
a strong lens.

# Folders

- `chapter_1_introduction`: An introduction to strong gravitational lensing and **PyAutolens**.
- `chapter_2_lens_modeling`: How to model strong lenses, including a primer on Bayesian non-linear analysis
  and non-linear search chaining.
- `chapter_3_pixelizations`: How to perform pixelized reconstructions of the source-galaxy.
- `chapter_4_scaling_up_lensing`: How to scale lens modeling up to extra galaxies, multi-galaxy lenses,
  scaling relations, groups, clusters and weak lensing.
- `chapter_optional`: Optional tutorials.
- `simulator`: Scripts used to simulate the strong lens datasets fitted throughout the **HowToLens** lectures.

# Full Explanation

Welcome to **HowToLens** - The **PyAutoLens** tutorial!

# JUYPTER NOTEBOOKS

All tutorials are supplied as Jupyter Notebooks, which come with a '.ipynb' suffix. For those new to Python, Jupyter
Notebooks are a different way to write, view and use Python code. Compared to the traditional Python scripts,
they allow:

- Small blocks of code to be viewed and run at a time.
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the **HowToLens** lecture series, therefore I recommend you get yourself
a Jupyter notebook viewer (<https://jupyter.org/>) if you have not done so already.

If you *really* want to use Python scripts, all tutorials are supplied a \`\` python files in the 'scripts' folder of
each chapter.

For actual **PyAutoLens** use, I recommend you use Python scripts. Therefore, as you go through the lecture series
you will notice that we will transition you to Python scripts in the third chapter.

# LENSING THEORY

HowToLens assumes minimal previous knowledge of gravitational lensing and astronomy. However, it is beneficial to give
yourself a basic theoretical grounding as you go through the lectures. I heartily recommend you have open the
lecture course on gravitational lensing by Massimo Meneghetti below as you go through the tutorials, and refer to it
for anything that isn't clear in HowToLens.

<http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf>

# VISUALIZATION

Before beginning the **HowToLens** lecture series, in chapter 1 you should do 'tutorial_0_visualization'. This will
take you through how **PyAutoLens** interfaces with matplotlib to perform visualization and will get you setup such that
images and figures display correctly in your Jupyter notebooks.

# CODE STYLE AND FORMATTING

When you begin the notebooks, you may notice the style and formatting of our Python code looks different to what you
are used to. For example, it is common for brackets to be placed on their own line at the end of function calls,
the inputs of a function or class may be listed over many separate lines and the code in general takes up a lot more
space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact - lots of people do,
which is why we use an auto-formatter to produce the code in a standardized format. If you're interested in the style
and would like to adapt it to your own code, check out the Python auto-code formatter 'black'.

<https://github.com/python/black>

# HOW TO TACKLE HowToLens

The **HowToLens** lecture series currently sits at 4 chapters, and each will take a day or so to go through
properly. You probably want to be modeling lenses faster than that! Furthermore, the concepts in the
later chapters are pretty challenging, and familiarity with lens modeling is desirable before you
tackle them.

Therefore, we recommend that you complete chapters 1 & 2 and then apply what you've learnt to the modeling of simulated
and real strong lens data, using the scripts found in the 'autolens_workspace'. Once you're happy
with the results and confident with your use of **PyAutoLens**, you can then begin to cover the advanced functionality
covered in chapters 3 & 4.

# OVERVIEW OF CHAPTER 1 (Beginner)

**Strong Lensing with PyAutoLens**

In chapter 1, we'll learn about strong gravitational lensing and **PyAutoLens**. At the end, you'll
be able to:

1. Create uniform grid's of (x,y) Cartesian coordinates.
2. Combine these grid's with light and mass profiles to make images, convergence maps, gravitational potentials and deflection angle-maps.
3. Combine these light and mass profiles to make galaxies.
4. Perform ray-tracing with these galaxy's whereby a grid is ray-traced through an image-plane / source-plane strong lensing configuration.
5. Simulate telescope CCD imaging data of a strong gravitational lens.
6. Fit strong lensing data with model images generated via ray-tracing.

# OVERVIEW OF CHAPTER 2 (Beginner)

**Bayesian Inference and Non-linear Searches**

In chapter 2, we'll cover Bayesian inference and model-fitting via a non-linear search. We will use these tools to
fit CCD imaging data of a strong gravitational lens with a lens model. At the end, you'll understand:

1. The concept of a non-linear search and non-linear parameter space.
2. How to fit a lens model to strong lens CCD imaging via a non-linear search.
3. The trade-off between realism and complexity when choosing a lens model.
4. Why an incorrect lens model may be inferred and how to prevent this from happening.
5. The challenges that are involved in inferring a robust lens model in a computationally reasonable run-time.
6. How to chain non-linear searches together to build automated lens modeling pipelines, including the
   Source, Light and Mass (SLaM) pipelines.

**Once completed, you'll be ready to model your own strong gravitational lenses with PyAutoLens!**

# OVERVIEW OF CHAPTER 3 (Intermediate)

**Using an inversion to perform a pixelized source reconstruction**

In chapter 3, we'll learn how to reconstruct the lensed source galaxy using a pixel-grid, ensuring that we can fit an
accurate lens model to sources with complex and irregular morphologies. You'll learn how to:

1. Pixelize a source-plane into a set of source-plane pixels defined by mappings to image pixels.
2. Perform a linear inversion on this source-plane pixelization to reconstruct the source's light.
3. Apply a smoothness prior on the source reconstruction, called regularization.
4. Apply smoothing within a Bayesian framework to objectively quantify the source reconstruction's complexity.
5. Write down the linear algebra and Bayesian evidence equations that underpin the whole framework.
6. Define a border in the source-plane to prevent pixels tracing outside the source reconstruction.
7. Use alternative pixelizations, for example a Voronoi mesh whose pixels adapt to the lens's mass model
   or to the source's own brightness, alongside adaptive regularization schemes.
8. Use these features to fit a lens model via non-linear searches.

# OVERVIEW OF CHAPTER 4 (Advanced)

**Scaling Up Lensing**

In chapter 4, we'll scale lens modeling up beyond a single lens galaxy, learning how to:

1. Handle extra galaxies near a lens, by scaling their light out of the fit or modeling them explicitly.
2. Model systems with two or more co-dominant lens galaxies and understand the degeneracies this creates.
3. Use scaling relations to tie galaxy masses to their luminosities, so model complexity stops growing
   with every galaxy.
4. Model group-scale lenses, using truncated dPIE mass profiles and an optional group dark matter halo.
5. Model cluster-scale lenses, using point-source positions, a CSV interface for member catalogues and
   multi-plane ray tracing.
6. Fit weak-lensing shear catalogues, which probe dark matter far beyond the strong lensing region.
