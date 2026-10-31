Look at a resolved lensed arc. Its broad shape may be smooth, but the light along it can contain bright knots, faint extensions and irregular structure. A model might reproduce the overall arc while leaving compact, coherent features in the residual image.

In Chapter 2, we introduced linear light profiles and multi-Gaussian expansions (MGEs). By combining Gaussians of different widths, an MGE can describe a much more flexible radial brightness distribution than a single Sérsic profile. Gaussian groups with different ellipticities and orientations can also describe changes in flattening and isophotal twists. Fixed widths and linearly solved intensities keep the number of non-linear parameters manageable.

That flexibility makes MGE a useful modelling tool, but consider the geometry of our worked example. Within each galaxy’s Gaussian group, the components shared a common centre and elliptical shape. Adding more Gaussians gives us finer control over how brightness changes around that centre. It does not give us separate centres for star-forming knots, off-centre clumps or several distinct source components.

We could add offset Gaussian groups. MGEs are not fundamentally restricted to one centre. The practical difficulty is deciding how many groups to use, where to place them and how to fit their geometry as the source becomes more complicated.

This is where the image residuals become useful. If an MGE captures the broad light distribution but leaves compact structure along the arcs, its source description may need more spatial freedom. Those residuals are not proof of source clumpiness: errors in the mass model, foreground light subtraction, telescope blurring or other aspects of the data treatment can also produce structure. We need to investigate their origin.

In this chapter, we introduce pixelized source reconstruction. Instead of prescribing a light-profile centre and shape for every feature, we describe the source brightness on a mesh. Different regions of that mesh can take different brightness values, allowing the reconstruction to represent more complicated source structure.

The connection to Chapter 2 is direct. There, we solved for Gaussian intensities linearly. Here, for a specified mass model and mesh, we solve for source brightness values using the same underlying advantage: they enter the predicted image linearly.

This additional freedom also allows the reconstruction to fit noise. We therefore introduce regularization—a prior that controls how freely neighbouring source brightnesses can vary. Pixelization is not assumption-free: the mass model, mesh geometry, resolution and regularization all affect the inferred source.

MGE remains valuable, especially for foreground lens light and initial source fits. Pixelization is the next tool when the source requires more spatial freedom. We will learn how to use these tools together, inspect their reconstructions and recognise when their assumptions need revisiting.

- [Pixelizations, Mappers and Inversions](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_1_pixelizations.ipynb)
- [Bayesian Regularization](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_2_bayesian_regularization.ipynb)
- [Borders](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_3_borders.ipynb)
- [Lens Modeling](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_4_lens_modeling.ipynb)

The following tutorials explore adaptive meshes and regularization:

- [Adaptive Pixelization](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_8_adaptive_pixelization.ipynb)
- [Brightness Adaption](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_11_brightness_adaption.ipynb)
- [Adaptive Regularization](https://colab.research.google.com/github/PyAutoLabs/HowToLens/blob/main/notebooks/chapter_3_pixelizations/tutorial_12_adaptive_regularization.ipynb)
