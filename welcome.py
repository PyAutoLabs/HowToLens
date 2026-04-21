import autolens as al
from autoconf import check_version

check_version(al.__version__)

input(
    "########################################\n"
    "### WELCOME TO HOWTOLENS             ###\n"
    "########################################\n\n"
    "This script runs a few checks to ensure PyAutoLens is set up correctly.\n"
    ""
    "Once they pass, read through start_here.py (or start_here.ipynb) and then open\n"
    "scripts/chapter_1_introduction/tutorial_0_visualization.py to begin the tutorial series.\n\n"
    "\n"
    "########################################\n"
    "### HOWTOLENS WORKING DIRECTORY       ###\n"
    "########################################\n\n"
    """
    PyAutoLens assumes that the `HowToLens` directory is the Python working directory.
    This means that, when you run a tutorial script, you should run it from the `HowToLens`
    repository root as follows:


    cd path/to/HowToLens (if you are not already in HowToLens).
    python3 scripts/chapter_1_introduction/tutorial_1_grids_and_galaxies.py


    The reasons for this are so that PyAutoLens can:

    - Load configuration settings from config files in the `HowToLens/config` folder.
    - Write simulated tutorial data to `HowToLens/dataset/`.
    - Output the results of model fits to `HowToLens/output/`.

    If you get errors relating to importing modules, loading data or writing output, it is
    most likely because you are not running the script with `HowToLens` as the working
    directory.

    [Press Enter to continue]"""
)

input(
    "########################################\n"
    "### MATPLOTLIB BACKEND                 ###\n"
    "########################################\n\n"
    """
    Figures produced by the tutorials are rendered with `matplotlib`. Depending on your
    system, you may need to configure the matplotlib backend. `tutorial_0_visualization.py`
    in `chapter_1_introduction` walks you through the options.

    [Press Enter to finish]"""
)
