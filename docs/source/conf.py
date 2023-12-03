# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "Macrosynergy Package"
copyright = "2023, Macrosynergy"
author = "Macrosynergy"
title = "Macrosynergy Package Documentation"
# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

autodoc_member_order = "bysource"
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["*/__init__.py"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_book_theme"
html_theme_options = {
    "github_url": "https://github.com/macrosynergy/macrosynergy",
}


html_static_path = ["_static"]
html_css_files = ["style.css"]
source_suffix = [".rst", ".ipynb", ".md"]

# -- Autosummary Settings ----------------------------------------------------
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_imported_members = True  # Document imported members
add_module_names = False  # Do not prepend module names to members

# -- Extension Settings ------------------------------------------------------
# Add any additional extension specific settings here
# always_document_param_types = True
# napoleon_numpy_docstring = True

# set documentation title to "Macrosynergy Package Documentation"
#



nav_order = [
    "index",
    "Context",
    "Installation",
    "DataQuery",
    "Academy",
    "Contribution Guide",
    "Definitions",
]


html_sidebars = {
    "**": ["sidebar-nav-bs"],
}
