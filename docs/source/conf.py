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
    "sphinx_autodoc_typehints",
    # "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
    # "sphinx_changelog",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

autodoc_member_order = "bysource"
numpydoc_show_class_members = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
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
    "show_toc_level": 5,
    "header_links_before_dropdown": 4,
    "content_footer_items": ["last-updated"],
    "github_url": "https://github.com/macrosynergy/macrosynergy",
    "logo": {
        "image_light": "MACROSYNERGY_Logo_Primary.png",
        "image_dark": "MACROSYNERGY_Logo_White.png",
    },
    "favicons": [
        {
            "rel": "icon",
            "sizes": "300x300",
            "href": "https://macrosynergy.com/wp-content/uploads/2024/02/macrosynergy-logo-favicon-300x300.png",
        },
        {
            "rel": "apple-touch-icon",
            "sizes": "300x300",
            "href": "https://macrosynergy.com/wp-content/uploads/2024/02/macrosynergy-logo-favicon-300x300.png",
        },
    ],
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://twitter.com/macro_synergy",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/macrosynergy/",
            "icon": "fa-custom fa-pypi",
        },
    ],
}


html_static_path = ["_static"]
html_css_files = [
    "style.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
]
html_js_files = ["custom-icon.js"]
source_suffix = [".rst", ".ipynb", ".md"]

# -- Autosummary Settings ----------------------------------------------------
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autosummary_imported_members = True  # Document imported members
add_module_names = False  # Do not prepend module names to members

# -- Extension Settings ------------------------------------------------------
# Add any additional extension specific settings here
# always_document_param_types = True
# napoleon_numpy_docstring = True


html_sidebars = {
    "**": ["sidebar-nav-bs"],
}
