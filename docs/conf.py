# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path so autodoc can import redi
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "REDI"
copyright = "2024, Wes Brewer"
author = "Wes Brewer"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # Google/NumPy docstring styles
    "sphinx.ext.viewcode",       # [source] links
    "sphinx_autodoc_typehints",
    "sphinxarg.ext",             # sphinx-argparse
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Project-level markdown files picked up by myst-parser; not part of the docs
    "CLAUDE_PROJECT_INSTRUCTIONS.md",
    "DEVELOPMENT_STATUS.md",
    "TECHNICAL_OVERVIEW.md",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

# -- autodoc options ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# autodoc_mock_imports prevents import errors for heavy optional deps and broken packages.
# pandas and numpy are listed here because the system pyarrow version conflicts with them.
autodoc_mock_imports = [
    "pandas",
    "numpy",
    "xarray",
    "xesmf",
    "netCDF4",
    "zarr",
    "pymatgen",
    "matminer",
    "biopython",
    "Bio",
    "h5py",
    "h3",
    "geopandas",
    "shapely",
    "torch",
    "torch_geometric",
    "pyarrow",
    "sklearn",
    "scikit_learn",
    "flowcept",
    "esmpy",
    "cdsapi",
    "yaml",
]

# -- Napoleon (Google / NumPy docstrings) ------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- sphinx-autodoc-typehints ------------------------------------------------
always_document_param_types = True
typehints_fully_qualified = False

# -- MyST (Markdown support) -------------------------------------------------
myst_enable_extensions = ["colon_fence"]


def setup(app):
    # Workaround: Sphinx 9.x has both a `mock` function (sphinx.ext.autodoc.__init__)
    # and a `mock` submodule (sphinx.ext.autodoc.mock). When the submodule is first
    # imported during autodoc processing, it registers in sys.modules as
    # "sphinx.ext.autodoc.mock" which causes subsequent `from sphinx.ext.autodoc import mock`
    # to return the MODULE instead of the function.  sphinxarg 0.5.x relies on that import
    # for its `with mock(...)` call, so we restore the correct reference in its namespace.
    def _fix_sphinxarg_mock(app, env, docnames):
        from sphinx.ext.autodoc._dynamic._mock import mock as _real_mock
        import sphinxarg.ext as _sphinxarg_ext
        if not callable(getattr(_sphinxarg_ext, "mock", None)):
            _sphinxarg_ext.mock = _real_mock

    app.connect("env-before-read-docs", _fix_sphinxarg_mock)
