from __future__ import annotations

import os
import sys

repo_root = os.environ.get("STREAMLINE_DOCS_REPO_ROOT", os.path.abspath("../.."))
sys.path.insert(0, repo_root)

project = "STREAMLINE"
copyright = "2026, Ryan Urbanowicz, Harsh Bandhey"
author = "Ryan Urbanowicz, Harsh Bandhey"
release = "3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = [
    "catboost",
    "dask",
    "dask_jobqueue",
    "fpdf",
    "gplearn",
    "graphviz",
    "lightgbm",
    "matplotlib",
    "mlxtend",
    "optuna",
    "plotly",
    "seaborn",
    "skrebate",
    "tqdm",
    "xgboost",
]

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except Exception:
    html_theme = "alabaster"

html_static_path = []
html_logo = "pictures/STREAMLINE_Logo_NoText.png"
html_title = "STREAMLINE"
