"""Sphinx configuration for the quantum_al documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import quantum_al  # noqa: E402

project = "quantum_al"
author = "Arnav Kapoor"
copyright = "2025-2026, Arnav Kapoor"
release = quantum_al.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
# Optional heavy dependencies are mocked so the API docs build without them.
autodoc_mock_imports = ["qiskit", "qiskit_aer", "pymatgen"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = []
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = f"quantum_al {release}"
