"""Sphinx configuration for the D-ACE Read the Docs site."""

project = "D-ACE"
author = "Dependable Intelligent Systems Lab"
copyright = "2026, Dependable Intelligent Systems Lab"

extensions = [
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
root_doc = "README"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = "D-ACE Documentation"
html_static_path = ["_static"]
html_css_files = ["ai-sidebar.css"]
html_js_files = ["ai-sidebar.js"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
