import importlib
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from docutils import nodes
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.ext.autosummary import generate
from sphinx.ext.autosummary.generate import AutosummaryEntry

project = "Structured Data Models"
author = "NVIDIA"
copyright = f"{date.today().year}, NVIDIA"  # noqa: A001

try:
    release = package_version("structured-data-models")
except PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
html_theme = "shibuya"
html_title = project
html_logo = "images/nvidia.svg"
html_theme_options = {
    "accent_color": "green",
    "github_url": "https://github.com/NVIDIA/structured-data-models",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
autosummary_generate = True
autosummary_context = {"import_module": importlib.import_module}
autodoc_member_order = "bysource"
autodoc_typehints = "both"
suppress_warnings = ["config.cache"]
intersphinx_mapping = {
    "cudf": ("https://docs.rapids.ai/api/cudf/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
    "typing_extensions": (
        "https://typing-extensions.readthedocs.io/en/latest",
        None,
    ),
}
doctest_global_setup = """
from unittest.mock import patch

import sdm.models


def _skip_pretrained(self):
    return self


_pretrained_patchers = []
for _model_class in vars(sdm.models).values():
    if not isinstance(_model_class, type):
        continue
    if not hasattr(_model_class, "_load_from_pretrained"):
        continue

    _patcher = patch.object(
        _model_class,
        "_load_from_pretrained",
        _skip_pretrained,
    )
    _patcher.start()
    _pretrained_patchers.append(_patcher)
"""
doctest_global_cleanup = """
for _patcher in reversed(_pretrained_patchers):
    _patcher.stop()
"""


def api_names(module_name: str) -> list[str]:
    """Return documented names for an API module."""
    return importlib.import_module(module_name).__all__


def _render_jinja(app: Sphinx, docname: str, source: list[str]) -> None:
    renderer = generate.AutosummaryRenderer(app)
    source[0] = renderer.env.from_string(source[0]).render(
        api_names=api_names,
    )


def _patch_autosummary_jinja(app: Sphinx) -> None:
    renderer = generate.AutosummaryRenderer(app)

    def find_autosummary_in_files(
        filenames: list[str],
    ) -> list[AutosummaryEntry]:
        documented: list[AutosummaryEntry] = []
        for filename in filenames:
            source = Path(filename).read_text(
                encoding=app.config.source_encoding,
                errors="ignore",
            )
            source = renderer.env.from_string(source).render(
                api_names=api_names,
            )
            documented.extend(
                generate.find_autosummary_in_lines(
                    source.splitlines(),
                    filename=filename,
                )
            )
        return documented

    generate.find_autosummary_in_files = find_autosummary_in_files  # type: ignore


def _run_on_doctree_read(
    _app: Sphinx,
    doctree: Node,
) -> None:
    for node in doctree.findall(nodes.literal_block):
        if node.get("testnodetype") != "testcode":
            continue

        # HTML renders the node text; the doctest builder executes "test".
        code = node["test"] if "test" in node else node.astext()
        code = code.replace('"cuda"', '"cpu"')
        node["test"] = code.replace("'cuda'", "'cpu'")


def setup(app: Sphinx) -> None:
    """Register Jinja rendering for dynamic autosummary lists."""
    app.connect("builder-inited", _patch_autosummary_jinja, priority=400)
    app.connect("source-read", _render_jinja)
    app.connect("doctree-read", _run_on_doctree_read)
