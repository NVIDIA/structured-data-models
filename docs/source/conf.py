import importlib
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.ext.autosummary.generate import AutosummaryEntry

project = "Structured Data Models"
author = "NVIDIA"
copyright = f"{date.today().year}, NVIDIA"

try:
    release = package_version("structured-data-models")
except PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
html_theme = "shibuya"
html_title = project
html_theme_options = {
    "accent_color": "green",
    "github_url": "https://github.com/NVIDIA/structured-data-models",
}
autosummary_generate = True
autodoc_typehints = "both"
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "typing_extensions": (
        "https://typing-extensions.readthedocs.io/en/latest",
        None,
    ),
}


def api_names(module_name: str) -> list[str]:
    """Return documented names for an API module."""
    return importlib.import_module(module_name).__all__


def _render_jinja(app: Sphinx, docname: str, source: list[str]) -> None:
    source[0] = app.builder.templates.render_string(
        source[0],
        {"api_names": api_names},
    )


def _patch_autosummary_jinja(app: Sphinx) -> None:
    from sphinx.ext.autosummary import generate

    def find_autosummary_in_files(
        filenames: list[str],
    ) -> list[AutosummaryEntry]:
        documented: list[AutosummaryEntry] = []
        for filename in filenames:
            source = Path(filename).read_text(
                encoding=app.config.source_encoding,
                errors="ignore",
            )
            source = app.builder.templates.render_string(
                source,
                {"api_names": api_names},
            )
            documented.extend(
                generate.find_autosummary_in_lines(
                    source.splitlines(),
                    filename=filename,
                )
            )
        return documented

    generate.find_autosummary_in_files = find_autosummary_in_files  # type: ignore


def setup(app: Sphinx) -> None:
    """Register Jinja rendering for dynamic autosummary lists."""
    app.connect("builder-inited", _patch_autosummary_jinja, priority=400)
    app.connect("source-read", _render_jinja)
