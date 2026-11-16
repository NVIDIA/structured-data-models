# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TypeVar

from docutils import nodes
from docutils.nodes import Node
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import AttributeDocumenter, PropertyDocumenter
from sphinx.ext.autosummary import generate
from sphinx.ext.autosummary.generate import AutosummaryEntry
from sphinx.ext.intersphinx import missing_reference

project = "Structured Data Models"
author = "NVIDIA"
copyright = f"{date.today().year}, NVIDIA CORPORATION & AFFILIATES."  # noqa: A001

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
html_favicon = "images/nvidia.svg"
html_theme_options = {
    "accent_color": "green",
    "github_url": "https://github.com/NVIDIA/structured-data-models",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
autosummary_generate = True
autosummary_context = {
    "import_module": importlib.import_module,
    "getdoc": inspect.getdoc,
}
autodoc_member_order = "bysource"
autodoc_use_legacy_class_based = True
AttributeDocumenter.member_order = 45
PropertyDocumenter.member_order = 45
autodoc_typehints = "both"
napoleon_custom_sections = [
    ("Additional Forward Parameters", "params_style"),
]
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
suppress_warnings = ["config.cache"]
intersphinx_mapping = {
    "cudf": ("https://docs.rapids.ai/api/cudf/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "python": ("https://docs.python.org/3", None),
    "sentence_transformers": ("https://sbert.net", None),
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
import torch


def _reset_parameters(module):
    reset_parameters = getattr(module, "reset_parameters", None)
    if reset_parameters is not None:
        reset_parameters()


def _skip_pretrained(self, *_args, device=None, **_kwargs):
    if not any(parameter.is_meta for parameter in self.parameters()):
        return self

    target_device = torch.get_default_device() if device is None else device
    self.to_empty(device=target_device)
    self.apply(_reset_parameters)
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
        **app.config.autosummary_context,
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
                **app.config.autosummary_context,
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


_reference_aliases = {
    "Tensor": "torch.Tensor",
    "pd.DataFrame": "pandas.DataFrame",
    "pa.Table": "pyarrow.Table",
    "StypeLike": "sdm.StypeLike",
    "TaskLike": "sdm.TaskLike",
    "Processor": "sdm.processing.base.Processor",
    "EnsembleProcessor": "sdm.processing.ensemble.EnsembleProcessor",
    "SDPA": "sdm.nn.SDPA",
    "Stype.categorical": "sdm.Stype.categorical",
    "Stype.text": "sdm.Stype.text",
    "sdm.processing.AlignCategories": (
        "sdm.processing.categorical.AlignCategories"
    ),
    "sdm.processing.ToNumerical": "sdm.processing.common.ToNumerical",
    "sdm.processing.StypeDispatch": "sdm.processing.common.StypeDispatch",
    "sdm.processing.AverageEstimators": (
        "sdm.processing.output.AverageEstimators"
    ),
    "sentence_transformers.SentenceTransformer": "sentence_transformers.sentence_transformer.model.SentenceTransformer",  # noqa: E501
    "sentence_transformers.SentenceTransformer.encode": "sentence_transformers.sentence_transformer.model.SentenceTransformer.encode",  # noqa: E501
}


def _resolve_reference_alias(
    app: Sphinx,
    env: BuildEnvironment,
    node: pending_xref,
    content: nodes.Element,
) -> nodes.Element | None:
    if node.get("refdomain") != "py":
        return None
    module_name, _, name = node["reftarget"].rpartition(".")
    module = sys.modules.get(module_name or node.get("py:module"))
    if not module_name:
        cls = getattr(module, node.get("py:class") or "", None)
        module = sys.modules.get(getattr(cls, "__module__", None), module)
    if isinstance(getattr(module, name, None), TypeVar):
        return content
    target = _reference_aliases.get(node["reftarget"])
    if target is None:
        return None
    node["reftarget"] = target
    resolved = env.get_domain("py").resolve_xref(
        env=env,
        fromdocname=node["refdoc"],
        builder=app.builder,
        type=node["reftype"],
        target=target,
        node=node,
        contnode=content,
    )
    if resolved is not None:
        return resolved
    return missing_reference(app, env, node, content)


def setup(app: Sphinx) -> None:
    """Register Jinja rendering for dynamic autosummary lists."""
    app.connect("builder-inited", _patch_autosummary_jinja, priority=400)
    app.connect("source-read", _render_jinja)
    app.connect("doctree-read", _run_on_doctree_read)
    app.connect("missing-reference", _resolve_reference_alias, priority=400)
