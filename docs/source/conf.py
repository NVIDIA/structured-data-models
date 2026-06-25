import ast
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGE = "sdm"
PACKAGE_ROOT = ROOT / PACKAGE

project = "Structured Data Models"
author = "NVIDIA"
copyright = f"{date.today().year}, NVIDIA"

try:
    release = package_version("structured-data-models")
except PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
suppress_warnings = ["autoapi.python_import_resolution"]
html_theme = "shibuya"
html_title = project
html_theme_options = {
    "accent_color": "green",
    "github_url": "https://github.com/NVIDIA/structured-data-models",
}

autoapi_type = "python"
autoapi_dirs = [str(PACKAGE_ROOT)]
autoapi_ignore = [
    "*/sdm/testing/*",
]
autoapi_template_dir = "_templates/autoapi"
autoapi_root = "api/reference"
autoapi_add_toctree_entry = False
autoapi_own_page_level = "class"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}


def _literal_names(path: Path, name: str) -> set[str] | None:
    module = ast.parse(path.read_text())
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        try:
            values = ast.literal_eval(node.value)
        except (SyntaxError, ValueError):
            return None
        return {value for value in values if isinstance(value, str)}
    return None


def _package_docs(root: Path, package: str) -> dict[str, set[str]]:
    exports: dict[str, set[str]] = {}
    for path in root.rglob("__init__.py"):
        docs_names = _literal_names(path, "__docs__")
        if docs_names is None:
            continue
        relative = path.parent.relative_to(root)
        module = ".".join((package, *relative.parts))
        exports[module] = docs_names
    return exports


PACKAGE_DOCS = _package_docs(PACKAGE_ROOT, PACKAGE)


def _parent_package(name: str) -> str | None:
    parts = name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        package = ".".join(parts[:index])
        if package in PACKAGE_DOCS:
            return package
    return None


def _skip_undocumented_member(app, what, name, obj, skip, options):
    if not name.startswith(f"{PACKAGE}."):
        return skip

    if what in {"attribute", "class", "data", "exception", "function"}:
        parent = name.rsplit(".", 1)[0]
        docs = PACKAGE_DOCS.get(parent)
        if docs is not None and obj.short_name not in docs:
            return True
        return skip

    if what != "module":
        return skip

    parent = _parent_package(name)
    if parent is None:
        return skip

    docs = PACKAGE_DOCS[parent]
    module_name = name.rsplit(".", 1)[-1]
    return module_name not in docs


def setup(app):
    """Register documentation-only API filtering."""
    app.connect("autoapi-skip-member", _skip_undocumented_member)
