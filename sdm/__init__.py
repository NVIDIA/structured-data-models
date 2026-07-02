"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm.stype import Stype, StypeLike
from sdm.tensor import (
    CategoricalTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
    Relationship,
    RelatedTables,
)

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Stype",
    "StypeLike",
    "VarLenTensor",
    "StringTensor",
    "CategoricalTensor",
    "TableTensor",
    "Relationship",
    "RelatedTables",
    "__version__",
]
