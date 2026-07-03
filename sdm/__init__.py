"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm.stype import Stype, StypeLike
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
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
    "ColumnarTensor",
    "TableTensor",
    "Relationship",
    "RelatedTables",
    "__version__",
]
