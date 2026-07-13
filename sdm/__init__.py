"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm.stype import Stype, StypeLike, infer_stypes
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from sdm.relational import (
    Relationship,
    RelationalData,
    RelationalSample,
    TaskLink,
    RelatedTables,
)

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Stype",
    "StypeLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "Relationship",
    "RelationalData",
    "RelationalSample",
    "TaskLink",
    "RelatedTables",
    "__version__",
]
