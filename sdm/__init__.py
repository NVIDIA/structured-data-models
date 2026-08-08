"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm._constants import NaT
from sdm.stype import Stype, StypeLike, infer_stypes
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    NullableIntTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
    TemporalSamplingConfig,
)
from sdm.processing import Recipe
from sdm import evaluation, models

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "NaT",
    "Stype",
    "StypeLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "NullableIntTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "TemporalSamplingConfig",
    "Recipe",
    "evaluation",
    "models",
    "__version__",
]
