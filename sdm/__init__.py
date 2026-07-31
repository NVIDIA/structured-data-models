"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm.stype import Stype, StypeLike, infer_stypes
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
    EnsembleTable,
)
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
    TemporalSamplingConfig,
)
from sdm import evaluation, models

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
    "EnsembleTable",
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "TemporalSamplingConfig",
    "evaluation",
    "models",
    "__version__",
]
