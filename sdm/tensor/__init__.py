"""Structured data containers."""

from sdm.tensor.var_len import VarLenTensor
from sdm.tensor.string import StringTensor
from sdm.tensor.categorical import CategoricalTensor
from sdm.tensor.columnar import ColumnarTensor
from sdm.tensor.table import TableTensor
from sdm.tensor.related_tables import Relationship, RelationalContext

__all__ = [
    "VarLenTensor",
    "StringTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "Relationship",
    "RelationalContext",
]
