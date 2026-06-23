"""Tensor subclasses for structured table data."""

from schemafm.tensor.var_len import VarLenTensor
from schemafm.tensor.string import StringTensor
from schemafm.tensor.categorical import CategoricalTensor
from schemafm.tensor.table import TableTensor

__all__ = [
    "CategoricalTensor",
    "StringTensor",
    "TableTensor",
    "VarLenTensor",
]
