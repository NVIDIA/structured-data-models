"""Tensor subclasses for structured table data."""

from sdm.tensor.var_len import VarLenTensor
from sdm.tensor.string import StringTensor
from sdm.tensor.categorical import CategoricalTensor
from sdm.tensor.table import TableTensor

__all__ = [
    "CategoricalTensor",
    "StringTensor",
    "TableTensor",
    "VarLenTensor",
]
