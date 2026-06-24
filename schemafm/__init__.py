"""Foundation models and tensor containers for structured data."""

from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype, StypeLike
from schemafm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    TableTensor,
)

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "CategoricalTensor",
    "StringTensor",
    "Stype",
    "StypeLike",
    "TableTensor",
    "VarLenTensor",
    "__version__",
]
