"""Foundation models and tensor containers for structured data."""

from importlib.metadata import PackageNotFoundError, version

from sdm.stype import Stype, StypeLike
from sdm.tensor import (
    CategoricalTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
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
    "__version__",
]
