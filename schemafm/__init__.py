from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype, StypeLike
from schemafm.tensor import (
    VarLenTensor,
    StringTensor,
    CategoricalTensor,
    TableTensor,
)

try:
    __version__ = version("schema-fm")
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
