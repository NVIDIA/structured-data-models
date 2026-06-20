from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype, StypeLike
from schemafm.tensor import VarLenTensor, StringTensor, TableTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "StringTensor",
    "Stype",
    "StypeLike",
    "TableTensor",
    "VarLenTensor",
    "__version__",
]
