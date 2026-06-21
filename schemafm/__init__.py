from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype, StypeLike
from schemafm.tensor import VarLenTensor, StringTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "StringTensor",
    "Stype",
    "StypeLike",
    "VarLenTensor",
    "__version__",
]
