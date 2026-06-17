from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype, StypeLike
from schemafm.table_tensor import TableTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Stype",
    "StypeLike",
    "TableTensor",
    "__version__",
]
