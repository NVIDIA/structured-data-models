from importlib.metadata import PackageNotFoundError, version

from schemafm.stype import Stype
from schemafm.table_tensor import TableTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Stype",
    "TableTensor",
    "__version__",
]
