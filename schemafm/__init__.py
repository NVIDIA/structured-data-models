from importlib.metadata import PackageNotFoundError, version

from schemafm.schema import Schema, Stype
from schemafm.table_tensor import TableTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Schema",
    "Stype",
    "TableTensor",
    "__version__",
]
