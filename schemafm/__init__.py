from importlib.metadata import PackageNotFoundError, version

from schemafm.column import Stype, StypeLike, Column, ColumnLike
from schemafm.table_tensor import TableTensor

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "Column",
    "ColumnLike",
    "Stype",
    "StypeLike",
    "TableTensor",
    "__version__",
]
