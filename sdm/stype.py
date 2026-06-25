"""Semantic column types."""

from enum import Enum
from typing import Literal, TypeAlias


class Stype(str, Enum):
    """Semantic column type."""

    numerical = "numerical"
    categorical = "categorical"


StypeLike: TypeAlias = Stype | Literal["numerical", "categorical"]
