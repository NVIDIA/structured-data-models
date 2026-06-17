from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias


class Stype(str, Enum):
    numerical = "numerical"
    categorical = "categorical"


StypeLike: TypeAlias = Stype | Literal["numerical", "categorical"]


@dataclass(frozen=True)
class Column:
    name: str
    stype: Stype
    width: int = 1

    def __init__(
        self,
        name: str,
        stype: StypeLike,
        width: int = 1,
    ) -> None:
        if width < 1:
            raise ValueError("'width' must be positive")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stype", Stype(stype))
        object.__setattr__(self, "width", width)


ColumnLike: TypeAlias = Column | Mapping[str, Any]
