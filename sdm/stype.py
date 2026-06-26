"""Semantic column types sidentifiers."""

from enum import Enum
from typing import Literal, TypeAlias


class Stype(str, Enum):
    r"""The semantic type of a table column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into a feature space.
    Possible values are:

    Attributes:
        numerical: Numerical columns.
        categorical: Categorical columns.
    """

    numerical = "numerical"
    categorical = "categorical"


StypeLike: TypeAlias = Stype | Literal["numerical", "categorical"]
