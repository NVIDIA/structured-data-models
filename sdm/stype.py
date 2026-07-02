"""Semantic column types identifiers."""

from enum import Enum
from typing import TypeAlias


class Stype(str, Enum):
    r"""The semantic type of a table column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into a feature space.
    Possible values are:

    Attributes:
        numerical: Numerical columns.
        categorical: Categorical columns.
        id: Identifier values used to distinguish or link entities. Identifier
        columns are not used as model features by default.
    """

    numerical = "numerical"
    categorical = "categorical"
    id = "id"


StypeLike: TypeAlias = Stype | str
