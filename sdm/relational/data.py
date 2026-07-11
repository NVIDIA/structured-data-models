from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import Self

from sdm import Stype, TableTensor
from sdm.tensor.mixin import DeviceMixin

PREFIX = "sdm_internal"
ROW_ID = f"__{PREFIX}_row_id__"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"

if TYPE_CHECKING:
    from sdm.relational import RelationalSampler


@dataclass(frozen=True)
class Relationship:
    r"""Join relationship between two tables.

    Args:
        left_table: Name of the left table.
        left_columns: Column names from the left table.
        right_table: Name of the right table.
        right_columns: Column names from the right table.
    """

    left_table: str
    left_columns: Sequence[str]
    right_table: str
    right_columns: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.left_columns) != len(self.right_columns):
            raise ValueError(
                f"Expected 'left_columns' and 'right_columns' to have the "
                f"same length (got {len(self.left_columns)} and "
                f"{len(self.right_columns)})"
            )

        if len(self.left_columns) == 0:
            raise ValueError(
                "Expected 'left_columns' and 'right_columns' to be non-empty"
            )

        for column in (*self.left_columns, *self.right_columns):
            for reserved in (ROW_ID, LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name '{column}' is reserved for internal "
                        f"row indexing"
                    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | Sequence[str]]) -> Self:
        r"""Create a :class:`Relationship` from a mapping.

        Args:
            mapping: The mapping with ``"left_table"``, ``"left_columns"``,
                ``"right_table"``, ``"right_columns"`` entries.
        """
        left_table = mapping["left_table"]
        assert isinstance(left_table, str)

        if "left_column" in mapping:
            left_columns = mapping["left_column"]
        else:
            left_columns = mapping["left_columns"]
        if isinstance(left_columns, str):
            left_columns = (left_columns,)

        right_table = mapping["right_table"]
        assert isinstance(right_table, str)

        if "right_column" in mapping:
            right_columns = mapping["right_column"]
        else:
            right_columns = mapping["right_columns"]
        if isinstance(right_columns, str):
            right_columns = (right_columns,)

        return cls(
            left_table=left_table,
            left_columns=left_columns,
            right_table=right_table,
            right_columns=right_columns,
        )


@dataclass(frozen=True, init=False)
class RelationalData(DeviceMixin):
    r"""Collection of named tables and join relationships.

    .. code-block:: python

        from sdm import RelationalData, TableTensor

        data = RelationalData(
            tables={
                "users": TableTensor.from_pandas(...),
                "orders": TableTensor.from_pandas(...),
                "items": TableTensor.from_pandas(...),
            },
            relationships=[
                # Foreign key from orders to users:
                dict(left_table="orders", left_column="user_id",
                     right_table="users", right_column="user_id"),
                # Foreign key from orders to items:
                dict(left_table="orders", left_column="item_id",
                     right_table="items", right_column="item_id"),
            ],
        )

    Args:
        tables: Tables keyed by table name.
        relationships: Join relationships among ``tables``.
    """

    tables: Mapping[str, TableTensor]
    relationships: tuple[Relationship, ...]

    def __init__(
        self,
        tables: Mapping[str, TableTensor],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str]]
        ],
    ) -> None:

        relationships = tuple(
            relationship
            if isinstance(relationship, Relationship)
            else Relationship.from_mapping(relationship)
            for relationship in relationships
        )

        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", relationships)
        self.__post_init__()

    def __post_init__(self) -> None:
        for table in self.tables.values():
            if table.dim() != 2:
                raise ValueError("Tables need to be two-dimensional")

        for relationship in self.relationships:
            for table, columns in (
                (relationship.left_table, relationship.left_columns),
                (relationship.right_table, relationship.right_columns),
            ):
                if table not in self.tables:
                    raise ValueError(
                        f"Expected '{table}' to be registered as a table"
                    )

                for column in columns:
                    stype = self.tables[table].stype(column)
                    if stype != Stype.id:
                        raise ValueError(
                            f"Expected column '{column}' in table '{table}' "
                            f"to have semantic type '{Stype.id.value}' "
                            f"(got '{stype.value}')"
                        )

    def to(self, device: torch.device | str | None) -> Self:  # noqa: D102
        return self.__class__(
            tables={
                table_name: cast(TableTensor, table.to(device))
                for table_name, table in self.tables.items()
            },
            relationships=self.relationships,
        )

    @property
    def device(self) -> torch.device:  # noqa: D102
        devices = {table.device for table in self.tables.values()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"'{self.__class__.__name__}'"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tables in '{self.__class__.__name__}' to be on "
                f"the same device (got {list(devices)})"
            )
        return next(iter(devices))

    def edge_indices(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, ...]:
        r"""Materialize heterogeneous graph edges for table relationships.

        Args:
            dtype: The dtype.
            device: The device.

        Returns:
            The edge indices for each relationship in order.
            Each edge index has shape ``[2, num_edges]`` and stores left table
            indices in the first row and right table indices in the second row.
        """
        device = self.device if device is None else device

        columns: dict[str, list[str]] = defaultdict(list)
        for rel in self.relationships:
            columns[rel.left_table].extend(rel.left_columns)
            columns[rel.right_table].extend(rel.right_columns)

        tables = {
            name: table[..., columns[name]].to_arrow()
            for name, table in self.tables.items()
            if name in columns
        }

        tables = {
            name: table.append_column(
                ROW_ID,
                pa.array(torch.arange(table.num_rows, dtype=dtype).numpy()),
            )
            for name, table in tables.items()
        }

        edge_indices: list[Tensor] = []
        for rel in self.relationships:
            left = tables[rel.left_table]
            left = left.select((*rel.left_columns, ROW_ID))
            left = left.rename_columns({ROW_ID: LEFT_ROW_ID})
            right = tables[rel.right_table]
            right = right.select((*rel.right_columns, ROW_ID))
            right = right.rename_columns({ROW_ID: RIGHT_ROW_ID})

            joined = left.join(
                right,
                keys=list(rel.left_columns),
                right_keys=list(rel.right_columns),
                join_type="inner",
            )

            src = torch.from_numpy(joined[LEFT_ROW_ID].to_numpy()).to(device)
            dst = torch.from_numpy(joined[RIGHT_ROW_ID].to_numpy()).to(device)
            edge_indices.append(torch.stack([src, dst], dim=0))

        return tuple(edge_indices)

    def sampler(
        self,
        time_columns: Mapping[str, str] | None = None,
    ) -> "RelationalSampler":
        r"""Create a subgraph sampler over this relational data.

        .. code-block:: python

            from sdm import RelationalData, TableTensor

            data = RelationalData(
                tables={
                    "users": TableTensor.from_pandas(...),
                    "orders": TableTensor.from_pandas(...),
                    "items": TableTensor.from_pandas(...),
                },
                relationships=[
                    # Foreign key from orders to users:
                    dict(left_table="orders", left_column="user_id",
                         right_table="users", right_column="user_id"),
                    # Foreign key from orders to items:
                    dict(left_table="orders", left_column="item_id",
                         right_table="items", right_column="item_id"),
                ],
            )

            sampler = data.sampler(
                time_columns={"orders": "order_date"},
            )

        Args:
            time_columns: Mapping from table name to the datetime column used
                for temporal sampling. A row in a time-aware table can only be
                sampled if its timestamp does not exceed the query timestamp.
        """
        from sdm.relational import RelationalSampler

        return RelationalSampler(
            data=self,
            time_columns=time_columns,
        )
