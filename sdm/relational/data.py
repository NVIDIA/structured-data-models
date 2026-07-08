from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, NamedTuple

import pyarrow as pa
import torch
from torch import Tensor

from sdm import Stype, TableTensor

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


class HomogeneousGraph(NamedTuple):
    r"""Materialized homogeneous graph.

    Args:
        node_offsets: Global node offsets keyed by table name.
        edge_index: Edge index with shape ``[2, num_edges]`` over the
            concatenated rows of all tables.
    """

    edge_index: Tensor
    node_offsets: dict[str, int]


@dataclass(frozen=True, init=False)
class RelationalData:
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
        relationships: Join relationships among tables.
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
        parsed_relationships = []
        for relationship in relationships:
            if isinstance(relationship, Relationship):
                parsed_relationships.append(relationship)
            else:
                left_table = relationship["left_table"]
                assert isinstance(left_table, str)
                if "left_column" in relationship:
                    left_columns = relationship["left_column"]
                else:
                    left_columns = relationship["left_columns"]
                if isinstance(left_columns, str):
                    left_columns = (left_columns,)

                right_table = relationship["right_table"]
                assert isinstance(right_table, str)
                if "right_column" in relationship:
                    right_columns = relationship["right_column"]
                else:
                    right_columns = relationship["right_columns"]
                if isinstance(right_columns, str):
                    right_columns = (right_columns,)

                relationship = Relationship(
                    left_table=left_table,
                    left_columns=left_columns,
                    right_table=right_table,
                    right_columns=right_columns,
                )
                parsed_relationships.append(relationship)

        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", tuple(parsed_relationships))
        self.__post_init__()

    def __post_init__(self) -> None:
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
                    if self.tables[table].stype(column) != Stype.id:
                        raise ValueError(
                            f"Expected column '{column}' in table '{table}' "
                            f"to have semantic type '{Stype.id.value}'"
                        )

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

    def homogeneous_graph(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> HomogeneousGraph:
        r"""Materialize homogeneous graph edges for table relationships.

        Args:
            dtype: The dtype.
            device: The device.
        """
        offset = 0
        node_offsets: dict[str, int] = {}
        for name, table in self.tables.items():
            node_offsets[name] = offset
            offset += prod(table.size()[:-1])

        edge_indices = []
        for relationship, edge_index in zip(
            self.relationships,
            self.edge_indices(dtype=dtype, device=device),
        ):
            edge_index += edge_index.new_tensor(
                [
                    [node_offsets[relationship.left_table]],
                    [node_offsets[relationship.right_table]],
                ],
            )
            edge_indices.append(edge_index)
            edge_indices.append(edge_index.flip(0))

        if len(edge_indices) == 0:
            dtype = torch.int64 if dtype is None else dtype
            edge_index = torch.empty((2, 0), dtype=dtype, device=device)
        else:
            edge_index = torch.cat(edge_indices, dim=1)

        return HomogeneousGraph(
            edge_index=edge_index,
            node_offsets=node_offsets,
        )

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
