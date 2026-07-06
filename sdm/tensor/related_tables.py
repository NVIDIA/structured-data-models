from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Literal, NamedTuple, overload

import pyarrow as pa
import torch
from torch import Tensor

from sdm import Stype
from sdm.tensor import TableTensor

PREFIX = "sdm_internal"
ROW_ID = f"__{PREFIX}_row_id__"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"


@dataclass(frozen=True)
class Relationship:
    r"""Join relationship between two tables.

    Args:
        left_table: Name of the left related table, or ``None`` for the task
            table.
        left_columns: Column names from the left table.
        right_table: Name of the right related table.
        right_columns: Column names from the right table.
    """

    left_table: str | None
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
        num_rows: The number of rows in the homogeneous graph.
        node_offsets: Global node offsets keyed by related table name.
        edge_index: Edge index with shape ``[2, num_edges]`` over the
            concatenated rows of all related tables.
        task_edge_indices: Task-to-related-table edge indices keyed by
            relationship index.
    """

    num_rows: int
    node_offsets: dict[str, int]
    edge_index: Tensor
    task_edge_indices: dict[int, Tensor]


@dataclass(frozen=True, init=False)
class RelatedTables:
    r"""Related table context for relational data models.

    .. code-block:: python

        from sdm import RelatedTables

        related_tables = RelatedTables(
            tables={
                "users": ...,
                "orders": ...,
                "items": ...,
            },
            relationships=[
                # Foreign key from the task table to the entity table:
                dict(left_table=None, left_column="user_id",
                     right_table="users", right_column="user_id"),
                # Foreign key from orders to users:
                dict(left_table="orders", left_column="user_id",
                     right_table="users", right_column="user_id"),
                # Foreign key from orders to items:
                dict(left_table="orders", left_column="item_id",
                     right_table="items", right_column="item_id"),
            ],
        )

    Args:
        tables: Related tables keyed by table name.
        relationships: Join relationships among related tables and the
            implicit task table.
    """

    tables: Mapping[str, TableTensor]
    relationships: tuple[Relationship, ...]

    def __init__(
        self,
        tables: Mapping[str, TableTensor],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str] | None]
        ],
    ):

        parsed_relationships = []
        for relationship in relationships:
            if isinstance(relationship, Relationship):
                parsed_relationships.append(relationship)
            else:
                left_table = relationship.get("left_table")
                assert left_table is None or isinstance(left_table, str)
                if "left_column" in relationship:
                    left_columns = relationship["left_column"]
                else:
                    left_columns = relationship["left_columns"]
                assert left_columns is not None
                if isinstance(left_columns, str):
                    left_columns = (left_columns,)
                right_table = relationship["right_table"]
                assert isinstance(right_table, str)
                if "right_column" in relationship:
                    right_columns = relationship["right_column"]
                else:
                    right_columns = relationship["right_columns"]
                assert right_columns is not None
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
        object.__setattr__(self, "relationships", parsed_relationships)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not any(
            relationship.left_table is None or relationship.right_table is None
            for relationship in self.relationships
        ):
            raise ValueError(
                "Expected at least one relationship to refer to the task table"
            )

        for relationship in self.relationships:
            for table, columns in (
                (relationship.left_table, relationship.left_columns),
                (relationship.right_table, relationship.right_columns),
            ):
                if table is None:
                    continue

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
        task_table: TableTensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, ...]:
        r"""Materialize graph edges for table relationships.

        Args:
            task_table: The task table referenced by relationships whose
            ``left_table`` or ``right_table`` is ``None``.
            dtype: The dtype.
            device: The device.

        Returns:
            The edge indices for each relationship in order.
            Each edge index has shape ``[2, num_edges]`` and stores left table
            indices in the first row and right table indices in the second row.
        """
        columns: dict[str | None, list[str]] = defaultdict(list)
        for rel in self.relationships:
            columns[rel.left_table].extend(rel.left_columns)
            columns[rel.right_table].extend(rel.right_columns)

        tables = {
            name: table[..., columns[name]].to_arrow()
            for name, table in self.tables.items()
            if name in columns
        } | {None: task_table[..., columns[None]].to_arrow()}

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
        task_table: TableTensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> HomogeneousGraph:
        r"""Materialize homogeneous graph edges for table relationships.

        Args:
            task_table: The task table referenced by relationships whose
                ``left_table`` or ``right_table`` is ``None``.
            dtype: The dtype.
            device: The device.

        Returns:
            A materialized related-table graph.
        """
        offset = 0
        offsets: dict[str, int] = {}
        for name, table in self.tables.items():
            offsets[name] = offset
            offset += prod(table.size()[:-1])

        edge_indices: list[Tensor] = []
        task_edge_indices: dict[int, Tensor] = {}
        for i, (relationship, edge_index) in enumerate(
            zip(
                self.relationships,
                self.edge_indices(task_table, dtype=dtype, device=device),
            )
        ):
            if relationship.left_table is None:
                task_edge_indices[i] = edge_index
                continue

            edge_index += edge_index.new_tensor(
                [
                    [offsets[relationship.left_table]],
                    [offsets[relationship.right_table]],
                ],
            )
            edge_indices.append(edge_index)
            edge_indices.append(edge_index.flip(0))

        if len(edge_indices) == 0:
            dtype = torch.long if dtype is None else dtype
            edge_index = torch.empty((2, 0), dtype=dtype, device=device)
        elif len(edge_indices) == 1:
            edge_index = edge_indices[0]
        else:
            edge_index = torch.cat(edge_indices, dim=1)

        return HomogeneousGraph(
            num_rows=offset,
            node_offsets=offsets,
            edge_index=edge_index,
            task_edge_indices=task_edge_indices,
        )

    @overload
    def row_batch(
        self,
        graph: HomogeneousGraph,
        *,
        return_num_hops: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def row_batch(
        self,
        graph: HomogeneousGraph,
        *,
        return_num_hops: Literal[True],
    ) -> tuple[Tensor, int]: ...

    @overload
    def row_batch(
        self,
        graph: HomogeneousGraph,
        *,
        return_num_hops: bool,
    ) -> Tensor | tuple[Tensor, int]: ...

    def row_batch(
        self,
        graph: HomogeneousGraph,
        *,
        return_num_hops: bool = False,
    ) -> Tensor | tuple[Tensor, int]:
        r"""Return the task-row assignment for each related table row.

        Related table neighborhoods are assumed to be disjoint: Each reachable
        table row should belong to at most one task-table row.

        Args:
            graph: The homogeneous graph.
            return_num_hops: Whether to also return the number of propagation
            hops needed to assign reachable rows.

        Returns:
            A row-batch vector with shape ``[R]`` where ``R`` is the
            total number of rows across all related tables, which assigns each
            row to its task row, or ``-1`` otherwise.
            If ``return_num_hops`` is ``True``, also returns the number of
            propagation hops used.
        """
        row_batch = graph.edge_index.new_full((graph.num_rows,), fill_value=-1)
        frontier = torch.zeros_like(row_batch, dtype=torch.bool)

        for i, task_edge_index in graph.task_edge_indices.items():
            rel = self.relationships[i]
            dst = task_edge_index[1] + graph.node_offsets[rel.right_table]

            row_batch[dst] = task_edge_index[0]
            frontier[dst] = True

        num_hops = 0
        while True:
            src, dst = graph.edge_index
            mask = frontier[src] & (row_batch[dst] < 0)

            src = src[mask]
            if src.numel() == 0:
                break
            dst = dst[mask]

            row_batch[dst] = row_batch[src]
            frontier.fill_(False)
            frontier[dst] = True
            num_hops += 1

        if return_num_hops:
            return row_batch, num_hops

        return row_batch
