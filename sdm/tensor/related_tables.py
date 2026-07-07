from collections import defaultdict
from collections.abc import Mapping, Sequence
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


def _coalesce_task_assignments(
    task_index: Tensor,
    node_index: Tensor,
) -> tuple[Tensor, Tensor]:
    perm = node_index.argsort(stable=True)
    node_index = node_index[perm]
    task_index = task_index[perm]

    duplicate = node_index[1:] == node_index[:-1]
    conflict = duplicate & (task_index[1:] != task_index[:-1])
    if bool(conflict.any()):
        raise ValueError(
            "Conflicting task assignments in a related-table component"
        )

    keep = torch.ones_like(node_index, dtype=torch.bool)
    keep[1:] = ~duplicate
    return task_index[keep], node_index[keep]


def _row_batch(
    *,
    num_rows: int,
    edge_index: Tensor,
    task_edge_indices: Mapping[int, Tensor],
) -> tuple[Tensor, int]:
    row_batch = edge_index.new_full((num_rows,), fill_value=-1)
    frontier = torch.zeros_like(row_batch, dtype=torch.bool)

    if task_edge_indices:
        task_index = torch.cat(
            [edge_index[0] for edge_index in task_edge_indices.values()]
        )
        node_index = torch.cat(
            [edge_index[1] for edge_index in task_edge_indices.values()]
        )
    else:
        task_index = edge_index.new_empty(0)
        node_index = edge_index.new_empty(0)

    task_index, node_index = _coalesce_task_assignments(
        task_index,
        node_index,
    )
    row_batch[node_index] = task_index
    frontier[node_index] = True

    num_hops = 0
    while True:
        src, dst = edge_index
        mask = frontier[src]

        src = src[mask]
        if src.numel() == 0:
            break
        dst = dst[mask]

        task_index = row_batch[src]
        assigned_task_index = row_batch[dst]
        assigned = assigned_task_index >= 0
        conflict = assigned & (assigned_task_index != task_index)
        if bool(conflict.any()):
            raise ValueError(
                "Conflicting task assignments in a related-table component"
            )

        task_index = task_index[~assigned]
        dst = dst[~assigned]
        if dst.numel() == 0:
            break
        task_index, dst = _coalesce_task_assignments(task_index, dst)

        row_batch[dst] = task_index
        frontier.fill_(False)
        frontier[dst] = True
        num_hops += 1

    src, dst = edge_index
    src_task_index = row_batch[src]
    dst_task_index = row_batch[dst]
    assigned = (src_task_index >= 0) & (dst_task_index >= 0)
    conflict = assigned & (src_task_index != dst_task_index)
    if bool(conflict.any()):
        raise ValueError(
            "Conflicting task assignments in a related-table component"
        )

    return row_batch, num_hops


@dataclass(frozen=True)
class Relationship:
    r"""Join relationship between two tables.

    Args:
        left_table: Name of the left related table, or ``None`` for the task
            table.
        left_columns: Column names from the left table.
        right_table: Name of the right related table, or ``None`` for the task
            table.
        right_columns: Column names from the right table.
    """

    left_table: str | None
    left_columns: Sequence[str]
    right_table: str | None
    right_columns: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_columns", tuple(self.left_columns))
        object.__setattr__(self, "right_columns", tuple(self.right_columns))

        if self.left_table is None and self.right_table is None:
            raise ValueError(
                "Expected at least one side of a relationship to refer to a "
                "related table"
            )

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
        num_task_rows: The number of rows in the task table.
        node_offsets: Global node offsets keyed by related table name.
        edge_index: Edge index with shape ``[2, num_edges]`` over the
            concatenated rows of all related tables.
        edge_type: Directed edge-type IDs with shape ``[num_edges]``.
        num_edge_types: The number of directed edge types in the relationship
            schema.
        task_edge_indices: Task-to-related-table edge indices keyed by
            relationship index. Each tensor stores task-table row indices in
            its first row and global related-table node indices in its second.
        node_batch: Task-row assignment for every graph node, with ``-1`` for
            nodes outside all task neighborhoods.
        num_hops: Number of propagation hops used to assign ``node_batch``.
    """

    num_rows: int
    num_task_rows: int
    node_offsets: dict[str, int]
    edge_index: Tensor
    edge_type: Tensor
    num_edge_types: int
    task_edge_indices: dict[int, Tensor]
    node_batch: Tensor
    num_hops: int


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
        relationships: Sequence[
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
                right_table = relationship.get("right_table")
                assert right_table is None or isinstance(right_table, str)
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
        object.__setattr__(self, "relationships", tuple(parsed_relationships))
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
        for rel in self.relationships:
            for table, rel_columns in (
                (rel.left_table, rel.left_columns),
                (rel.right_table, rel.right_columns),
            ):
                if table is not None:
                    continue
                for column in rel_columns:
                    if task_table.stype(column) != Stype.id:
                        raise ValueError(
                            f"Expected column '{column}' in task table "
                            f"to have semantic type '{Stype.id.value}'"
                        )

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
        device: torch.device | str | None = None,
    ) -> HomogeneousGraph:
        r"""Materialize homogeneous graph edges for table relationships.

        Args:
            task_table: The task table referenced by relationships whose
                ``left_table`` or ``right_table`` is ``None``.
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
        edge_types: list[Tensor] = []
        num_edge_types = 0
        task_edge_indices: dict[int, Tensor] = {}
        for i, (relationship, edge_index) in enumerate(
            zip(
                self.relationships,
                self.edge_indices(
                    task_table,
                    dtype=torch.long,
                    device=device,
                ),
            )
        ):
            left_table = relationship.left_table
            right_table = relationship.right_table
            if left_table is None or right_table is None:
                if left_table is None:
                    assert right_table is not None
                    task_index, node_index = edge_index
                    related_table = right_table
                else:
                    node_index, task_index = edge_index
                    related_table = left_table
                task_edge_indices[i] = torch.stack(
                    (
                        task_index,
                        node_index + offsets[related_table],
                    )
                )
                continue

            edge_offset = edge_index.new_tensor(
                [
                    [offsets[left_table]],
                    [offsets[right_table]],
                ],
            )
            edge_index = edge_index + edge_offset
            edge_indices.append(edge_index)
            edge_types.append(
                torch.full(
                    (edge_index.size(1),),
                    num_edge_types,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            )
            num_edge_types += 1
            edge_indices.append(edge_index.flip(0))
            edge_types.append(
                torch.full(
                    (edge_index.size(1),),
                    num_edge_types,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            )
            num_edge_types += 1

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        elif len(edge_indices) == 1:
            edge_index = edge_indices[0]
        else:
            edge_index = torch.cat(edge_indices, dim=1)

        if len(edge_types) == 0:
            edge_type = torch.empty(0, dtype=torch.long, device=device)
        elif len(edge_types) == 1:
            edge_type = edge_types[0]
        else:
            edge_type = torch.cat(edge_types)

        node_batch, num_hops = _row_batch(
            num_rows=offset,
            edge_index=edge_index,
            task_edge_indices=task_edge_indices,
        )
        return HomogeneousGraph(
            num_rows=offset,
            num_task_rows=prod(task_table.size()[:-1]),
            node_offsets=offsets,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            task_edge_indices=task_edge_indices,
            node_batch=node_batch,
            num_hops=num_hops,
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

        .. note::

            Related table neighborhoods are assumed to be disjoint: Each
            reachable table row should belong to at most one task-table row.

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
        if return_num_hops:
            return graph.node_batch, graph.num_hops

        return graph.node_batch
