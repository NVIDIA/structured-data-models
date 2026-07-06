from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import prod

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
        right_table: Name of the right related table, or ``None`` for the task
            table.
        right_columns: Column names from the right table.
    """

    left_table: str | None
    left_columns: Sequence[str]
    right_table: str | None
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

        if self.left_table is None and self.right_table is None:
            raise ValueError(
                "Expected either 'left_table' or 'right_table' to refer to a "
                "related table"
            )

        for column in (*self.left_columns, *self.right_columns):
            for reserved in (ROW_ID, LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name '{column}' is reserved for internal "
                        f"row indexing"
                    )


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

    def node_batch(
        self,
        task_table: TableTensor,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        r"""Assign each graph node to a task-row graph.

        Args:
            task_table: The task table referenced by relationships whose
                ``left_table`` or ``right_table`` is ``None``.
            device: The device.

        Returns:
            A PyG-style batch vector with shape ``[N]`` where ``N`` is the
            total number of rows across the task table and all related tables.
            Values are task table row indices, and ``-1`` marks nodes not
            reachable from a task row.
        """
        device = (
            torch.device(device) if device is not None else task_table.device
        )

        tables: dict[str | None, TableTensor] = {None: task_table}
        tables.update(self.tables)

        offsets: dict[str | None, int] = {}
        num_nodes = 0
        for name, table in tables.items():
            offsets[name] = num_nodes
            num_nodes += prod(table.size()[:-1])

        node_batch = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        num_task_rows = prod(task_table.size()[:-1])
        task_rows = torch.arange(
            num_task_rows,
            dtype=torch.long,
            device=device,
        )
        node_batch[task_rows + offsets[None]] = task_rows

        edge_indices = self.edge_indices(
            task_table,
            dtype=torch.long,
            device=device,
        )
        for _ in range(max(num_nodes, 1)):
            changed = False
            for rel, edge_index in zip(self.relationships, edge_indices):
                edge_index = edge_index.clone()
                edge_index[0] += offsets[rel.left_table]
                edge_index[1] += offsets[rel.right_table]

                for src, dst in (
                    (edge_index[0], edge_index[1]),
                    (edge_index[1], edge_index[0]),
                ):
                    src_owner = node_batch[src]
                    mask = src_owner >= 0
                    if not mask.any():
                        continue

                    dst = dst[mask]
                    proposed_owner = src_owner[mask]

                    order = dst.argsort()
                    dst = dst[order]
                    proposed_owner = proposed_owner[order]

                    duplicate = dst[1:] == dst[:-1]
                    conflict = duplicate & (
                        proposed_owner[1:] != proposed_owner[:-1]
                    )
                    if conflict.any():
                        raise ValueError(
                            "Expected each graph node to belong to at most "
                            "one task row"
                        )

                    existing_owner = node_batch[dst]
                    conflict = (existing_owner >= 0) & (
                        existing_owner != proposed_owner
                    )
                    if conflict.any():
                        raise ValueError(
                            "Expected each graph node to belong to at most "
                            "one task row"
                        )

                    mask = existing_owner < 0
                    node_batch[dst[mask]] = proposed_owner[mask]
                    changed = changed or bool(mask.any())

            if not changed:
                return node_batch

        return node_batch
