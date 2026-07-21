from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from typing_extensions import Self

from sdm import TableTensor
from sdm.relational import RelationalData, Relationship
from sdm.relational.join import LEFT_ROW_ID, RIGHT_ROW_ID
from sdm.tensor.mixin import DeviceMixin
from sdm.tensor.table import TableSchema

if TYPE_CHECKING:
    import graphviz


TASK_TABLE = "__task_table__"


@dataclass(frozen=True, repr=False)
class TaskLink:
    r"""Link between task rows to a table in relational data.

    Args:
        task_columns: Column names in the task table.
        table: Name of the table referenced by task rows.
        table_columns: Column names in ``table``.
    """

    task_columns: tuple[str, ...]
    table: str
    table_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.task_columns) != len(self.table_columns):
            raise ValueError(
                f"Expected 'task_columns' and 'table_columns' to have the "
                f"same length (got {len(self.task_columns)} and "
                f"{len(self.table_columns)})"
            )

        if len(self.task_columns) == 0:
            raise ValueError(
                "Expected 'task_columns' and 'table_columns' to be non-empty"
            )

        for column in (*self.task_columns, *self.table_columns):
            for reserved in (LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name '{column}' is reserved for internal "
                        f"row indexing"
                    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | Sequence[str]]) -> Self:
        r"""Create a :class:`TaskLink` from a mapping.

        Args:
            mapping: The mapping with ``"task_columns"``, ``"table"``, and
                ``"table_columns"`` entries.
        """
        if "task_column" in mapping:
            task_columns = mapping["task_column"]
        else:
            task_columns = mapping["task_columns"]
        if isinstance(task_columns, str):
            task_columns = (task_columns,)

        table = mapping["table"]
        assert isinstance(table, str)

        if "table_column" in mapping:
            table_columns = mapping["table_column"]
        else:
            table_columns = mapping["table_columns"]
        if isinstance(table_columns, str):
            table_columns = (table_columns,)

        return cls(
            task_columns=tuple(task_columns),
            table=table,
            table_columns=tuple(table_columns),
        )

    def _task_columns_repr(self) -> str:
        if len(self.task_columns) == 1:
            return self.task_columns[0]
        return f"[{','.join(self.task_columns)}]"

    def _table_columns_repr(self) -> str:
        if len(self.table_columns) == 1:
            return f"{self.table}.{self.table_columns[0]}"
        return f"{self.table}.[{','.join(self.table_columns)}]"

    def __repr__(self) -> str:
        return f"{self._task_columns_repr()} > {self._table_columns_repr()}"


@dataclass(frozen=True)
class RelatedTablesSchema:
    r"""The schema of :class:`RelatedTables`.

    Args:
        tables: Table schema keyed by table name.
        relationships: Join relationships among ``tables``.
        task_links: Links from task columns to related ``tables``.
    """

    tables: Mapping[str, TableSchema]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def is_subset_of(self, other: RelatedTablesSchema) -> bool:
        r"""Whether this schema is an induced subset of ``other``."""
        for table_name, schema in self.tables.items():
            if schema != other.tables.get(table_name):
                return False

        other_relationships = {
            relationship
            for relationship in other.relationships
            if relationship.left_table in self.tables
            and relationship.right_table in self.tables
        }
        if set(self.relationships) != other_relationships:
            return False

        other_task_links = {
            task_link
            for task_link in other.task_links
            if task_link.table in self.tables
        }
        return set(self.task_links) == other_task_links


@dataclass(frozen=True, init=False, repr=False)
class RelatedTables(DeviceMixin):
    r"""Task-specific related tables attached to model inputs.

    :class:`RelatedTables` store the relational context provided to a model
    for a particular task table.
    It may contain a sampled subset of a larger :class:`RelationalData`.
    The ``task_links`` describe how rows in the model input match to rows in
    the related tables.

    .. code-block:: python

        from sdm import RelatedTables, TableTensor

        data = RelatedTables(
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
            task_links=[
                # Foreign key in the task table to users:
                dict(task_column="ENTITY", table="users",
                     table_column="user_id")
            ],
        )

    Args:
        tables: Related tables keyed by table name.
        relationships: Join relationships among ``tables``.
        task_links: Links from task columns to related ``tables``.
    """

    tables: Mapping[str, TableTensor]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def __init__(
        self,
        tables: Mapping[str, TableTensor],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str]]
        ],
        task_links: Collection[TaskLink | Mapping[str, str | Sequence[str]]],
    ) -> None:

        relationships = tuple(
            relationship
            if isinstance(relationship, Relationship)
            else Relationship.from_mapping(relationship)
            for relationship in relationships
        )

        task_links = tuple(
            task_link
            if isinstance(task_link, TaskLink)
            else TaskLink.from_mapping(task_link)
            for task_link in task_links
        )

        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", relationships)
        object.__setattr__(self, "task_links", task_links)
        self.__post_init__()

    def __post_init__(self) -> None:
        for table in self.tables.values():
            if table.dim() != 2:
                raise ValueError("Tables need to be two-dimensional")

    def to(self, device: torch.device | str | None) -> Self:
        r""":meta private:"""  # noqa: D415
        return self.__class__(
            tables={
                table_name: cast(TableTensor, table.to(device))
                for table_name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415
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

    @property
    def schema(self) -> RelatedTablesSchema:
        r"""The schema of this related context."""
        return RelatedTablesSchema(
            tables={name: table.schema for name, table in self.tables.items()},
            relationships=self.relationships,
            task_links=self.task_links,
        )

    def is_same_schema(self, other: RelatedTables) -> bool:
        r"""Whether ``other`` has the same schema layout.

        Args:
            other: The object to compare against.
        """
        return self.schema == other.schema

    def select_tables(self, tables: Iterable[str]) -> Self:
        r"""Return related tables containing only ``tables``.

        Args:
            tables: The table names to select.
        """
        tables = set(tables)

        return self.__class__(
            tables={
                table_name: table
                for table_name, table in self.tables.items()
                if table_name in tables
            },
            relationships=tuple(
                relationship
                for relationship in self.relationships
                if relationship.left_table in tables
                and relationship.right_table in tables
            ),
            task_links=tuple(
                task_link
                for task_link in self.task_links
                if task_link.table in tables
            ),
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
        return RelationalData(
            tables=self.tables,
            relationships=self.relationships,
        ).edge_indices(dtype=dtype, device=device)

    def task_indices(
        self,
        task_table: TableTensor,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, ...]:
        r"""Materialize graph edges for task links.

        Args:
            task_table: The task table.
            dtype: The dtype.
            device: The device.

        Returns:
            The edge indices for each task link in order.
            Each edge index has shape ``[2, num_edges]`` and stores task table
            indices in the first row and table indices in the second row.
        """
        data = RelationalData(
            tables={**self.tables, "__task_table__": task_table},
            relationships=tuple(
                Relationship(
                    left_table="__task_table__",
                    left_columns=task_link.task_columns,
                    right_table=task_link.table,
                    right_columns=task_link.table_columns,
                )
                for task_link in self.task_links
            ),
        )
        return data.edge_indices(dtype=dtype, device=device)

    def to_graphviz(
        self,
        *,
        hide_columns: bool = False,
        **kwargs: Any,
    ) -> graphviz.Graph:
        r"""Return a task visualization of the relational schema.

        Args:
            hide_columns: Whether to hide column name descriptions.
            **kwargs: Additional keyword arguments passed to
                :class:`graphviz.Graph`.
        """
        graph = RelationalData(
            tables=self.tables,
            relationships=self.relationships,
        ).to_graphviz(hide_columns=hide_columns, **kwargs)

        graph.node(TASK_TABLE, label="", shape="point")

        for link in self.task_links:
            label = "\\n".join(
                f" {task_column} > {table_column} "
                for task_column, table_column in zip(
                    link.task_columns, link.table_columns
                )
            )
            graph.edge(
                TASK_TABLE,
                link.table,
                label=label,
                fontsize="11pt",
            )

        return graph

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}(\n"
        if len(self.tables) > 0:
            out += "  tables={\n"
            out += "".join(
                f"    {name}: {table.__repr__(indent=4)[4:]},\n"
                for name, table in self.tables.items()
            )
            out += "  },\n"
        else:
            out += "  tables={},\n"
        if len(self.relationships) > 0:
            out += "  relationships=[\n"
            out += "".join(f"    {rel},\n" for rel in self.relationships)
            out += "  ],\n"
        else:
            out += "  relationships=[],\n"
        if len(self.task_links) > 0:
            out += "  task_links=[\n"
            out += "".join(f"    {link},\n" for link in self.task_links)
            out += "  ],\n"
        else:
            out += "  task_links=[],\n"
        out += ")"
        return out

    def _repr_html_(self) -> str:
        from html import escape

        import pandas as pd

        rows = [
            [
                name,
                table.size(-2),
                table.size(-1),
                ", ".join(
                    stype.value
                    for stype, tensor in table.items()
                    if tensor.size(-1) > 0
                ),
            ]
            for name, table in self.tables.items()
        ]
        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Table", "Rows", "Columns", "Stypes"]),
        )

        ul = "".join(
            f"<li>"
            f"<code>{escape(link._task_columns_repr())}</code>"
            f" ➡️ "
            f"<code>{escape(link._table_columns_repr())}</code>"
            f"</li>"
            for link in self.task_links
        )
        ul += "".join(
            f"<li>"
            f"<code>{escape(rel._left_columns_repr())}</code>"
            f" ↔️ "
            f"<code>{escape(rel._right_columns_repr())}</code>"
            f"</li>"
            for rel in self.relationships
        )

        return df.to_html(index=False, escape=True) + f"<ul>{ul}</ul>"
