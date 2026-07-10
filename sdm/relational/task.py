from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from typing_extensions import Self

from sdm import TableTensor
from sdm.relational import Relationship
from sdm.relational.data import LEFT_ROW_ID, RIGHT_ROW_ID, ROW_ID
from sdm.tensor.mixin import DeviceMixin


@dataclass(frozen=True)
class TaskLink:
    r"""Link between task rows to a table in relational data.

    Args:
        task_columns: Column names in the task table.
        table: Name of the table referenced by task rows.
        table_columns: Column names in ``table``.
    """

    task_columns: Sequence[str]
    table: str
    table_columns: Sequence[str]

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
            for reserved in (ROW_ID, LEFT_ROW_ID, RIGHT_ROW_ID):
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
            task_columns=task_columns,
            table=table,
            table_columns=table_columns,
        )


@dataclass(frozen=True, init=False)
class RelatedTables(DeviceMixin):
    r"""Task-specific related tables attached to model inputs.

    :class:`RelatedTables` store the relational context provided to a model
    for a particular task table.
    It may contain a sampled subset of a larger :class:`RelationalData`.
    The ``task_link`` describes how rows in the model input match rows in
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
        return self.__class__(
            tables={
                table_name: cast(TableTensor, table.to(device))
                for table_name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
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
