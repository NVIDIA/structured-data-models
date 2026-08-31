from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Generic, Literal, NamedTuple, Self, TypeVar, cast

import torch
from torch import Tensor

from sdm import ColumnarTensor, Stype, TableTensor
from sdm.relational import (
    RelatedTables,
    RelationalData,
    Relationship,
    TaskLink,
)
from sdm.relational.backend import (
    CuGraphRelationalSampler,
    PyGLibRelationalSampler,
)
from sdm.tensor import EnsembleTable
from sdm.tensor.mixin import DeviceMixin

T = TypeVar("T", bound=TableTensor | EnsembleTable)

EXAMPLE_ID = "__example__"


class _RelationalSamplerOutput(NamedTuple, Generic[T]):
    task_table: TableTensor
    related_tables: RelatedTables[T]


class RelationalSamplerOutput(
    _RelationalSamplerOutput[T], DeviceMixin, Generic[T]
):
    r"""Relational sampler output.

    Args:
        task_table: The task table.
        related_tables: The related tables for the task table.
    """

    def _tensors(self) -> Iterator[Tensor]:
        yield self.task_table
        yield from self.related_tables._tensors()

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            task_table=cast(TableTensor, fn(self.task_table)),
            related_tables=self.related_tables._apply_tensor(fn),
        )


class RelationalSampler:
    r"""Subgraph sampler over relational data.

    Args:
        data: The collection of named tables and their relationships.
        time_columns: Mapping from table name to the datetime column used for
            temporal sampling. A row in a time-aware table can only be sampled
            if its timestamp does not exceed the query timestamp.
    """

    def __init__(
        self,
        data: RelationalData,
        time_columns: Mapping[str, str] | None = None,
    ) -> None:
        self.data = data
        self.time_columns = time_columns or {}

        for table_name, column_name in self.time_columns.items():
            stype = data.tables[table_name].stype(column_name)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected '{column_name}' in table '{table_name}' to "
                    f"have semantic type {str(Stype.datetime)!r} "
                    f"(got {str(stype)!r})"
                )

        if self.data.is_cuda:
            self._sampler = CuGraphRelationalSampler(data, self.time_columns)
        else:
            self._sampler = PyGLibRelationalSampler(data, self.time_columns)

    def __call__(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
        temporal_strategy: Literal["last", "uniform"] = "last",
    ) -> RelationalSamplerOutput:
        r"""Alias of :meth:`sample`."""
        return self.sample(
            task_table=task_table,
            task_link=task_link,
            num_neighbors=num_neighbors,
            task_time_column=task_time_column,
            temporal_strategy=temporal_strategy,
        )

    def sample(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
        temporal_strategy: Literal["last", "uniform"] = "last",
    ) -> RelationalSamplerOutput:
        r"""Sample :class:`RelatedTables` for task rows.

        Args:
            num_neighbors: Number of neighbors to sample per hop.
            task_table: Task table whose rows define the sampling queries.
            task_link: Link from ``task_table`` rows to a table in the
                relational data.
            task_time_column: Datetime column in ``task_table`` used as the
                query timestamp for temporal sampling.
            temporal_strategy: How temporal neighbors are selected. ``"last"``
                selects the most recent neighbors before each seed timestamp.
                ``"uniform"`` samples uniformly from neighbors before each seed
                timestamp.
        """
        if not isinstance(task_link, TaskLink):
            task_link = TaskLink.from_mapping(task_link)

        for table, columns in (
            (task_table, task_link.task_columns),
            (self.data.tables[task_link.table], task_link.table_columns),
        ):
            for column in columns:
                stype = table.stype(column)
                if stype != Stype.id:
                    raise ValueError(
                        f"Expected column '{column}' to have semantic type "
                        f"{str(Stype.id)!r} (got {str(stype)!r})"
                    )

        if task_time_column is not None:
            stype = task_table.stype(task_time_column)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected task time column to have semantic type "
                    f"{str(Stype.datetime)!r} (got {str(stype)!r})"
                )

        if task_table.dim() not in (2, 3):
            raise ValueError(
                f"Task table needs to be either 2D or 3D "
                f"(got {task_table.dim()})"
            )
        if task_table.dim() == 3:
            num_members, num_rows = task_table.size()[:2]
            task_table = cast(TableTensor, task_table.flatten(0, 1))
        else:
            num_members, num_rows = None, task_table.size(0)

        nodes = self._sampler.sample(
            task_table=task_table,
            task_link=task_link,
            num_neighbors=num_neighbors,
            task_time_column=task_time_column,
            temporal_strategy=temporal_strategy,
        )

        tables: dict[str, TableTensor | EnsembleTable] = {}
        for table_name, (example, index) in nodes.items():
            table = torch.cat(
                [
                    self.data.tables[table_name][index],
                    TableTensor(
                        columns={"id": (EXAMPLE_ID,)},
                        id=ColumnarTensor(
                            (
                                example
                                if num_members is None
                                else example % num_rows,
                            )
                        ),
                    ),
                ],
                dim=-1,
            )
            table = cast(TableTensor, table)

            if num_members is None:
                tables[table_name] = table
                continue

            member = example // num_rows
            tables[table_name] = EnsembleTable.from_tables(
                tables=[table[member == i] for i in range(num_members)],
                member_table_ids=range(num_members),
            )

        # Build composite keys for disjoint linkage across examples:
        relationships = tuple(
            Relationship(
                left_table=rel.left_table,
                left_columns=(*rel.left_columns, EXAMPLE_ID),
                right_table=rel.right_table,
                right_columns=(*rel.right_columns, EXAMPLE_ID),
            )
            for rel in self.data.relationships
            if rel.left_table in tables and rel.right_table in tables
        )

        task_link = TaskLink(
            task_columns=(*task_link.task_columns, EXAMPLE_ID),
            table=task_link.table,
            table_columns=(*task_link.table_columns, EXAMPLE_ID),
        )

        arange = torch.arange(task_table.size(0), device=task_table.device)
        if num_members is not None:
            arange = arange % num_rows
        task_table: Tensor = torch.cat(
            [
                task_table,
                TableTensor(
                    columns={"id": (EXAMPLE_ID,)},
                    id=ColumnarTensor((arange,)),
                ),
            ],
            dim=-1,
        )
        if num_members is not None:
            task_table = task_table.view(num_members, num_rows, -1)

        return RelationalSamplerOutput(
            task_table=cast(TableTensor, task_table),
            related_tables=RelatedTables(
                tables=tables,
                relationships=relationships,
                task_links=(task_link,),
            ),
        )
