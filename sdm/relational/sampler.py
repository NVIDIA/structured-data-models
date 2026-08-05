from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Self, cast

import torch
from torch import Tensor

from sdm import ColumnarTensor, Stype, TableTensor
from sdm.relational import (
    RelatedTables,
    RelationalData,
    Relationship,
    TaskLink,
)
from sdm.relational.join import join_index
from sdm.tensor.mixin import DeviceMixin

EXAMPLE_ID = "__example__"


@dataclass(frozen=True)
class TemporalSamplingConfig:
    r"""Configuration for temporal relational sampling.

    Args:
        time_columns: Mapping from table names to their datetime columns.
        strategy: How to select neighbors that satisfy the request cutoff.
    """

    time_columns: Mapping[str, str]
    strategy: Literal["uniform", "last"] = "last"

    def __post_init__(self) -> None:
        if len(self.time_columns) == 0:
            raise ValueError("Expected at least one time column")


def _validate_time_columns(
    data: RelationalData,
    time_columns: Mapping[str, str],
) -> None:
    for table_name, column_name in time_columns.items():
        stype = data.tables[table_name].stype(column_name)
        if stype != Stype.datetime:
            raise ValueError(
                f"Expected '{column_name}' in table '{table_name}' to "
                f"have semantic type '{Stype.datetime.value}' "
                f"(got '{stype.value}')"
            )


class _RelationalSamplerOutput(NamedTuple):
    task_table: TableTensor
    related_tables: RelatedTables


class RelationalSamplerOutput(_RelationalSamplerOutput, DeviceMixin):
    r"""Relational sampler output.

    Args:
        task_table: The task table.
        related_tables: The related tables for the task table.
    """

    def to(self, device: torch.device | str | None) -> Self:
        r""":meta private:"""  # noqa: D415
        return self.__class__(
            task_table=cast(TableTensor, self.task_table.to(device)),
            related_tables=self.related_tables.to(device),
        )

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415
        devices = list({self.task_table.device, self.related_tables.device})
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected 'task_table' and 'related_tables' to be on the "
                f"same device (got '{self.task_table.device}' and "
                f"'{self.related_tables.device}')"
            )
        return next(iter(devices))


class RelationalSampler:
    r"""Subgraph sampler over relational data.

    Args:
        data: The collection of named tables and their relationships.
        temporal: Temporal sampling configuration. A row in a time-aware table
            can only be sampled if its timestamp does not exceed the query
            timestamp.
    """

    def __init__(
        self,
        data: RelationalData,
        temporal: TemporalSamplingConfig | None = None,
    ) -> None:
        self.data = data
        self.temporal = temporal
        self.time_columns = (
            temporal.time_columns if temporal is not None else {}
        )
        _validate_time_columns(self.data, self.time_columns)

        self._row_dict: dict[tuple[str, str, str], Tensor] = {}
        self._colptr_dict: dict[tuple[str, str, str], Tensor] = {}
        self._time_dict: dict[str, Tensor] = {}
        for table_name, time_column in self.time_columns.items():
            time = self.data.tables[table_name][time_column].datetime
            self._time_dict[table_name] = time.squeeze(-1).contiguous()

        for i, (rel, edge_index) in enumerate(
            zip(self.data.relationships, self.data.edge_indices())
        ):
            edge_type = (rel.left_table, str(2 * i), rel.right_table)
            self._row_dict[edge_type], self._colptr_dict[edge_type] = _to_csc(
                edge_index=edge_index,
                num_dst_nodes=self.data.tables[rel.right_table].size(0),
                src_time=self._time_dict.get(rel.left_table),
            )
            edge_type = (rel.right_table, str(2 * i + 1), rel.left_table)
            self._row_dict[edge_type], self._colptr_dict[edge_type] = _to_csc(
                edge_index=edge_index.flip(0),
                num_dst_nodes=self.data.tables[rel.left_table].size(0),
                src_time=self._time_dict.get(rel.right_table),
            )

    def __call__(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
    ) -> RelationalSamplerOutput:
        r"""Alias of :meth:`sample`."""
        return self.sample(
            task_table=task_table,
            task_link=task_link,
            num_neighbors=num_neighbors,
            task_time_column=task_time_column,
        )

    def sample(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
    ) -> RelationalSamplerOutput:
        r"""Sample :class:`RelatedTables` for task rows.

        Args:
            num_neighbors: Number of neighbors to sample per hop.
            task_table: Task table whose rows define the sampling queries.
            task_link: Link from ``task_table`` rows to a table in the
                relational data.
            task_time_column: Datetime column in ``task_table`` used as the
                query timestamp for temporal sampling.
        """
        task_link = self._validate_sample_inputs(
            task_table=task_table,
            task_link=task_link,
            task_time_column=task_time_column,
        )

        try:
            import pyg_lib  # noqa
        except ImportError as e:
            torch_version = torch.__version__.split("+", maxsplit=1)[0]
            if not all(part.isdigit() for part in torch_version.split(".")):
                raise ImportError(
                    "No module named 'pyg_lib'. Pre-built pyg-lib wheels are "
                    "only published for stable PyTorch releases. Please "
                    "install a stable PyTorch release or build pyg-lib "
                    "from source (see 'https://github.com/pyg-team/pyg-lib' "
                    "for more information)"
                ) from e
            if torch.version.cuda is None:
                cuda_version = "cpu"
            else:
                cuda_version = f"cu{torch.version.cuda.replace('.', '')}"
            raise ImportError(
                f"No module named 'pyg_lib'. Please install it via "
                f"'pip install pyg-lib -f https://data.pyg.org/whl/"
                f"torch-{torch_version}+{cuda_version}.html' (see "
                "'https://github.com/pyg-team/pyg-lib' for more information)"
            ) from e

        if not task_table.is_cpu or not self.data.is_cpu:
            raise NotImplementedError(
                f"{self.__class__.__name__!r} requires input data on CPU"
            )

        # Resolve entity table node indices:
        task_index, seed = join_index(
            left_table=task_table,
            right_table=self.data.tables[task_link.table],
            left_keys=task_link.task_columns,
            right_keys=task_link.table_columns,
            device=task_table.device,
        )
        task_index, perm = task_index.sort()
        seed = seed[perm]

        expected = torch.arange(
            task_table.size(-2),
            dtype=task_index.dtype,
            device=task_index.device,
        )
        if not task_index.equal(expected):
            raise ValueError(
                f"Expected each task row to match exactly one row in "
                f"{task_link.table!r}"
            )

        if task_time_column is not None:
            seed_time = task_table[task_time_column].datetime.squeeze(-1)
        else:
            fill_value = torch.iinfo(torch.int64).max
            seed_time = torch.full_like(seed, fill_value)

        # Perform subgraph sampling:
        _, _, node_dict, *_ = torch.ops.pyg.hetero_neighbor_sample(
            node_types=list(self.data.tables),
            edge_types=list(self._colptr_dict),
            rowptr_dict={
                "__".join(edge_type): colptr
                for edge_type, colptr in self._colptr_dict.items()
            },
            col_dict={
                "__".join(edge_type): row
                for edge_type, row in self._row_dict.items()
            },
            seed_dict={task_link.table: seed},
            num_neighbors_dict={
                "__".join(edge_type): list(num_neighbors)
                for edge_type in self._row_dict
            },
            node_time_dict=self._time_dict,
            edge_time_dict=None,
            seed_time_dict={task_link.table: seed_time},
            edge_weight_dict=None,
            csc=True,
            replace=False,
            directed=True,
            disjoint=True,
            temporal_strategy=(
                self.temporal.strategy if self.temporal is not None else "last"
            ),
            return_edge_id=False,
        )

        nodes = {
            table_name: tuple(node.t().contiguous())
            for table_name, node in node_dict.items()
            if node.numel() > 0
        }
        return self._to_output(
            task_table=task_table,
            task_link=task_link,
            nodes=cast(dict[str, tuple[Tensor, Tensor]], nodes),
        )

    def _validate_sample_inputs(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        task_time_column: str | None,
    ) -> TaskLink:
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
                        f"'{Stype.id.value}' (got '{stype.value}')"
                    )

        if task_time_column is not None:
            stype = task_table.stype(task_time_column)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected task time column to have semantic type "
                    f"'{Stype.datetime.value}' (got '{stype.value}')"
                )
        return task_link

    def _to_output(
        self,
        task_table: TableTensor,
        task_link: TaskLink,
        nodes: Mapping[str, tuple[Tensor, Tensor]],
    ) -> RelationalSamplerOutput:
        tables: dict[str, Tensor] = {}
        for table_name, (example, index) in nodes.items():
            if index.numel() == 0:
                continue
            tables[table_name] = torch.cat(
                [
                    self.data.tables[table_name][index],
                    TableTensor(
                        columns={"id": (EXAMPLE_ID,)},
                        id=ColumnarTensor((example,)),
                    ),
                ],
                dim=-1,
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

        task_table: Tensor = torch.cat(
            [
                task_table,
                TableTensor(
                    columns={"id": (EXAMPLE_ID,)},
                    id=ColumnarTensor(
                        (
                            torch.arange(
                                task_table.size(0),
                                device=task_table.device,
                            ),
                        )
                    ),
                ),
            ],
            dim=-1,
        )

        return RelationalSamplerOutput(
            task_table=cast(TableTensor, task_table),
            related_tables=RelatedTables(
                tables=cast(dict[str, TableTensor], tables),
                relationships=relationships,
                task_links=(task_link,),
            ),
        )


def _to_csc(
    edge_index: Tensor,
    num_dst_nodes: int,
    src_time: Tensor | None = None,
) -> tuple[Tensor, Tensor]:

    if src_time is None:  # Sort primarily by destination node:
        perm = edge_index[1].argsort()
    else:  # Sort secondarily by source timestamp:
        perm = src_time[edge_index[0]].argsort()
        edge_index = edge_index[:, perm]
        perm = edge_index[1].argsort(stable=True)
    edge_index = edge_index[:, perm]

    row, col = edge_index
    colptr = torch._convert_indices_from_coo_to_csr(
        col, num_dst_nodes, out_int32=col.dtype != torch.int64
    )

    return row, colptr
