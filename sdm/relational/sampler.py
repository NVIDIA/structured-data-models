from collections.abc import Mapping, Sequence
from typing import cast

import pyarrow as pa
import torch
from torch import Tensor

from sdm import ColumnarTensor, Stype, TableTensor
from sdm.relational import (
    RelatedTables,
    RelationalData,
    Relationship,
    TaskLink,
)
from sdm.relational.data import LEFT_ROW_ID, RIGHT_ROW_ID

EXAMPLE_ID = "__example__"


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
            stype = self.data.tables[table_name].stype(column_name)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected '{column_name}' in table '{table_name}' to "
                    f"have semantic type '{Stype.datetime.value}' "
                    f"(got '{stype.value}')"
                )

        self._row_dict: dict[tuple[str, str, str], Tensor] = {}
        self._colptr_dict: dict[tuple[str, str, str], Tensor] = {}
        self._time_dict: dict[str, Tensor] = {}
        for table_name, time_column in self.time_columns.items():
            time = self.data.tables[table_name][time_column].datetime
            self._time_dict[table_name] = time.squeeze(-1).contiguous().cpu()

        for i, (rel, edge_index) in enumerate(
            zip(self.data.relationships, self.data.edge_indices())
        ):
            relation = f"relationship_{i}"
            edge_type = (rel.left_table, relation, rel.right_table)
            self._row_dict[edge_type], self._colptr_dict[edge_type] = _to_csc(
                edge_index=edge_index,
                num_dst_nodes=self.data.tables[rel.right_table].size(0),
                src_time=self._time_dict.get(rel.left_table),
            )
            edge_type = (
                rel.right_table,
                f"rev_{relation}",
                rel.left_table,
            )
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
    ) -> tuple[TableTensor, RelatedTables]:
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
    ) -> tuple[TableTensor, RelatedTables]:
        r"""Sample :class:`RelatedTables` for task rows.

        Args:
            num_neighbors: Number of neighbors to sample per hop.
            task_table: Task table whose rows define the sampling queries.
            task_link: Link from ``task_table`` rows to a table in the
                relational data.
            task_time_column: Datetime column in ``task_table`` used as the
                query timestamp for temporal sampling.

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
                        f"'{Stype.id.value}' (got '{stype.value}')"
                    )

        if task_time_column is not None:
            stype = task_table.stype(task_time_column)
            if stype != Stype.datetime:
                raise ValueError(
                    f"Expected task time column to have semantic type "
                    f"'{Stype.datetime.value}' (got '{stype.value}')"
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

        # Resolve entity table node indices:
        left = task_table[task_link.task_columns].to_arrow()
        left = left.append_column(
            LEFT_ROW_ID,
            pa.array(torch.arange(left.num_rows).numpy()),
        )
        right = self.data.tables[task_link.table][
            task_link.table_columns
        ].to_arrow()
        right = right.append_column(
            RIGHT_ROW_ID,
            pa.array(torch.arange(right.num_rows).numpy()),
        )
        joined = left.join(
            right,
            keys=list(task_link.task_columns),
            right_keys=list(task_link.table_columns),
            join_type="left outer",
        )
        joined = joined.select([LEFT_ROW_ID, RIGHT_ROW_ID])
        joined = joined.sort_by([(LEFT_ROW_ID, "ascending")])
        if len(joined) != left.num_rows or joined[RIGHT_ROW_ID].null_count > 0:
            raise ValueError(
                f"Expected each task row to match exactly one row in "
                f"'{task_link.table}'"
            )

        seed = torch.from_numpy(joined[RIGHT_ROW_ID].to_numpy())
        if task_time_column is not None:
            seed_time = task_table[task_time_column].datetime.squeeze(-1)
        else:
            fill_value = torch.iinfo(torch.int64).max
            seed_time = torch.full_like(seed, fill_value)

        # Perform subgraph sampling:
        edge_type_by_key = {
            "__".join(edge_type): edge_type for edge_type in self._row_dict
        }
        (
            row_dict,
            col_dict,
            node_dict,
            _,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        ) = torch.ops.pyg.hetero_neighbor_sample(
            node_types=list(self.data.tables),
            edge_types=list(self._colptr_dict),
            rowptr_dict={
                key: self._colptr_dict[edge_type]
                for key, edge_type in edge_type_by_key.items()
            },
            col_dict={
                key: self._row_dict[edge_type]
                for key, edge_type in edge_type_by_key.items()
            },
            seed_dict={task_link.table: seed},
            num_neighbors_dict={
                key: list(num_neighbors) for key in edge_type_by_key
            },
            node_time_dict=self._time_dict,
            edge_time_dict=None,
            seed_time_dict={task_link.table: seed_time},
            edge_weight_dict=None,
            csc=True,
            replace=False,
            directed=True,
            disjoint=True,
            temporal_strategy="last",
            return_edge_id=False,
        )
        node_index_dict, metadata = _convert_hetero_sample(
            row_dict=row_dict,
            col_dict=col_dict,
            node_dict=node_dict,
            num_sampled_nodes_dict=num_sampled_nodes_dict,
            num_sampled_edges_dict=num_sampled_edges_dict,
            edge_type_by_key=edge_type_by_key,
            seed_time=seed_time,
        )

        tables: dict[str, Tensor] = {}
        for table_name, index in node_index_dict.items():
            if index.numel() == 0:
                continue
            tables[table_name] = torch.cat(
                [
                    self.data.tables[table_name][index],
                    TableTensor(
                        columns={"id": (EXAMPLE_ID,)},
                        id=ColumnarTensor((metadata.batch_dict[table_name],)),
                    ),
                ],
                dim=-1,
            )

        # Build composite keys for disjoint linkage across examples:
        relationships = tuple(
            Relationship(
                left_table=rel.left_table,
                left_columns=(EXAMPLE_ID, *rel.left_columns),
                right_table=rel.right_table,
                right_columns=(EXAMPLE_ID, *rel.right_columns),
            )
            for rel in self.data.relationships
            if rel.left_table in tables and rel.right_table in tables
        )

        task_link = TaskLink(
            task_columns=(EXAMPLE_ID, *task_link.task_columns),
            table=task_link.table,
            table_columns=(EXAMPLE_ID, *task_link.table_columns),
        )

        task_table: Tensor = torch.cat(
            [
                task_table,
                TableTensor(
                    columns={"id": (EXAMPLE_ID,)},
                    id=ColumnarTensor((torch.arange(task_table.size(0)),)),
                ),
            ],
            dim=-1,
        )

        return cast(TableTensor, task_table), RelatedTables(
            tables=cast(dict[str, TableTensor], tables),
            relationships=relationships,
            task_links=(task_link,),
            metadata=metadata,
        )


def _convert_hetero_sample(
    *,
    row_dict: Mapping[str, Tensor],
    col_dict: Mapping[str, Tensor],
    node_dict: Mapping[str, Tensor],
    num_sampled_nodes_dict: Mapping[str, Sequence[int]],
    num_sampled_edges_dict: Mapping[str, Sequence[int]],
    edge_type_by_key: Mapping[str, tuple[str, str, str]],
    seed_time: Tensor,
) -> tuple[dict[str, Tensor], SampledGraphMetadata]:
    node_index_dict: dict[str, Tensor] = {}
    batch_dict: dict[str, Tensor] = {}
    for node_type, node in node_dict.items():
        if node.numel() == 0:
            batch_dict[node_type] = node.new_empty(0)
            node_index_dict[node_type] = node.new_empty(0)
            continue

        batch, node_index = node.t().contiguous()
        batch_dict[node_type] = batch
        node_index_dict[node_type] = node_index

    edge_index_dict = {
        edge_type_by_key[key]: torch.stack((row, col_dict[key]))
        for key, row in row_dict.items()
    }
    typed_num_sampled_edges_dict = {
        edge_type_by_key[key]: counts
        for key, counts in num_sampled_edges_dict.items()
    }
    metadata = SampledGraphMetadata(
        edge_index_dict=edge_index_dict,
        batch_dict=batch_dict,
        num_sampled_nodes_dict=num_sampled_nodes_dict,
        num_sampled_edges_dict=typed_num_sampled_edges_dict,
        seed_time=seed_time,
    )
    return node_index_dict, metadata


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
