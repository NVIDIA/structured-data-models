from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import Self

from sdm import RelatedTables, TableTensor
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.relational.join import join_index


@dataclass(frozen=True)
class TaskGraph:  # noqa: D101
    x: TableTensor
    related_tables: RelatedTables
    graph: HomogeneousGraph
    readout_index: Tensor  # Entity-table rows ordered by task row.
    row_batches: dict[str, Tensor]  # Per table row-to-task assignment.
    num_hops: int

    @classmethod
    def from_input(  # noqa: D102
        cls,
        x: TableTensor,
        related_tables: RelatedTables,
        num_hops: int | None = None,
    ) -> Self:

        if x.dim() != 2:
            raise ValueError("Tables need to be two-dimensional")

        for table in related_tables.tables.values():
            if table.dim() != 2:
                raise ValueError("Tables need to be two-dimensional")

        if len(related_tables.task_links) != 1:
            raise ValueError(
                f"'KumoRFM' expects exactly one task link to an entity table "
                f"(got {len(related_tables.task_links)})"
            )

        graph = HomogeneousGraph.from_tables(
            tables=related_tables.tables,
            relationships=related_tables.relationships,
        )
        entity_table_name = related_tables.task_links[0].table
        entity_offset = graph.start_node_offsets[entity_table_name]

        # Map each task row to exactly one entity-table row:
        task_index, readout_index = join_index(
            left_table=x,
            right_table=related_tables.tables[entity_table_name],
            left_keys=related_tables.task_links[0].task_columns,
            right_keys=related_tables.task_links[0].table_columns,
            how="inner",
        )
        task_index, perm = task_index.sort()
        readout_index = readout_index[perm]
        global_readout_index = readout_index + entity_offset

        arange = torch.arange(
            x.size(-2),
            dtype=task_index.dtype,
            device=task_index.device,
        )
        if (
            not task_index.equal(arange)
            or readout_index.unique().numel() != readout_index.numel()
        ):
            raise ValueError(
                f"Expected each task row to match exactly one distinct row in "
                f"'{related_tables.task_links[0].table}'"
            )

        row_batch = readout_index.new_full((graph.num_nodes,), fill_value=-1)
        row_batch[global_readout_index] = task_index
        frontier = torch.zeros_like(row_batch, dtype=torch.bool)
        frontier[global_readout_index] = True

        # Propagate task assignment along graph edges.
        # NOTE This assumes neighborhoods do not overlap.
        propagated_hops = 0
        while True:
            if num_hops is not None and propagated_hops >= num_hops:
                break
            mask = frontier[graph.row] & (row_batch[graph.col] < 0)
            row = graph.row[mask]
            if row.numel() == 0:
                break
            col = graph.col[mask]

            row_batch[col] = row_batch[row]
            frontier.fill_(False)
            frontier[col] = True
            propagated_hops += 1

        return cls(
            x=x,
            related_tables=related_tables,
            graph=graph,
            readout_index=readout_index,
            row_batches={
                table_name: row_batch[graph.node_slice(table_name)]
                for table_name in related_tables.tables
            },
            num_hops=propagated_hops if num_hops is None else num_hops,
        )
