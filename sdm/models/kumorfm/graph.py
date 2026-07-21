from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import Self

from sdm import RelatedTables, RelationalData, Relationship


@dataclass(frozen=True)
class HomogeneousGraph:  # noqa: D101
    row: Tensor
    colptr: Tensor
    edge_type: Tensor
    num_edge_types: int
    start_node_offsets: dict[str, int]
    end_node_offsets: dict[str, int]

    @classmethod
    def from_tables(  # noqa: D102
        cls,
        related_tables: RelatedTables,
        relationship_order: Sequence[Relationship] | None = None,
    ) -> Self:

        if relationship_order is None:
            relationship_order = related_tables.relationships
        relationship_to_index = {
            relationship: i
            for i, relationship in enumerate(relationship_order)
        }

        start = 0
        start_node_offsets: dict[str, int] = {}
        end_node_offsets: dict[str, int] = {}
        for table_name, table in tables.items():
            assert table.dim() == 2
            start_node_offsets[table_name] = start
            start += table.size(0)
            end_node_offsets[table_name] = start

        rows: list[Tensor] = []
        cols: list[Tensor] = []
        edge_types: list[Tensor] = []
        for rel, edge_index in zip(
            related_tables.relationships,
            RelationalData(
                tables=related_tables.tables,
                relationships=related_tables.relationships,
            ).edge_indices(),
        ):
            i = relationship_to_index[rel]
            row, col = edge_index
            row += start_node_offsets[rel.left_table]
            col += start_node_offsets[rel.right_table]
            edge_type = edge_index.new_full((edge_index.size(1),), 2 * i)
            rows.extend([row, col])
            cols.extend([col, row])
            edge_types.extend([edge_type, edge_type + 1])

        if len(rows) == 0:
            table = next(iter(tables.values()))
            row = torch.empty(0, dtype=torch.long, device=table.device)
            colptr = torch.zeros(
                start + 1, dtype=torch.long, device=table.device
            )
            edge_type = torch.empty(0, dtype=torch.long, device=table.device)
        else:
            row = torch.cat(rows, dim=0)
            col = torch.cat(cols, dim=0)
            edge_type = torch.cat(edge_types, dim=0)

            col, perm = col.sort()
            row = row[perm]
            edge_type = edge_type[perm]
            colptr = torch._convert_indices_from_coo_to_csr(
                col, start, out_int32=col.dtype != torch.int64
            )

        return cls(
            row=row,
            colptr=colptr,
            edge_type=edge_type,
            num_edge_types=2 * len(relationship_order),
            start_node_offsets=start_node_offsets,
            end_node_offsets=end_node_offsets,
        )
