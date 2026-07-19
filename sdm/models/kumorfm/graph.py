from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import Self

from sdm import RelatedTables, RelationalData


@dataclass(frozen=True)
class HomogeneousGraph:  # noqa: D101
    row: Tensor
    colptr: Tensor
    edge_type: Tensor
    num_edge_types: int
    start_node_offsets: dict[str, int]
    end_node_offsets: dict[str, int]

    @classmethod
    def from_related_tables(  # noqa: D102
        cls,
        related_tables: RelatedTables,
    ) -> Self:

        start = 0
        start_node_offsets: dict[str, int] = {}
        end_node_offsets: dict[str, int] = {}
        for table_name, table in related_tables.tables.items():
            assert table.dim() == 2
            start_node_offsets[table_name] = start
            start += table.size(0)
            end_node_offsets[table_name] = start

        rows: list[Tensor] = []
        cols: list[Tensor] = []
        edge_types: list[Tensor] = []
        for i, (rel, edge_index) in enumerate(
            zip(
                related_tables.relationships,
                RelationalData(
                    tables=related_tables.tables,
                    relationships=related_tables.relationships,
                ).edge_indices(),
            )
        ):
            row, col = edge_index
            row += start_node_offsets[rel.left_table]
            col += start_node_offsets[rel.right_table]
            edge_type = edge_index.new_full((edge_index.size(1),), 2 * i)
            rows.extend([row, col])
            cols.extend([col, row])
            edge_types.extend([edge_type, edge_type + 1])

        if len(rows) == 0:
            table = next(iter(related_tables.tables.values()))
            row = torch.empty(0, dtype=torch.long, device=table.device)
            colptr = torch.zeros(1, dtype=torch.long, device=table.device)
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
            num_edge_types=len(edge_types),
            start_node_offsets=start_node_offsets,
            end_node_offsets=end_node_offsets,
        )
