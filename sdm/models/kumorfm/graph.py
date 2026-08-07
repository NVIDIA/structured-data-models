from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from sdm import Relationship, TableTensor
from sdm.relational.join import join_index


@dataclass(frozen=True)
class LayeredGraphLayer:  # noqa: D101
    row: Tensor
    colptr: Tensor
    edge_type: Tensor
    dst_index: Tensor | None


@dataclass(frozen=True)
class LayeredGraph:  # noqa: D101
    input_index: Tensor | None
    layers: tuple[LayeredGraphLayer, ...]
    shared_edge_type: Tensor | None
    output_index: Tensor
    num_edge_types: int


@dataclass(frozen=True)
class HomogeneousGraph:  # noqa: D101
    row: Tensor
    col: Tensor
    colptr: Tensor
    edge_type: Tensor
    num_edge_types: int
    start_node_offsets: dict[str, int]
    end_node_offsets: dict[str, int]

    @property
    def num_nodes(self) -> int:  # noqa: D102
        return self.colptr.numel() - 1

    @property
    def num_edges(self) -> int:  # noqa: D102
        return self.row.numel()

    def node_slice(self, name: str) -> slice:  # noqa: D102
        start = self.start_node_offsets[name]
        end = self.end_node_offsets[name]
        return slice(start, end)

    def layered(
        self,
        *,
        node_masks: Sequence[Tensor],
        readout_index: Tensor,
    ) -> LayeredGraph:
        r"""Compact cumulative node masks into message-passing layers."""
        if len(node_masks) == 0:
            raise ValueError("'node_masks' cannot be empty")

        output_nodes = node_masks[0].nonzero().flatten()
        output_map = self.row.new_full((self.num_nodes,), fill_value=-1)
        output_map[output_nodes] = torch.arange(
            output_nodes.numel(),
            dtype=self.row.dtype,
            device=self.row.device,
        )
        output_index = output_map[readout_index]

        layers = [
            self._layer(previous=previous, active=active)
            for previous, active in zip(
                reversed(node_masks[1:]),
                reversed(node_masks[:-1]),
                strict=True,
            )
        ]
        input_nodes = node_masks[-1].nonzero().flatten()
        input_index = (
            None if input_nodes.numel() == self.num_nodes else input_nodes
        )
        return LayeredGraph(
            input_index=input_index,
            layers=tuple(layers),
            shared_edge_type=None,
            output_index=output_index,
            num_edge_types=self.num_edge_types,
        )

    def full_layered(
        self,
        *,
        num_layers: int,
        readout_index: Tensor,
    ) -> LayeredGraph:
        r"""Represent full-graph message passing as repeated layers."""
        layer = LayeredGraphLayer(
            row=self.row,
            colptr=self.colptr,
            edge_type=self.edge_type,
            dst_index=None,
        )
        return LayeredGraph(
            input_index=None,
            layers=(layer,) * num_layers,
            shared_edge_type=self.edge_type,
            output_index=readout_index,
            num_edge_types=self.num_edge_types,
        )

    def _layer(
        self,
        previous: Tensor,
        active: Tensor,
    ) -> LayeredGraphLayer:
        previous_nodes = previous.nonzero().flatten()
        active_nodes = active.nonzero().flatten()

        if active_nodes.numel() == self.num_nodes:
            return LayeredGraphLayer(
                row=self.row,
                colptr=self.colptr,
                edge_type=self.edge_type,
                dst_index=None,
            )

        edge_mask = active[self.col]
        row = self.row[edge_mask]
        edge_type = self.edge_type[edge_mask]
        if previous_nodes.numel() == self.num_nodes:
            dst_index = active_nodes
        else:
            previous_map = self.row.new_full((self.num_nodes,), fill_value=-1)
            previous_map[previous_nodes] = torch.arange(
                previous_nodes.numel(),
                dtype=self.row.dtype,
                device=self.row.device,
            )
            row = previous_map[row]
            dst_index = previous_map[active_nodes]

        degree = self.colptr.diff()[active_nodes]
        colptr = torch.cat(
            [
                degree.new_zeros(1),
                degree.cumsum(dim=0, dtype=degree.dtype),
            ]
        )
        return LayeredGraphLayer(
            row=row,
            colptr=colptr,
            edge_type=edge_type,
            dst_index=dst_index,
        )

    @classmethod
    def from_tables(  # noqa: D102
        cls,
        tables: Mapping[str, TableTensor],
        relationships: Sequence[Relationship],
    ) -> Self:
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
        for i, rel in enumerate(relationships):
            if rel.left_table not in tables or rel.right_table not in tables:
                continue
            row, col = join_index(
                left_table=tables[rel.left_table],
                right_table=tables[rel.right_table],
                left_keys=rel.left_columns,
                right_keys=rel.right_columns,
                how="inner",
            )
            row += start_node_offsets[rel.left_table]
            col += start_node_offsets[rel.right_table]
            edge_type = row.new_full((row.size(0),), 2 * i)
            rows.extend([row, col])
            cols.extend([col, row])
            edge_types.extend([edge_type, edge_type + 1])

        if len(rows) == 0:
            table = next(iter(tables.values()))
            row = torch.empty(0, dtype=torch.long, device=table.device)
            col = torch.empty(0, dtype=torch.long, device=table.device)
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
            col=col,
            colptr=colptr,
            edge_type=edge_type,
            num_edge_types=2 * len(relationships),
            start_node_offsets=start_node_offsets,
            end_node_offsets=end_node_offsets,
        )
