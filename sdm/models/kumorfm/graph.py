"""Graph layout helpers for KumoRFM."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class _HomogeneousGraph:
    edge_index: Tensor
    edge_type: Tensor
    colptr: Tensor
    offset_dict: dict[str, tuple[int, int]]


def _make_homogeneous_graph(
    x_dict: Mapping[str, Tensor],
    edge_index_dict: Mapping[tuple[str, str, str], Tensor],
) -> _HomogeneousGraph:
    start = 0
    offset_dict: dict[str, tuple[int, int]] = {}
    for table_name, table_x in x_dict.items():
        end = start + table_x.size(0)
        offset_dict[table_name] = (start, end)
        start = end

    rows: list[Tensor] = []
    cols: list[Tensor] = []
    edge_types: list[Tensor] = []
    for i, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
        src, _, dst = edge_type
        row = edge_index[0] + offset_dict[src][0]
        col = edge_index[1] + offset_dict[dst][0]
        edge_type_index = edge_index.new_full((edge_index.size(1),), 2 * i)
        rows.extend([row, col])
        cols.extend([col, row])
        edge_types.extend([edge_type_index, edge_type_index + 1])

    if rows:
        row = torch.cat(rows)
        col = torch.cat(cols)
        edge_type = torch.cat(edge_types)
    else:
        value = next(iter(x_dict.values()))
        row = torch.empty(0, dtype=torch.long, device=value.device)
        col = torch.empty(0, dtype=torch.long, device=value.device)
        edge_type = torch.empty(0, dtype=torch.long, device=value.device)

    col, perm = col.sort()
    row = row[perm]
    edge_type = edge_type[perm]
    colptr = torch._convert_indices_from_coo_to_csr(
        col, start, out_int32=col.dtype != torch.int64
    )

    return _HomogeneousGraph(
        edge_index=torch.stack((row, col)),
        edge_type=edge_type,
        colptr=colptr,
        offset_dict=offset_dict,
    )
