# ruff: noqa: D102

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear


class InvariantGNN(torch.nn.Module):
    r"""Schema-agnostic message passing over heterogeneous graphs.

    Args:
        channels: The number of input and output channels.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.edge_type_lin = Linear(channels, channels, **factory_kwargs)

        self.skip_lin = Linear(channels, channels, **factory_kwargs)
        self.src_lin = Linear(channels, channels, **factory_kwargs)

        self.sum_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.avg_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.std_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.min_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.max_lin = Linear(channels, channels, bias=False, **factory_kwargs)

        self.norm = LayerNorm(channels, **factory_kwargs)

        self.out_lin = Linear(channels, channels, **factory_kwargs)
        self.out_norm = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],  # {name: [R, C]}
        edge_index_dict: Mapping[tuple[str, str, str], Tensor],
        readout_table: str,
        num_hops: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:  # [R, C]

        if num_hops == 0 or len(edge_index_dict) == 0:
            return x_dict[readout_table]

        start = 0
        offset_dict: dict[str, tuple[int, int]] = {}
        for table_name, table_x in x_dict.items():
            end = start + table_x.size(0)
            offset_dict[table_name] = (start, end)
            start = end

        if len(x_dict) == 1:
            x = next(iter(x_dict.values()))
        else:
            x = torch.cat(list(x_dict.values()), dim=0)

        rows: list[Tensor] = []
        cols: list[Tensor] = []
        edge_types: list[Tensor] = []
        for i, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
            src, _, dst = edge_type
            row = edge_index[0] + offset_dict[src][0]
            col = edge_index[1] + offset_dict[dst][0]
            edge_type = edge_index.new_full((edge_index.size(1),), 2 * i)
            rows.extend([row, col])
            cols.extend([col, row])
            edge_types.extend([edge_type, edge_type + 1])
        row = torch.cat(rows, dim=0)
        col = torch.cat(cols, dim=0)
        edge_type = torch.cat(edge_types, dim=0)
        del rows
        del cols
        del edge_types

        col, perm = col.sort()
        colptr = torch._convert_indices_from_coo_to_csr(
            col, x.size(0), out_int32=col.dtype != torch.int64
        )
        row = row[perm]
        edge_type = edge_type[perm]
        del col
        del perm

        edge_type_emb = torch.randn(
            (2 * len(edge_index_dict), x.size(-1)),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        edge_type_emb = F.normalize(edge_type_emb, dim=-1)
        edge_type_emb = self.edge_type_lin(edge_type_emb)[edge_type]
        del edge_type

        for i in range(num_hops):
            src_x = self.src_lin(x)[row] + edge_type_emb
            x = self.skip_lin(x)

            h = torch.segment_reduce(  # Sum aggregation:
                src_x, offsets=colptr, reduce="sum", unsafe=True, initial=0
            )
            x = x + self.sum_lin(h)

            h = h / colptr.diff().clamp(min=1).view(-1, 1)  # Mean aggregation:
            x = x + self.avg_lin(h)

            h = (  # Std aggregation:
                torch.segment_reduce(
                    src_x.square(),
                    offsets=colptr,
                    reduce="mean",
                    unsafe=True,
                    initial=0,
                )
                - h.square()
            )
            # Zero out (near-)zero variance segments. Compare against the
            # variance *before* taking the square root: in `float16`,
            # `clamp(0.0, min=1e-5).sqrt()` rounds up above
            # `math.sqrt(1e-5)`, so a post-`sqrt` comparison would fail to
            # zero out zero-variance segments:
            h = torch.where(h <= 1e-5, 0.0, h.clamp(min=1e-5).sqrt())
            x = x + self.std_lin(h)

            h = torch.segment_reduce(  # Min aggregation:
                src_x, offsets=colptr, reduce="min", unsafe=True
            )
            h = torch.where(h.isinf(), 0.0, h)
            x = x + self.min_lin(h)

            h = torch.segment_reduce(  # Max aggregation:
                src_x, offsets=colptr, reduce="max", unsafe=True
            )
            h = torch.where(h.isinf(), 0.0, h)
            x = x + self.max_lin(h)

            del h
            del src_x

            if i == num_hops - 1:
                start, end = offset_dict[readout_table]
                x = x[start:end]

            x = F.gelu(self.norm(x))

        return self.out_norm(self.out_lin(x))
