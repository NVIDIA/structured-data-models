# ruff: noqa: D102

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from sdm._kernels import segment_multi_reduce
from sdm.cache import Cache
from sdm.models.kumo.relational.graph import HomogeneousGraph


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

        self.aggregation_lin = Linear(
            5 * channels,
            channels,
            bias=False,
            **factory_kwargs,
        )

        self.norm = LayerNorm(channels, **factory_kwargs)

        self.out_lin = Linear(channels, channels, **factory_kwargs)
        self.out_norm = LayerNorm(channels, **factory_kwargs)

    def get_edge_type_emb(
        self,
        num_edge_types: int,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        edge_type_emb = torch.randn(
            (num_edge_types, self.edge_type_lin.weight.size(-1)),
            dtype=dtype,
            device=self.edge_type_lin.weight.device,
            generator=generator,
        )
        edge_type_emb = F.normalize(edge_type_emb, dim=-1)
        return self.edge_type_lin(edge_type_emb)

    def forward(
        self,
        x: Tensor,
        graph: HomogeneousGraph,
        *,
        readout_table: str,
        readout_index: Tensor,
        num_hops: int,
        cache: Cache | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:

        if num_hops == 0:
            start = graph.start_node_offsets[readout_table]
            end = graph.end_node_offsets[readout_table]
            return x[start:end][readout_index]

        if cache is None or cache.is_recording:
            edge_type_emb = torch.randn(
                (graph.num_edge_types, self.edge_type_lin.weight.size(-1)),
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            edge_type_emb = F.normalize(edge_type_emb, dim=-1)
            edge_type_emb = self.edge_type_lin(edge_type_emb)
            if cache is not None and cache.is_recording:
                cache["edge_type_emb"] = edge_type_emb
        else:
            edge_type_emb = cast(Tensor, cache["edge_type_emb"])

        for i in range(num_hops):
            stats = segment_multi_reduce(
                src=self.src_lin(x),
                index=graph.row,
                edge_attr=edge_type_emb,
                offsets=graph.colptr,
                edge_type=graph.edge_type,
            )
            x = self.skip_lin(x).addmm_(
                stats.flatten(1),
                self.aggregation_lin.weight.T,
            )

            if i == num_hops - 1:
                start = graph.start_node_offsets[readout_table]
                end = graph.end_node_offsets[readout_table]
                x = x[start:end][readout_index]

            x = F.gelu(self.norm(x))

        return self.out_norm(self.out_lin(x))
