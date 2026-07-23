# ruff: noqa: D102

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from sdm.cache import Cache
from sdm.models.kumorfm.graph import LayeredGraph


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
        graph: LayeredGraph,
        *,
        cache: Cache | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if graph.input_index is not None:
            x = x[graph.input_index]
        if len(graph.layers) == 0:
            return x[graph.output_index]

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

        for i, layer in enumerate(graph.layers):
            src_x = self.src_lin(x)[layer.row] + edge_type_emb[layer.edge_type]
            skip_x = x if layer.dst_index is None else x[layer.dst_index]
            x = self._aggregate(
                src_x=src_x,
                colptr=layer.colptr,
                skip_x=self.skip_lin(skip_x),
            )
            if i == len(graph.layers) - 1:
                x = x[graph.output_index]
            x = F.gelu(self.norm(x))

        return self.out_norm(self.out_lin(x))

    def _aggregate(
        self,
        src_x: Tensor,
        colptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        # Sum aggregation:
        h = torch.segment_reduce(
            src_x,
            offsets=colptr,
            reduce="sum",
            unsafe=True,
            initial=0,
        )
        x = skip_x + self.sum_lin(h)

        # Mean aggregation:
        h = h / colptr.diff().clamp(min=1).view(-1, 1)
        x = x + self.avg_lin(h)

        # Std aggregation:
        h = (
            torch.segment_reduce(
                src_x.square(),
                offsets=colptr,
                reduce="mean",
                unsafe=True,
                initial=0,
            )
            - h.square()
        )
        h = torch.where(h <= 1e-5, 0.0, h.clamp(min=1e-5).sqrt())
        x = x + self.std_lin(h)

        # Min aggregation:
        h = torch.segment_reduce(
            src_x, offsets=colptr, reduce="min", unsafe=True
        )
        h = torch.where(h.isinf(), 0.0, h)
        x = x + self.min_lin(h)

        # Max aggregation:
        h = torch.segment_reduce(
            src_x, offsets=colptr, reduce="max", unsafe=True
        )
        h = torch.where(h.isinf(), 0.0, h)
        return x + self.max_lin(h)
