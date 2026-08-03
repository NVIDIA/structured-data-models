# ruff: noqa: D102

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from sdm.cache import Cache
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.nn._memory import cuda_memory_budget

# Empirical upper bounds for transient aggregation work.
_AGGREGATION_EDGE_WORK_FACTOR = 4
_AGGREGATION_NODE_WORK_FACTOR = 6


def _aggregation_required_bytes(
    *,
    num_nodes: int,
    num_edges: int,
    value_bytes: int,
) -> int:
    return value_bytes * (
        _AGGREGATION_NODE_WORK_FACTOR * num_nodes
        + _AGGREGATION_EDGE_WORK_FACTOR * num_edges
    )


def _automatic_aggregation_work_byte_limit(
    x: Tensor,
    graph: HomogeneousGraph,
) -> int | None:
    if x.device.type != "cuda":
        return None

    available_bytes, _ = cuda_memory_budget(x.device)
    value_bytes = x.size(-1) * max(x.element_size(), 4)
    node_bytes = _aggregation_required_bytes(
        num_nodes=graph.num_nodes,
        num_edges=0,
        value_bytes=value_bytes,
    )
    if (
        _aggregation_required_bytes(
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            value_bytes=value_bytes,
        )
        <= available_bytes
    ):
        return None
    return max(available_bytes - node_bytes, 0)


def _aggregation_slices(
    *,
    colptr: Tensor,
    work_byte_limit: int,
    value_bytes: int,
) -> list[tuple[int, int, int, int]]:
    workptr = value_bytes * (
        _AGGREGATION_EDGE_WORK_FACTOR * colptr
        + _AGGREGATION_NODE_WORK_FACTOR * torch.arange(colptr.numel())
    )
    slices: list[tuple[int, int, int, int]] = []
    start = 0
    while start < colptr.numel() - 1:
        target = workptr[start] + work_byte_limit
        end = int(torch.searchsorted(workptr, target, right=True)) - 1
        end = max(start + 1, end)
        slices.append((start, end, int(colptr[start]), int(colptr[end])))
        start = end
    return slices


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

        aggregation_slices: list[tuple[int, int, int, int]] | None = None
        if (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            work_byte_limit = _automatic_aggregation_work_byte_limit(x, graph)
            if work_byte_limit is not None:
                aggregation_slices = _aggregation_slices(
                    colptr=graph.colptr.cpu(),
                    work_byte_limit=work_byte_limit,
                    value_bytes=x.size(-1) * max(x.element_size(), 4),
                )
        edge_emb = (
            edge_type_emb[graph.edge_type]
            if aggregation_slices is None
            else None
        )

        for i in range(num_hops):
            if aggregation_slices is None:
                assert edge_emb is not None
                x = self._aggregate(
                    src_x=self.src_lin(x)[graph.row] + edge_emb,
                    colptr=graph.colptr,
                    skip_x=self.skip_lin(x),
                )
            else:
                src_x = self.src_lin(x)
                skip_x = self.skip_lin(x)
                out = torch.empty_like(skip_x)
                for start, end, edge_start, edge_end in aggregation_slices:
                    out[start:end] = self._aggregate(
                        src_x=(
                            src_x[graph.row[edge_start:edge_end]]
                            + edge_type_emb[
                                graph.edge_type[edge_start:edge_end]
                            ]
                        ),
                        colptr=(
                            graph.colptr[start : end + 1] - graph.colptr[start]
                        ),
                        skip_x=skip_x[start:end],
                    )
                x = out

            if i == num_hops - 1:
                start = graph.start_node_offsets[readout_table]
                end = graph.end_node_offsets[readout_table]
                x = x[start:end][readout_index]

            x = F.gelu(self.norm(x))

        return self.out_norm(self.out_lin(x))

    def _aggregate(
        self,
        *,
        src_x: Tensor,
        colptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        h = torch.segment_reduce(
            src_x,
            offsets=colptr,
            reduce="sum",
            unsafe=True,
            initial=0,
        )
        out = skip_x + self.sum_lin(h)

        h = h / colptr.diff().clamp(min=1).view(-1, 1)
        out = out + self.avg_lin(h)

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
        out = out + self.std_lin(h)

        h = torch.segment_reduce(
            src_x, offsets=colptr, reduce="min", unsafe=True
        )
        h = torch.where(h.isinf(), 0.0, h)
        out = out + self.min_lin(h)

        h = torch.segment_reduce(
            src_x, offsets=colptr, reduce="max", unsafe=True
        )
        h = torch.where(h.isinf(), 0.0, h)
        return out + self.max_lin(h)
