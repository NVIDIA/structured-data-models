# ruff: noqa: D102

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear

from sdm._kernels import segment_multi_reduce
from sdm.cache import Cache
from sdm.models.nemotron.relational.graph import HomogeneousGraph
from sdm.nn.memory import cuda_memory_availability

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

    available_memory, _ = cuda_memory_availability(x.device)
    value_bytes = x.size(-1) * max(x.element_size(), 4)
    node_bytes = _aggregation_required_bytes(
        num_nodes=graph.num_nodes,
        num_edges=0,
        value_bytes=value_bytes,
    )
    required_bytes = _aggregation_required_bytes(
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
        value_bytes=value_bytes,
    )
    if required_bytes <= available_memory or available_memory <= node_bytes:
        return None
    return available_memory - node_bytes


def _aggregation_slices(
    *,
    colptr: Tensor,
    num_edges: int,
    work_byte_limit: int,
    value_bytes: int,
) -> list[tuple[int, int, int, int]]:
    num_nodes = colptr.numel() - 1
    if num_nodes == 0:
        return []

    total_work = value_bytes * (
        _AGGREGATION_EDGE_WORK_FACTOR * num_edges
        + _AGGREGATION_NODE_WORK_FACTOR * num_nodes
    )
    num_targets = (total_work - 1) // work_byte_limit
    if num_targets >= num_nodes:
        boundaries = torch.arange(
            num_nodes + 1,
            dtype=torch.int64,
            device=colptr.device,
        )
    else:
        workptr = value_bytes * (
            _AGGREGATION_EDGE_WORK_FACTOR * colptr.to(torch.int64)
            + _AGGREGATION_NODE_WORK_FACTOR
            * torch.arange(
                colptr.numel(),
                dtype=torch.int64,
                device=colptr.device,
            )
        )
        targets = work_byte_limit * torch.arange(
            1,
            num_targets + 1,
            dtype=torch.int64,
            device=colptr.device,
        )
        lower = torch.searchsorted(workptr, targets, right=True) - 1
        upper = torch.searchsorted(workptr, targets)
        # Isolate any destination whose own work exceeds the limit.
        boundaries = torch.cat(
            [
                torch.tensor(
                    [0, num_nodes],
                    dtype=torch.int64,
                    device=colptr.device,
                ),
                lower,
                upper,
            ]
        ).clamp_(0, num_nodes)
        boundaries = boundaries.unique(sorted=True)

    edge_boundaries = colptr[boundaries]
    plan = torch.stack(
        [
            boundaries[:-1],
            boundaries[1:],
            edge_boundaries[:-1],
            edge_boundaries[1:],
        ],
        dim=-1,
    )
    return [
        (start, end, edge_start, edge_end)
        for start, end, edge_start, edge_end in plan.cpu().tolist()
    ]


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
                    colptr=graph.colptr,
                    num_edges=graph.num_edges,
                    work_byte_limit=work_byte_limit,
                    value_bytes=x.size(-1) * max(x.element_size(), 4),
                )
        edge_attr = (
            edge_type_emb[graph.edge_type]
            if aggregation_slices is None
            else None
        )

        for i in range(num_hops):
            src_x = self.src_lin(x)
            skip_x = self.skip_lin(x)
            if aggregation_slices is None:
                assert edge_attr is not None
                x = self._aggregate(
                    src_x=src_x,
                    index=graph.row,
                    edge_attr=edge_attr,
                    colptr=graph.colptr,
                    skip_x=skip_x,
                )
            else:
                out = torch.empty_like(skip_x)
                for start, end, edge_start, edge_end in aggregation_slices:
                    out[start:end] = self._aggregate(
                        src_x=src_x,
                        index=graph.row[edge_start:edge_end],
                        edge_attr=edge_type_emb[
                            graph.edge_type[edge_start:edge_end]
                        ],
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
        index: Tensor,
        edge_attr: Tensor,
        colptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        total, mean, std, minimum, maximum = segment_multi_reduce(
            src=src_x,
            index=index,
            edge_attr=edge_attr,
            offsets=colptr,
        )
        return (
            skip_x
            + self.sum_lin(total)
            + self.avg_lin(mean)
            + self.std_lin(std)
            + self.min_lin(minimum)
            + self.max_lin(maximum)
        )
