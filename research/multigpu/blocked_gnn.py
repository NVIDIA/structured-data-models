"""Research-only destination-blocked inference for the relational GNN."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sdm._kernels import segment_multi_reduce
from sdm._kernels.segment_multi_reduce import _triton_segment_multi_reduce
from sdm.cache import Cache
from sdm.models.kumo.relational.graph import HomogeneousGraph
from sdm.models.kumo.relational.invariant_gnn import InvariantGNN


def destination_statistics(
    src: Tensor,
    graph: HomogeneousGraph,
    edge_attr: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """Reduce complete destinations without changing their edge order."""
    offsets = graph.colptr[start : end + 1]
    index, edge_type = graph.row, graph.edge_type
    if src.is_cuda:
        if (
            _triton_segment_multi_reduce is None
            or src.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or edge_attr.dtype != src.dtype
        ):
            raise RuntimeError(
                "Blocked CUDA GNN requires native Triton reduction"
            )
        return _triton_segment_multi_reduce(
            src, index, edge_attr, offsets, edge_type
        )
    if src.device.type == "cpu":
        # Eager reduction materializes gathered messages; bound that gather
        # to this block on CPU. CUDA keeps absolute offsets and global views,
        # so the existing Triton path needs no scalar readback or edge copy.
        first, last = int(offsets[0]), int(offsets[-1])
        index, edge_type = index[first:last], edge_type[first:last]
        offsets = offsets - first
    return segment_multi_reduce(
        src=src,
        index=index,
        edge_attr=edge_attr,
        offsets=offsets,
        edge_type=edge_type,
    )


class BlockedInvariantGNN(nn.Module):
    """Reuse a fitted GNN, bounding each statistics buffer by destination rows.

    Args:
        block: Existing checkpoint module; parameters are shared, not copied.
        block_size: Maximum destination rows per statistics/projection block.

    This adapter is inference-only and does not alter the checkpoint format.
    CUDA memory bounds require SDM's existing Triton reduction dispatch.
    """

    def __init__(self, block: InvariantGNN, block_size: int = 16384) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block = block
        self.block_size = block_size

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
        """Preserve native edge randomness, cache ownership, and hop order."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Blocked GNN requires inference_mode or no_grad"
            )
        block = self.block
        readout = graph.node_slice(readout_table)
        if num_hops == 0:
            return x[readout][readout_index]

        if cache is None or cache.is_recording:
            edge_attr = block.get_edge_type_emb(
                graph.num_edge_types,
                dtype=x.dtype,
                generator=generator,
            )
            if cache is not None and cache.is_recording:
                cache["edge_type_emb"] = edge_attr
        else:
            edge_attr = cast(Tensor, cache["edge_type_emb"])

        for hop in range(num_hops):
            src = block.src_lin(x)
            output = block.skip_lin(x)
            for start in range(0, graph.num_nodes, self.block_size):
                end = min(start + self.block_size, graph.num_nodes)
                stats = destination_statistics(
                    src, graph, edge_attr, start, end
                )
                output[start:end] = torch.addmm(
                    output[start:end],
                    stats.flatten(1),
                    block.aggregation_lin.weight.T,
                )
                del stats
            del src
            x = output
            del output
            if hop == num_hops - 1:
                x = x[readout][readout_index]
            x = F.gelu(block.norm(x))
        return block.out_norm(block.out_lin(x))


def install_blocked_gnn(
    model: nn.Module, block_size: int = 16384
) -> nn.Module:
    """Install on a freshly loaded inner KumoRelational model before fit."""
    model.gnn = BlockedInvariantGNN(model.gnn, block_size)
    return model
