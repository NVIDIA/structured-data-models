"""Invariant graph neural network modules for homogeneous graphs."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear
from torch.utils.checkpoint import checkpoint


class InvariantGNN(torch.nn.Module):
    r"""Permutation-invariant message passing over homogeneous node tensors.

    Each forward pass assigns random unit embeddings to the supplied edge-type
    IDs and updates every node with sum, mean, standard-deviation, minimum, and
    maximum message reducers. Callers are responsible for preparing
    bidirectional edges, stable edge-type IDs, sampled-edge slicing,
    heterogeneous packing, and any readout selection.

    Args:
        channels: The number of input and output channels.
        dst_chunk_size: Maximum number of destination nodes to aggregate at
            once. Set to ``None`` to disable chunking. The chunked path reads
            pointer metadata on the CPU and is intended for eager execution.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        dst_chunk_size: int | None = 8192,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("`channels` must be positive")
        if dst_chunk_size is not None and dst_chunk_size <= 0:
            raise ValueError("`dst_chunk_size` must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.channels = channels
        self.dst_chunk_size = dst_chunk_size

        self.src_lin = Linear(channels, channels, **factory_kwargs)
        self.edge_type_lin = Linear(channels, channels, **factory_kwargs)

        self.skip_lin = Linear(channels, channels, **factory_kwargs)
        self.sum_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.avg_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.std_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.min_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.max_lin = Linear(channels, channels, bias=False, **factory_kwargs)

        self.norm = LayerNorm(channels, **factory_kwargs)
        self.act = GELU()

        self.post_lin = Linear(channels, channels, **factory_kwargs)
        self.post_norm = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        num_edge_types: int,
        num_hops: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Apply invariant message passing.

        Args:
            x: Node features with shape ``[N, C]``.
            edge_index: Source and destination node indices with shape
                ``[2, E]`` and dtype ``torch.long``.
            edge_type: Stable edge-type IDs with shape ``[E]`` and dtype
                ``torch.long``.
            num_edge_types: Number of edge types. IDs must be in
                ``[0, num_edge_types)``. Zero is valid when there are no edges.
            num_hops: Number of message-passing hops. Zero returns ``x``
                unchanged.
            generator: Optional random generator for edge-type embeddings.
                When omitted, this uses PyTorch's default random generator.

        Returns:
            Updated node features with shape ``[N, C]``.
        """
        self._validate_inputs(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=num_edge_types,
            num_hops=num_hops,
        )
        if num_hops == 0:
            return x

        dst, perm = edge_index[1].sort(stable=True)
        src = edge_index[0][perm]
        edge_type = edge_type[perm]

        counts = torch.bincount(dst, minlength=x.size(0))
        ptr = torch.cat([dst.new_zeros(1), counts.cumsum(dim=0)])
        edge_type_emb = self._make_edge_type_embeddings(
            num_edge_types=num_edge_types,
            reference=x,
            generator=generator,
        )

        for _ in range(num_hops):
            skip_x = self.skip_lin(x)
            src_x = self.src_lin(x)
            x = self._chunked_hop(
                src_x=src_x,
                edge_type_emb=edge_type_emb,
                src=src,
                edge_type=edge_type,
                ptr=ptr,
                skip_x=skip_x,
            )
            x = self.act(self.norm(x))

        return self.post_norm(self.post_lin(x))

    def _validate_inputs(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        num_edge_types: int,
        num_hops: int,
    ) -> None:
        if num_hops < 0:
            raise ValueError("`num_hops` must be non-negative")
        if num_edge_types < 0:
            raise ValueError("`num_edge_types` must be non-negative")
        if x.dim() != 2 or x.size(-1) != self.channels:
            raise ValueError(
                f"`x` must have shape [N, {self.channels}], got "
                f"{list(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("`x` must have a floating-point dtype")

        parameter = self.src_lin.weight
        if x.device != parameter.device:
            raise ValueError("`x` must be on the same device as the module")
        autocast_dtype = None
        if torch.is_autocast_enabled(x.device.type):
            autocast_dtype = torch.get_autocast_dtype(x.device.type)
        if x.dtype != parameter.dtype and x.dtype != autocast_dtype:
            raise ValueError("`x` must have the same dtype as the module")

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("`edge_index` must have shape [2, E]")
        if edge_index.dtype != torch.long:
            raise ValueError("`edge_index` must have dtype torch.long")
        if edge_index.device != x.device:
            raise ValueError("`edge_index` must be on the same device as `x`")

        if edge_type.dim() != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError("`edge_type` must have shape [E]")
        if edge_type.dtype != torch.long:
            raise ValueError("`edge_type` must have dtype torch.long")
        if edge_type.device != x.device:
            raise ValueError("`edge_type` must be on the same device as `x`")

        if edge_index.numel() > 0 and (
            bool((edge_index < 0).any())
            or bool((edge_index >= x.size(0)).any())
        ):
            raise ValueError("`edge_index` contains an out-of-range node ID")
        if edge_type.numel() > 0 and (
            bool((edge_type < 0).any())
            or bool((edge_type >= num_edge_types).any())
        ):
            raise ValueError(
                "`edge_type` contains an out-of-range edge-type ID"
            )

    def _make_edge_type_embeddings(
        self,
        num_edge_types: int,
        reference: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        if num_edge_types == 0:
            return reference.new_empty(0, reference.size(-1))

        edge_type_emb = torch.randn(
            (num_edge_types, reference.size(-1)),
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        edge_type_emb = F.normalize(edge_type_emb, dim=-1)
        return self.edge_type_lin(edge_type_emb)

    def _chunked_hop(
        self,
        src_x: Tensor,
        edge_type_emb: Tensor,
        src: Tensor,
        edge_type: Tensor,
        ptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        if self.dst_chunk_size is None or src_x.size(0) <= self.dst_chunk_size:
            return self._hop(src_x, edge_type_emb, src, edge_type, ptr, skip_x)

        # Reading pointer boundaries on the CPU makes chunking eager-only.
        ptr_cpu = ptr.cpu()
        outs: list[Tensor] = []
        out: Tensor | None = None
        requires_grad = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (src_x, edge_type_emb, skip_x)
        )
        if not requires_grad:
            out = torch.empty_like(skip_x)

        for start in range(0, src_x.size(0), self.dst_chunk_size):
            end = min(start + self.dst_chunk_size, src_x.size(0))
            edge_start = int(ptr_cpu[start])
            edge_end = int(ptr_cpu[end])
            args = (
                src_x,
                edge_type_emb,
                src[edge_start:edge_end],
                edge_type[edge_start:edge_end],
                ptr[start : end + 1] - ptr[start],
                skip_x[start:end],
            )
            if requires_grad:
                outs.append(checkpoint(self._hop, *args, use_reentrant=False))
                continue

            assert out is not None
            out[start:end] = self._hop(*args)

        if requires_grad:
            return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        assert out is not None
        return out

    def _hop(
        self,
        src_x: Tensor,
        edge_type_emb: Tensor,
        src: Tensor,
        edge_type: Tensor,
        ptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        messages = src_x[src] + edge_type_emb[edge_type]  # [E, C]
        empty = ptr.diff().unsqueeze(-1) == 0  # [N, 1]

        h_sum = torch.segment_reduce(messages, reduce="sum", offsets=ptr)
        out = skip_x + self.sum_lin(h_sum)

        h_mean = torch.segment_reduce(messages, reduce="mean", offsets=ptr)
        h_mean = h_mean.masked_fill(empty, 0.0)
        out = out + self.avg_lin(h_mean)

        h_second_moment = torch.segment_reduce(
            messages.square(),
            reduce="mean",
            offsets=ptr,
        )
        h_std = (h_second_moment - h_mean.square()).clamp(min=1e-5).sqrt()
        h_std = h_std.masked_fill(
            empty | (h_std <= math.sqrt(1e-5)),
            0.0,
        )
        out = out + self.std_lin(h_std)

        h_min = torch.segment_reduce(messages, reduce="min", offsets=ptr)
        h_min = h_min.masked_fill(empty, 0.0)
        out = out + self.min_lin(h_min)

        h_max = torch.segment_reduce(messages, reduce="max", offsets=ptr)
        h_max = h_max.masked_fill(empty, 0.0)
        return out + self.max_lin(h_max)
