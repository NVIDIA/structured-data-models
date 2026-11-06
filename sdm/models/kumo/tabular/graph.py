# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sdm.cache import Cache, KVCacheEntry


@dataclass
class _Graph:
    inputs: tuple[Tensor, ...]
    cache_tensors: list[Tensor]
    graph: torch.cuda.CUDAGraph
    output: Tensor


def _layout(tensor: Tensor) -> tuple[object, ...]:
    return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def _matmul_precision() -> str:
    precision = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if precision is not None:
        return precision
    return torch.get_float32_matmul_precision()


class _CudaGraphs:
    """Bounded replay storage for the tabular network, owned by one model."""

    def __init__(self, max_graphs: int, max_memory_bytes: int) -> None:
        self.max_graphs = max_graphs
        self.max_memory_bytes = max_memory_bytes
        self.states: dict[tuple[object, ...], _Graph | None] = {}
        self.memory_bytes = 0

    def clear(self) -> None:
        """Release captured graphs and their static buffers."""
        self.states.clear()
        self.memory_bytes = 0

    def __getstate__(self) -> dict[str, Any]:
        # CUDA execution objects are process-local; recapture after loading.
        return {
            "max_graphs": self.max_graphs,
            "max_memory_bytes": self.max_memory_bytes,
            "states": {},
            "memory_bytes": 0,
        }

    def __call__(
        self,
        forward: Callable[..., Tensor],
        x: Tensor,
        y: Tensor,
        categorical_mask: Tensor,
        cache: Cache,
    ) -> Tensor:
        """Replay compatible predictions, otherwise execute normally."""
        inputs = (x, y, categorical_mask)
        if (
            not x.is_cuda
            or torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or torch.cuda.get_allocator_backend() != "native"
        ):
            return forward(*inputs, cache=cache)
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                return forward(*inputs, cache=cache)
        # The network consumes K/V entries and cell imputation means, not
        # class vocabularies or other preprocessing/postprocessing metadata.
        cache = Cache(
            (key, value)
            for key, value in cache.items()
            if isinstance(value, KVCacheEntry) or key == "cell_impute_mean"
        ).freeze()
        tensors = tuple(cache._tensors())
        stream = torch.cuda.current_stream(x.device)
        key = (
            tuple(_layout(t) for t in inputs + tensors),
            stream.cuda_stream,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            _matmul_precision(),
            torch.is_inference_mode_enabled(),
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
            # sdpa_kernel(set_priority=True) has no public priority getter.
            tuple(torch._C._get_sdp_priority_order()),
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_accumulation,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        if key not in self.states:
            if len(self.states) >= self.max_graphs:
                return forward(*inputs, cache=cache)
            # Remember rejected shapes too, avoiding repeated capture attempts.
            self.states[key] = None
            size = sum(t.numel() * t.element_size() for t in inputs + tensors)
            if self.memory_bytes + size <= self.max_memory_bytes:
                state = self._capture(forward, inputs, cache)
                pools = {
                    tuple(s.graph.pool())
                    for s in [state]
                    + [s for s in self.states.values() if s is not None]
                }
                size += sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if tuple(segment.get("segment_pool_id", ())) in pools
                )
                # Pools already retained were counted in memory_bytes.
                buffers = sum(
                    t.numel() * t.element_size()
                    for s in self.states.values()
                    if s is not None
                    for t in (*s.inputs, *s.cache_tensors)
                )
                if size + buffers <= self.max_memory_bytes:
                    self.states[key] = state
                    self.memory_bytes = size + buffers
        state = self.states[key]
        if state is None:
            return forward(*inputs, cache=cache)
        for dst, src in zip(state.inputs, inputs):
            dst.copy_(src)
        # Prefetch can replace cache storage for every estimator. Captured
        # kernels always read these stable buffers, refreshed before replay.
        torch._foreach_copy_(state.cache_tensors, list(tensors))
        state.graph.replay()
        return state.output.clone()

    def _capture(
        self,
        forward: Callable[..., Tensor],
        inputs: tuple[Tensor, ...],
        cache: Cache,
    ) -> _Graph:
        with torch.cuda.device(inputs[0].device):
            static = tuple(t.clone() for t in inputs)
            cache = cache._apply_tensor(lambda t: t.clone())
            current = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            stream.wait_stream(current)
            # Do not capture temporary autocast weight-cache pointers.
            with torch.autocast(
                device_type="cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                dtype=torch.get_autocast_dtype("cuda"),
                cache_enabled=False,
            ):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward(*static, cache=cache)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = forward(*static, cache=cache)
                current.wait_stream(stream)
            return _Graph(
                inputs=static,
                cache_tensors=list(cache._tensors()),
                graph=graph,
                output=output,
            )
