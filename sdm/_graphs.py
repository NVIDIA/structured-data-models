# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

import torch
from torch import Tensor

_Entry = tuple[torch.cuda.CUDAGraph, tuple[Tensor, ...], Tensor]


class GraphCache:
    r"""Replays CUDA graphs of repeated inference calls with equal inputs.

    Small inputs leave the GPU waiting for kernel launches. The first call of
    a key and input layout runs eagerly, the second captures a CUDA graph and
    later calls replay it. Captured graphs share one memory pool, and at most
    :obj:`max_size` of them are kept.

    Args:
        max_size: The number of keys to remember.
    """

    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max_size
        self._entries: OrderedDict[Hashable, _Entry | None] = OrderedDict()
        self._pool: Any = None

    def __call__(
        self,
        fn: Callable[..., Tensor],
        *args: Tensor,
        key: Hashable,
    ) -> Tensor:
        key = (key, *((arg.size(), arg.dtype, arg.device) for arg in args))
        if key not in self._entries:
            self._remember(key, None)
            return fn(*args)

        entry = self._entries[key]
        if entry is None:
            entry = self._capture(fn, args)
            self._remember(key, entry)
        self._entries.move_to_end(key)

        graph, inputs, out = entry
        for buffer, arg in zip(inputs, args, strict=True):
            buffer.copy_(arg)
        graph.replay()
        return out.clone()

    def _remember(self, key: Hashable, entry: _Entry | None) -> None:
        self._entries[key] = entry
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def _capture(
        self,
        fn: Callable[..., Tensor],
        args: tuple[Tensor, ...],
    ) -> _Entry:
        device = args[0].device
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        inputs = tuple(arg.clone() for arg in args)

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Autocast must not reuse casts made outside of the graph:
        with (
            torch.cuda.stream(stream),
            torch.autocast(
                device.type,
                enabled=torch.is_autocast_enabled(device.type),
                dtype=torch.get_autocast_dtype(device.type),
                cache_enabled=False,
            ),
        ):
            graph.capture_begin(pool=self._pool)
            try:
                out = fn(*inputs)
            finally:
                graph.capture_end()
        torch.cuda.current_stream(device).wait_stream(stream)
        return graph, inputs, out

    def __getstate__(self) -> dict[str, int]:
        return {"max_size": self.max_size}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(max_size=state["max_size"])
