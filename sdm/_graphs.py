# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class _Timing:
    # CUDA events around the device work of one call.
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass
class _Layout:
    # Timed calls whose device time has not been read yet.
    pending: list[_Timing] = field(default_factory=list)
    # Shortest device time among the eager calls read so far.
    eager_time: float = math.inf
    num_eager_times: int = 0
    graph: torch.cuda.CUDAGraph | None = None
    pool: torch.cuda.MemPool | None = None
    # Share of the device memory taken by the graph's pool.
    memory_share: float = 0.0
    inputs: tuple[Tensor, ...] = ()
    out: Tensor | None = None
    # Longest device time among the replays read so far.
    replay_time: float = 0.0
    num_replay_times: int = 0
    verified: bool = False
    # Whether calls of this layout run eagerly from now on.
    eager: bool = False


class GraphCache:
    r"""Replays CUDA graphs of repeated calls when they are worth it.

    A CUDA graph removes the overhead of launching kernels one by one, which
    pays off for calls made of many short kernels, while it keeps the memory
    of its intermediates allocated. Instead of guessing when, the cache
    measures both for each key and input layout: it times eager calls with
    CUDA events and, once two of them have finished, captures a CUDA graph
    into its own memory pool. Replays after the first one are timed as well.
    The graph is kept if two of them save a larger share of the fastest eager
    call's time than the share of device memory taken by its pool, and it is
    dropped together with its pool as soon as one does not. As the benefit
    shrinks when calls take longer, layouts whose eager calls take at least
    as long as one whose graph was dropped are not captured. Timings are read
    from completed events on later calls, so the cache never synchronizes. At
    most :obj:`max_size` layouts are remembered.

    Args:
        max_size: The number of layouts to remember.
    """

    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max_size
        self._layouts: OrderedDict[Hashable, _Layout] = OrderedDict()
        # Shortest eager device time of a layout whose graph was dropped.
        self._dropped_time = math.inf

    def __call__(
        self,
        fn: Callable[..., Tensor],
        *args: Tensor,
        key: Hashable,
    ) -> Tensor:
        key = (key, *((arg.size(), arg.dtype, arg.device) for arg in args))
        layout = self._layouts.pop(key, None) or _Layout()
        self._layouts[key] = layout
        while len(self._layouts) > self.max_size:
            self._layouts.popitem(last=False)

        if layout.graph is not None and not layout.verified:
            self._verify(layout)
        if layout.graph is not None:
            return self._replay(layout, args, timed=not layout.verified)

        if not layout.eager:
            for eager_time in _finished(layout.pending):
                layout.eager_time = min(layout.eager_time, eager_time)
                layout.num_eager_times += 1
            if layout.num_eager_times >= 2:
                if layout.eager_time < self._dropped_time:
                    self._capture(layout, fn, args)
                    # The first replay also uploads the graph, so it is not
                    # timed.
                    return self._replay(layout, args, timed=False)
                layout.eager = True

        if layout.eager:
            return fn(*args)
        return self._timed(layout, lambda: fn(*args))

    def _verify(self, layout: _Layout) -> None:
        for replay_time in _finished(layout.pending):
            layout.replay_time = max(layout.replay_time, replay_time)
            layout.num_replay_times += 1
        if layout.num_replay_times == 0:
            return
        time_share = 1 - layout.replay_time / layout.eager_time
        if time_share <= layout.memory_share:
            self._dropped_time = min(self._dropped_time, layout.eager_time)
            layout.graph, layout.pool = None, None
            layout.inputs, layout.out = (), None
            layout.pending.clear()
            layout.eager = True
        elif layout.num_replay_times >= 2:
            layout.verified = True
            layout.pending.clear()

    def _replay(
        self,
        layout: _Layout,
        args: tuple[Tensor, ...],
        *,
        timed: bool,
    ) -> Tensor:
        graph, out = layout.graph, layout.out
        assert graph is not None
        assert out is not None

        def replay() -> Tensor:
            for buffer, arg in zip(layout.inputs, args, strict=True):
                buffer.copy_(arg)
            graph.replay()
            return out.clone()

        if timed:
            return self._timed(layout, replay)
        return replay()

    def _timed(self, layout: _Layout, fn: Callable[[], Tensor]) -> Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        layout.pending.append(_Timing(start=start, end=end))
        return out

    def _capture(
        self,
        layout: _Layout,
        fn: Callable[..., Tensor],
        args: tuple[Tensor, ...],
    ) -> None:
        device = args[0].device
        inputs = tuple(arg.clone() for arg in args)
        layout.pending.clear()

        pool = torch.cuda.MemPool()
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
            graph.capture_begin(pool=pool)
            try:
                out = fn(*inputs)
            finally:
                graph.capture_end()
        torch.cuda.current_stream(device).wait_stream(stream)

        pool_size = sum(segment["total_size"] for segment in pool.snapshot())
        device_size = torch.cuda.get_device_properties(device).total_memory
        layout.graph, layout.pool = graph, pool
        layout.memory_share = pool_size / device_size
        layout.inputs, layout.out = inputs, out

    def __getstate__(self) -> dict[str, int]:
        return {"max_size": self.max_size}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(max_size=state["max_size"])


def _finished(timings: list[_Timing]) -> list[float]:
    # Pops the timings whose calls have finished and returns their device
    # times in milliseconds.
    times, running = [], []
    for timing in timings:
        if timing.end.query():
            times.append(timing.start.elapsed_time(timing.end))
        else:
            running.append(timing)
    timings[:] = running
    return times
