# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit enqueue ordering with fake CUDA events, not CUDA performance."""

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import sdm.execution._tabular_icl as icl
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_cache import OwnedCache
from sdm.models.kumo.tabular.icl import ICLBlock


class Value:
    is_cuda = True
    device = torch.device("cuda:3")
    shape = (8, 4, 4)

    def __init__(self, calls: list[str], empty: bool = False) -> None:
        self.calls = calls
        self.empty = empty

    def numel(self) -> int:
        return 0 if self.empty else 32

    def size(self, axis: int) -> int:
        return 8 if axis == -2 else 4

    def __getitem__(self, key: object) -> "Value":
        return self

    def movedim(self, source: int, destination: int) -> "Value":
        self.calls.append("movedim")
        return self

    def contiguous(self) -> "Value":
        self.calls.append("copy")
        return self

    def new_empty(self, shape: tuple[int, ...]) -> "Value":
        return Value(self.calls)


def queue(
    monkeypatch: pytest.MonkeyPatch,
    window: int | None,
    world: int,
    count: int,
    fault: str | None = None,
) -> SimpleNamespace:
    calls: list[str] = []
    events: list[object] = []
    outstanding: list[int] = []
    snapshots: list[tuple[str, int, tuple[int, ...]]] = []
    primary = RuntimeError("first fit enqueue failure")

    class Event:
        def __init__(self) -> None:
            self.index = len(events)
            events.append(self)

        def record(self, stream: str) -> None:
            assert calls[-1] == "copy"
            assert stream == "stream:cuda:3"
            if fault == "record":
                raise primary
            outstanding.append(self.index)
            snapshots.append(("record", self.index, tuple(outstanding)))

        def synchronize(self) -> None:
            if fault == "wait":
                raise primary
            assert outstanding.pop(0) == self.index
            snapshots.append(("wait", self.index, tuple(outstanding)))

    def gather(received: Value, source: Value, *, group: object) -> None:
        if window is not None:
            assert len(outstanding) < window
        calls.append("gather")
        if fault == "gather":
            raise primary

    class Attention:
        query_scaling = None

        def __call__(self, *args: Value, **kwargs: object) -> Value:
            return Value(calls)

    class Model:
        def __call__(
            self,
            full: Value,
            labels: Value,
            *,
            cache: object,
            _attention: Callable[..., Value],
        ) -> Value:
            for _ in range(count):
                _attention(
                    Attention(),
                    query=Value(calls),
                    key=Value(calls),
                    value=Value(calls),
                )
            # Empty queries cannot issue a collective or queue an event.
            _attention(
                Attention(),
                query=Value(calls, empty=True),
                key=Value(calls),
                value=Value(calls),
            )
            return Value(calls)

    monkeypatch.setattr(
        icl,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Event=Event,
                current_stream=lambda device: f"stream:{device}",
            ),
            is_autocast_enabled=lambda device_type: False,
        ),
    )
    monkeypatch.setattr(
        icl, "dist", SimpleNamespace(all_gather_into_tensor=gather)
    )
    monkeypatch.setattr(icl, "sdpa_kernel", lambda backends: nullcontext())
    exchange = SimpleNamespace(
        world=world,
        rank=0,
        group=object(),
        gather=lambda *args: Value(calls),
        attempt=lambda operation: operation(),
    )

    def run() -> Value:
        # The fake CUDA protocol intentionally substitutes scalar test
        # doubles at this typed boundary; cast leaves their behavior intact.
        return cast(
            Value,
            icl.fit_icl(
                cast(ICLBlock, Model()),
                cast(torch.Tensor, Value(calls)),
                cast(torch.Tensor, Value(calls)),
                cast(OwnedCache, SimpleNamespace(lengths=(4,) * world)),
                cast(Exchange, exchange),
                window,
            ),
        )

    return SimpleNamespace(
        run=run,
        calls=calls,
        events=events,
        outstanding=outstanding,
        snapshots=snapshots,
        primary=primary,
    )


@pytest.mark.parametrize("window", [None, 1, 2, 4])
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 23])
def test_fit_head_gather_fifo(
    monkeypatch: pytest.MonkeyPatch,
    window: int | None,
    world: int,
    count: int,
) -> None:
    state = queue(monkeypatch, window, world, count)
    state.run()
    expected = count if world > 1 and window is not None else 0
    assert len(state.events) == expected
    assert state.calls.count("gather") == (count if world > 1 else 0)
    waits = [index for kind, index, _ in state.snapshots if kind == "wait"]
    assert waits == list(range(max(0, expected - (window or expected))))
    assert len(state.outstanding) == min(expected, window or expected)
    # The fit's outer completion fence owns the remaining tail; no extra
    # per-layer wait or device-wide synchronization is required here.


@pytest.mark.parametrize("fault", ["gather", "record", "wait"])
def test_fit_head_gather_stops_at_first_fault(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    state = queue(monkeypatch, window=2, world=2, count=23, fault=fault)
    with pytest.raises(RuntimeError, match="first fit enqueue") as caught:
        state.run()
    assert caught.value is state.primary
    assert state.calls.count("gather") == (2 if fault == "wait" else 1)
    assert len(state.events) == {"gather": 0, "record": 1, "wait": 2}[fault]
