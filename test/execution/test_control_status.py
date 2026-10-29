# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control status fast path must preserve mixed nested failure envelopes."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import group as group

from sdm.execution._collectives import Exchange


def journey(rank: int, world: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=25),
    )
    assert dist.group.WORLD is not None
    try:
        exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
        original = dist.all_gather_object
        with patch.object(dist, "all_gather_object", wraps=original) as gather:
            assert exchange.records(None) == [None] * world
            exchange.agree(None)
            assert exchange.attempt(lambda: None) is None
            assert gather.call_count == 0

            for count, payload in enumerate((0, False, [], ""), start=1):
                assert exchange.records(payload if rank == 0 else None) == [
                    payload
                ] + [None] * (world - 1)
                assert gather.call_count == count
            assert exchange.metadata({"owner": 0} if rank == 0 else None) == {
                "owner": 0
            }
            assert gather.call_count == 4
            assert exchange.records(rank) == list(range(world))
            assert gather.call_count == 5

        blank = Exchange(dist.group.WORLD, dist.group.WORLD)
        if rank == 0:
            empty_error = ValueError("")

            def fail_empty() -> None:
                raise empty_error

            with pytest.raises(ValueError, match=r"^$") as caught_empty:
                blank.attempt(fail_empty)
            assert caught_empty.value is empty_error
        else:
            with pytest.raises(RuntimeError) as caught_empty:
                blank.records("peer record")
            assert str(caught_empty.value) == ""

        failure = ValueError("original inner failure")

        def fail() -> None:
            raise failure

        with patch.object(dist, "all_gather_object", wraps=original) as gather:
            if rank == 0:
                with pytest.raises(
                    ValueError, match="original inner"
                ) as caught:
                    exchange.attempt(lambda: exchange.attempt(fail))
                assert caught.value is failure
            else:
                with pytest.raises(
                    RuntimeError, match="original inner"
                ) as caught:
                    exchange.records("peer record")
            assert gather.call_count == 1

        # An inner failure cannot cause the outer frame or a later call to
        # issue another control collective on any rank.
        with (
            patch.object(
                dist,
                "all_reduce",
                side_effect=AssertionError("late reduction"),
            ),
            patch.object(
                dist,
                "all_gather_object",
                side_effect=AssertionError("late object gather"),
            ),
        ):
            with pytest.raises(
                (ValueError, RuntimeError), match="original inner"
            ) as again:
                exchange.records(None)
            assert again.value is caught.value
            with pytest.raises(
                (ValueError, RuntimeError), match="original inner"
            ) as once_more:
                exchange.agree(None)
            assert once_more.value is caught.value
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_status_and_mixed_failure(tmp_path: Path, world: int) -> None:
    if world == 1:
        journey(0, world, str(tmp_path))
    else:
        mp.spawn(journey, args=(world, str(tmp_path)), nprocs=world)


@pytest.mark.parametrize("local_error", [False, True])
@pytest.mark.parametrize("phase", ["status", "object"])
def test_transport_failure_preserves_local_error(
    group, monkeypatch, local_error, phase
):
    exchange = Exchange(group, group)
    primary = ValueError("original local failure")
    secondary = RuntimeError("status transport failure")

    def fail_transport(*args, **kwargs):
        raise secondary

    monkeypatch.setattr(
        dist,
        "all_reduce" if phase == "status" else "all_gather_object",
        fail_transport,
    )

    def operation() -> None:
        if local_error:
            raise primary

    def invoke() -> object:
        if local_error:
            return exchange.attempt(operation)
        return exchange.records("payload")

    with pytest.raises((ValueError, RuntimeError)) as caught:
        invoke()
    assert caught.value is (primary if local_error else secondary)
    with pytest.raises((ValueError, RuntimeError)) as repeated:
        exchange.records(None)
    assert repeated.value is caught.value
