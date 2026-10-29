# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equal dim-zero context gathering preserves ordered bytes and failures."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sdm.execution._collectives import Exchange


def journey(
    rank: int, world: int, directory: str, failure: str | None
) -> None:
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
        if failure is not None:
            failed_allocation(rank, world, failure)
            return
        exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
        gather_into = dist.all_gather_into_tensor
        gather_list = dist.all_gather
        for dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int64,
        ):
            for lengths, dim in (
                ((3,) * world, 0),
                ((3,) * world, -3),
                (tuple(range(1, world + 1)), 0),
                (tuple(range(world)), 0),
                ((0,) * world, 0),
                ((3,) * world, 1),
            ):
                axis = dim % 3
                fast = (
                    world > 1
                    and axis == 0
                    and lengths[0] > 0
                    and len(set(lengths)) == 1
                )
                for layout in ("contiguous", "strided", "mixed"):
                    shape = [2, 2, 4]
                    shape[axis] = sum(lengths)
                    expected = torch.arange(
                        torch.tensor(shape).prod().item(), dtype=dtype
                    ).reshape(shape)
                    local = expected.narrow(
                        axis, sum(lengths[:rank]), lengths[rank]
                    ).clone()
                    if layout == "strided" or (layout == "mixed" and rank % 2):
                        # Preserve values/shape while forcing strided input.
                        local = (
                            local.transpose(0, 2).contiguous().transpose(0, 2)
                        )
                    before = local.clone()
                    with (
                        patch.object(
                            dist, "all_gather_into_tensor", wraps=gather_into
                        ) as direct,
                        patch.object(
                            dist, "all_gather", wraps=gather_list
                        ) as general,
                    ):
                        actual = exchange.gather(local, lengths, dim)
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0
                    )
                    assert torch.equal(
                        actual.view(torch.uint8), expected.view(torch.uint8)
                    )
                    torch.testing.assert_close(local, before, rtol=0, atol=0)
                    assert actual.dtype == dtype
                    assert actual.is_contiguous()
                    assert direct.call_count == int(fast)
                    assert general.call_count == int(world > 1 and not fast)
                    if world == 1 and local.numel():
                        assert actual.data_ptr() != local.data_ptr()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def failed_allocation(rank: int, world: int, failure: str) -> None:
    assert dist.group.WORLD is not None
    exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
    local = torch.arange(24).reshape(3, 2, 4).transpose(1, 2)
    primary = MemoryError(f"rank-local {failure} failure")
    method = "contiguous" if failure == "input" else "new_empty"
    original = getattr(torch.Tensor, method)

    def allocate(tensor, *args, **kwargs):
        if rank == world - 1 and tensor is local:
            raise primary
        return original(tensor, *args, **kwargs)

    with (
        patch.object(torch.Tensor, method, allocate),
        patch.object(
            dist,
            "all_gather_into_tensor",
            side_effect=AssertionError("unsafe tensor collective"),
        ),
        pytest.raises((MemoryError, RuntimeError), match=failure) as caught,
    ):
        exchange.attempt(lambda: exchange.gather(local, (3,) * world, 0))
    if rank == world - 1:
        assert caught.value is primary
    with (
        patch.object(
            dist, "all_reduce", side_effect=AssertionError("late agreement")
        ),
        pytest.raises((MemoryError, RuntimeError), match=failure) as repeated,
    ):
        exchange.attempt(lambda: exchange.gather(local, (3,) * world, 0))
    assert repeated.value is caught.value
    dist.barrier()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_equal_and_general_gather(tmp_path: Path, world: int) -> None:
    if world == 1:
        journey(0, world, str(tmp_path), None)
    else:
        mp.spawn(journey, args=(world, str(tmp_path), None), nprocs=world)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("failure", ["input", "output"])
def test_nested_allocation_failure(
    tmp_path: Path, world: int, failure: str
) -> None:
    mp.spawn(journey, args=(world, str(tmp_path), failure), nprocs=world)
