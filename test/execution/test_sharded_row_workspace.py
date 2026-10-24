# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual row-call bounds and unchanged public CPU/Gloo semantics."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture, journey

from sdm.execution import tabular_execution
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_rows import Rows


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("count", [1, 11])
def test_public_sharded_workspace(
    tmp_path: Path, task: str, world: int, count: int
) -> None:
    args = (world, str(tmp_path), task, count > 1, None, True, count, True)
    mp.spawn(journey, args=args, nprocs=world, join=True)


def component_worker(
    rank: int, world: int, directory: str, failure: bool = False
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
        model, _, _, _ = fixture("regression")
        encoder = next(iter(model.models.values())).row_embedding
        with torch.no_grad():
            for total in (1, 11, 29):
                raw = torch.arange(total * 3).float().reshape(total, 3).sin()
                mask = torch.zeros(3, dtype=torch.bool)
                exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
                baseline = Rows(
                    encoder,
                    raw,
                    torch.zeros(total),
                    mask,
                    exchange,
                    1 << 30,
                    owner_local_frontend=True,
                )
                candidate = Rows(
                    encoder,
                    raw,
                    torch.zeros(total),
                    mask,
                    exchange,
                    1 << 30,
                    owner_local_frontend=True,
                    shard_row_workspace=True,
                )
                assert baseline.shard_row_workspace is False
                features, readouts, channels = (
                    candidate.columns,
                    candidate.readouts,
                    candidate.channels,
                )
                block = encoder.row_blocks[-1]
                for limit in (1, 3, 8, 31):
                    local_limit = max(1, limit // world)
                    segments = []
                    for start in range(0, total, limit):
                        left = max(start, candidate.start)
                        right = min(start + limit, total, candidate.stop)
                        segments.extend(
                            (begin, min(begin + local_limit, right))
                            for begin in range(left, right, local_limit)
                        )
                    for tokens in (readouts, features + readouts):
                        for separate in (False, True):
                            full = (
                                torch.arange(
                                    total * (features + readouts) * channels,
                                    dtype=torch.float32,
                                )
                                .reshape(total, features + readouts, channels)
                                .cos()
                            )
                            original = full[candidate.start : candidate.stop]
                            expected_kv, actual_kv = (
                                original.clone(),
                                original.clone(),
                            )
                            expected = (
                                torch.empty(original.size(0), tokens, channels)
                                if separate
                                else expected_kv[:, :tokens]
                            )
                            actual = (
                                torch.empty(original.size(0), tokens, channels)
                                if separate
                                else actual_kv[:, :tokens]
                            )
                            baseline.row(
                                block,
                                limit,
                                tokens,
                                key_value=expected_kv,
                                out=expected,
                            )
                            calls = []
                            original_forward = block._forward
                            error = RuntimeError(
                                "rank-local subdivided row failure"
                            )

                            def observed(
                                *,
                                query,
                                key_value,
                                out,
                                _calls=calls,
                                _error=error,
                                _forward=original_forward,
                                **kwargs,
                            ):
                                _calls.append(
                                    (
                                        key_value.clone(),
                                        out.stride(),
                                    )
                                )
                                if failure and rank == world - 1:
                                    raise _error
                                return _forward(
                                    query=query,
                                    key_value=key_value,
                                    out=out,
                                    **kwargs,
                                )

                            with patch.object(block, "_forward", observed):
                                if failure:
                                    with pytest.raises(
                                        RuntimeError, match=str(error)
                                    ) as caught:
                                        candidate.row(
                                            block,
                                            limit,
                                            tokens,
                                            key_value=actual_kv,
                                            out=actual,
                                        )
                                    if rank == world - 1:
                                        assert caught.value is error
                                    with pytest.raises(
                                        RuntimeError
                                    ) as repeated:
                                        exchange.attempt(lambda: None)
                                    assert repeated.value is caught.value
                                    dist.barrier()
                                    return
                                assert (
                                    candidate.row(
                                        block,
                                        limit,
                                        tokens,
                                        key_value=actual_kv,
                                        out=actual,
                                    )
                                    is actual
                                )
                            assert len(calls) == len(segments)
                            for (value, strides), (
                                left,
                                right,
                            ) in zip(calls, segments, strict=True):
                                assert 0 < value.size(0) <= local_limit
                                assert strides == actual.stride()
                                torch.testing.assert_close(
                                    value, full[left:right], atol=0, rtol=0
                                )
                            torch.testing.assert_close(
                                actual, expected, atol=2e-6, rtol=1e-5
                            )
                            torch.testing.assert_close(
                                actual_kv, expected_kv, atol=2e-6, rtol=1e-5
                            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_actual_max_row_bound_and_outputs(tmp_path: Path, world: int) -> None:
    mp.spawn(
        component_worker, args=(world, str(tmp_path)), nprocs=world, join=True
    )


@pytest.mark.parametrize("world", [2, 4])
def test_rank_local_row_failure(tmp_path: Path, world: int) -> None:
    mp.spawn(
        component_worker,
        args=(world, str(tmp_path), True),
        nprocs=world,
        join=True,
    )


def policy_worker(rank: int, world: int, directory: str) -> None:
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
        model, x, y, query = fixture("regression")
        for frontend in [False, True] if world == 3 else [False]:
            with pytest.raises(
                ValueError, match="requires owner-local frontend and P1/2/4"
            ):
                tabular_execution(
                    model,
                    task="regression",
                    group=dist.group.WORLD,
                    control_group=dist.group.WORLD,
                    owner_local_frontend=frontend,
                    shard_row_workspace=True,
                )
            assert model._execution is None
        if world == 3:
            dist.barrier()
            return
        with pytest.raises(ValueError, match="policies differ"):
            tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                owner_local_frontend=True,
                shard_row_workspace=bool(rank),
            )
        assert model._execution is None
        run = tabular_execution(
            model,
            task="regression",
            group=dist.group.WORLD,
            control_group=dist.group.WORLD,
            owner_local_frontend=True,
            shard_row_workspace=True,
        )
        run.fit(x if rank == 0 else None, y if rank == 0 else None)
        cache = run._row_cache
        run.shard_row_workspace = False
        with pytest.raises(RuntimeError, match="policy changed") as caught:
            run.predict(query if rank == 0 else None)
        run.shard_row_workspace = True
        for operation in (run.clear, run.close):
            with pytest.raises(RuntimeError) as repeated:
                operation()
            assert repeated.value is caught.value
            assert run._row_cache is cache
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_workspace_admission_and_policy(tmp_path: Path, world: int) -> None:
    mp.spawn(
        policy_worker, args=(world, str(tmp_path)), nprocs=world, join=True
    )
