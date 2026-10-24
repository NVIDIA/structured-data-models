# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Budgeted ICL KV reuse with real CPU owners and public outputs."""

import copy
from datetime import timedelta
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture

import sdm.processing as sp
from sdm import Recipe
from sdm.cache import KVCacheEntry
from sdm.execution import tabular_execution


def recipe() -> Recipe:
    return Recipe(
        target=sp.StypeDispatch(numerical=sp.Standardize()),
        output=[sp.AverageEstimators()],
    )


def worker(
    rank: int,
    world: int,
    directory: str,
    task: str,
    precision: Literal["float32", "float16"],
    budget: int,
    mutation: str | None = None,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=35),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, query = fixture(task)
        native = copy.deepcopy(model)
        with torch.autocast(
            "cpu", dtype=torch.float16, enabled=precision == "float16"
        ):
            native.fit(x, y, recipe=recipe(), num_estimators=1)
            expected = native.predict(query[:2])
        run = tabular_execution(
            model,
            task=task,
            group=dist.group.WORLD,
            control_group=dist.group.WORLD,
            precision=precision,
            gathered_kv_cache_bytes=budget,
        )
        run.fit(
            x if rank == 0 else None,
            y if rank == 0 else None,
            recipe=recipe(),
        )
        cache = run._icl_cache
        assert cache is not None
        layout, needed = cache.replica_plan()
        assert layout
        assert needed > 1
        owned_bytes = cache.size()
        assert cache._replicas is None
        assert run.predict_batches([] if rank == 0 else None) == (
            [] if rank == 0 else None
        )
        assert cache._replicas is None

        with (
            patch.object(
                run.exchange, "records", wraps=run.exchange.records
            ) as records,
            patch.object(
                run.exchange, "gather", wraps=run.exchange.gather
            ) as gather,
        ):
            first = run.predict_batches([query[:2]] if rank == 0 else None)
            first_gathers = gather.call_count
            second = run.predict_batches([query[:2]] if rank == 0 else None)
            second_gathers = gather.call_count - first_gathers
            assert records.call_count == int(world > 1 and budget > 0)
            if world > 1 and budget >= needed:
                assert first_gathers == second_gathers + 2 * len(layout)
            else:
                assert first_gathers == second_gathers

        if rank == 0:
            assert first is not None
            assert second is not None
            for actual in (first[0], second[0]):
                assert actual.columns == expected.columns
                torch.testing.assert_close(
                    actual.numerical,
                    expected.numerical,
                    atol=2e-6,
                    rtol=1e-5,
                )
        if world > 1 and budget >= needed:
            assert cache._replicas is not None
            assert cache.size() == owned_bytes + needed
            # A hit has no per-layer Gloo handshake or tensor gather.
            with (
                patch.object(
                    run.exchange,
                    "records",
                    side_effect=AssertionError("per-hit Gloo record"),
                ),
                patch.object(
                    run.exchange,
                    "gather",
                    side_effect=AssertionError("per-hit tensor gather"),
                ),
            ):
                for key, _ in layout:
                    assert isinstance(cache[key], KVCacheEntry)
        else:
            assert cache._replicas is None
            assert cache.size() == owned_bytes

        if mutation is not None:
            if rank == 0:
                with torch.no_grad():
                    if mutation == "owner":
                        next(cache._tensors()).add_(1)
                    else:
                        assert cache._replicas is not None
                        next(iter(cache._replicas.values())).key.add_(1)
            with pytest.raises(RuntimeError, match="fitted cache changed"):
                run.predict_batches([query[:2]] if rank == 0 else None)
            assert run._icl_cache is cache
            return

        if world > 1 and budget >= needed:
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe(),
            )
            assert run._icl_cache is not cache
            assert run._icl_cache is not None
            assert run._icl_cache._replicas is None
            run.predict_batches([query[:2]] if rank == 0 else None)
            assert run._icl_cache._replicas is not None
        run.clear()
        assert run._icl_cache is None
        if world > 1 and budget >= needed:
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe(),
            )
            run.predict_batches([query[:2]] if rank == 0 else None)
            assert run._icl_cache is not None
            assert run._icl_cache._replicas is not None
        run.close()
        assert run._icl_cache is None
        assert model._execution is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2])
def test_float32_reuse(tmp_path: Path, task: str, world: int) -> None:
    args = (world, str(tmp_path), task, "float32", 1 << 20)
    if world == 1:
        worker(0, *args)
    else:
        mp.spawn(worker, args=args, nprocs=world)


@pytest.mark.parametrize("budget", [0, 1, 1 << 20])
def test_float16_and_budget(tmp_path: Path, budget: int) -> None:
    mp.spawn(
        worker,
        args=(2, str(tmp_path), "regression", "float16", budget),
        nprocs=2,
    )


@pytest.mark.parametrize("mutation", ["owner", "replica"])
def test_mutation_retires(tmp_path: Path, mutation: str) -> None:
    mp.spawn(
        worker,
        args=(
            2,
            str(tmp_path),
            "classification",
            "float32",
            1 << 20,
            mutation,
        ),
        nprocs=2,
    )


def mismatch(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    assert dist.group.WORLD is not None
    try:
        model, _, _, _ = fixture("regression")
        with pytest.raises(ValueError, match="policies differ"):
            tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                gathered_kv_cache_bytes=1 << 20 if rank == 0 else 0,
            )
        assert model._execution is None
    finally:
        dist.destroy_process_group()


def test_rank_budget_mismatch(tmp_path: Path) -> None:
    mp.spawn(mismatch, args=(str(tmp_path),), nprocs=2)
