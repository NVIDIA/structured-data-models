# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Successful cache release and early rejection of versionless inference."""

import copy
import gc
import weakref
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture

import sdm.execution.tabular as tabular
import sdm.processing as sp
from sdm import Recipe
from sdm.execution import tabular_execution


def start(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )


def execution(model, task: str, flag: bool):
    assert dist.group.WORLD is not None
    return tabular_execution(
        model,
        task=task,
        group=dist.group.WORLD,
        control_group=dist.group.WORLD,
        owner_local_frontend=flag,
        shard_row_workspace=flag,
    )


def recipe() -> Recipe:
    return Recipe(
        target=sp.StypeDispatch(numerical=sp.Standardize()),
        output=[sp.AverageEstimators()],
    )


def cache_refs(run):
    # Owned entries only: replay getters would allocate gathered temporaries.
    return [
        weakref.ref(tensor)
        for cache in (run._row_cache, run._icl_cache)
        for tensor in cache._tensors()
    ]


def release_journey(rank: int, directory: str, task: str, flag: bool) -> None:
    start(rank, directory)
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        model, x, y, query = fixture(task)
        native = copy.deepcopy(model)
        native.fit(x, y, recipe=recipe(), num_estimators=1)
        expected = native.predict(query)
        run = execution(model, task, flag)

        def fit() -> None:
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe() if rank == 0 else None,
            )

        fit()
        refs = cache_refs(run)
        actual = run.predict(query if rank == 0 else None)
        if rank == 0:
            assert actual.columns == expected.columns
            torch.testing.assert_close(
                actual.numerical, expected.numerical, atol=2e-6, rtol=1e-5
            )
            if task == "regression":
                assert actual.numerical.shape == (7, 999)
            else:
                assert torch.equal(
                    actual.numerical.argmax(-1),
                    expected.numerical.argmax(-1),
                )
        del actual
        fit()
        assert all(ref() is None for ref in refs)
        refs = cache_refs(run)
        run.clear()
        assert all(ref() is None for ref in refs)
        assert run._row_cache is run._icl_cache is None
        assert model._execution is run
        fit()
        refs = cache_refs(run)
        run.close()
        assert all(ref() is None for ref in refs)
        assert model._execution is None
        run.close()
        # A healthy close leaves native model reuse and borrowed groups intact.
        model.fit(x, y, recipe=recipe(), num_estimators=1)
        reused = model.predict(query)
        torch.testing.assert_close(
            reused.numerical, expected.numerical, atol=2e-6, rtol=1e-5
        )
        dist.barrier()
    finally:
        if enabled:
            gc.enable()
        dist.destroy_process_group()


def inference_journey(
    rank: int, directory: str, operation: str, flag: bool
) -> None:
    start(rank, directory)
    try:
        model, x, y, query = fixture("regression")
        if operation == "constructor":
            # Only one rank enables inference mode; both must reject before
            # acquiring a model lease or entering fit/core/cache allocation.
            with (
                torch.inference_mode() if rank == 1 else nullcontext(),
                pytest.raises(
                    (ValueError, RuntimeError), match="inference_mode"
                ),
            ):
                execution(model, "regression", flag)
            assert model._execution is None
            assert model._cache is None
            run = execution(model, "regression", flag)
            run.close()
            dist.barrier()
            return

        run = execution(model, "regression", flag)
        if operation != "fit":
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe() if rank == 0 else None,
            )
        old_identity = run._cache_identity
        old_cache = model._cache, run._row_cache, run._icl_cache

        def invoke() -> None:
            if operation in ("fit", "refit"):
                run.fit(x if rank == 0 else None, y if rank == 0 else None)
            elif operation == "predict":
                run.predict(query if rank == 0 else None)
            else:
                getattr(run, operation)()

        with (
            patch.object(
                tabular,
                "Rows",
                side_effect=AssertionError("Unexpected frontend allocation"),
            ) as rows,
            torch.inference_mode() if rank == 1 else nullcontext(),
            pytest.raises(
                (ValueError, RuntimeError), match="inference_mode"
            ) as caught,
        ):
            invoke()
        assert rows.call_count == 0
        assert run._cache_identity is old_identity
        assert all(
            actual is original
            for actual, original in zip(
                (model._cache, run._row_cache, run._icl_cache), old_cache
            )
        )
        if operation == "fit":
            assert not run._fit_started
            assert old_cache == (None, None, None)
        assert model._execution is run
        for repeat in (run.clear, run.close, invoke):
            with pytest.raises(type(caught.value)) as again:
                repeat()
            assert again.value is caught.value
        dist.barrier()
    finally:
        dist.destroy_process_group()


def failed_journey(rank: int, directory: str, point: str, flag: bool) -> None:
    start(rank, directory)
    try:
        model, x, y, _ = fixture("regression")
        run = execution(model, "regression", flag)
        original_call = tabular.Rows.__call__
        original_icl = tabular.fit_icl
        rows_refs = []
        first = RuntimeError("rank-local submitted fit failure")

        def row_call(rows, module, *args, **kwargs):
            rows_refs.append(weakref.ref(rows))
            if point == "frontend" and rank == 1:
                raise first
            return original_call(rows, module, *args, **kwargs)

        def icl(*args, **kwargs):
            if point == "icl":

                def fail() -> None:
                    if rank == 1:
                        raise first

                run.exchange.attempt(fail)
            return original_icl(*args, **kwargs)

        with (
            patch.object(tabular.Rows, "__call__", row_call),
            patch.object(tabular, "fit_icl", icl),
            pytest.raises(
                RuntimeError, match="rank-local submitted fit failure"
            ) as caught,
        ):
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe() if rank == 0 else None,
            )
        if rank == 1:
            assert caught.value is first
        assert rows_refs
        gc.collect()
        # Do not break cycles or discard owners when fit did not complete.
        assert all(ref() is not None and ref().handlers for ref in rows_refs)
        assert model._execution is run
        for repeat in (run.clear, run.close):
            with pytest.raises(RuntimeError) as again:
                repeat()
            assert again.value is caught.value
        dist.barrier()
    finally:
        dist.destroy_process_group()


def inference_cleanup_journey(rank: int, directory: str, flag: bool) -> None:
    start(rank, directory)
    try:
        model, x, y, query = fixture("regression")
        native = copy.deepcopy(model)
        native.fit(x, y, recipe=recipe(), num_estimators=1)
        expected = native.predict(query)
        run = execution(model, "regression", flag)

        def fit() -> None:
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe() if rank == 0 else None,
            )

        fit()
        with torch.inference_mode() if rank == 1 else nullcontext():
            run.clear()
        assert run._failure is None
        assert run._cache_identity is None
        assert run._row_cache is run._icl_cache is None
        assert model._execution is run
        fit()
        actual = run.predict(query if rank == 0 else None)
        if rank == 0:
            assert actual.numerical.shape == (7, 999)
            torch.testing.assert_close(
                actual.numerical, expected.numerical, atol=2e-6, rtol=1e-5
            )
        with torch.inference_mode() if rank == 1 else nullcontext():
            run.close()
            run.close()
        assert run._failure is None
        assert model._execution is None
        assert run._row_cache is run._icl_cache is None
        model.fit(x, y, recipe=recipe(), num_estimators=1)
        reused = model.predict(query)
        torch.testing.assert_close(
            reused.numerical, expected.numerical, atol=2e-6, rtol=1e-5
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_immediate_cache_release(tmp_path: Path, task: str, flag: bool):
    mp.spawn(
        release_journey, args=(str(tmp_path), task, flag), nprocs=2, join=True
    )


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "operation", ["constructor", "fit", "refit", "predict"]
)
def test_early_inference_rejection(tmp_path: Path, operation: str, flag: bool):
    mp.spawn(
        inference_journey,
        args=(str(tmp_path), operation, flag),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("point", ["frontend", "icl"])
def test_failed_rows_remain_owned(tmp_path: Path, point: str, flag: bool):
    mp.spawn(
        failed_journey, args=(str(tmp_path), point, flag), nprocs=2, join=True
    )


@pytest.mark.parametrize("flag", [False, True])
def test_healthy_cleanup_under_inference_mode(tmp_path: Path, flag: bool):
    mp.spawn(
        inference_cleanup_journey,
        args=(str(tmp_path), flag),
        nprocs=2,
        join=True,
    )
