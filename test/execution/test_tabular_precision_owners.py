# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP16 native controls for owner-local/subchunk compatibility, CPU only."""

import copy
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture
from test.execution.test_tabular_precision import no_cuda as no_cuda
from test.execution.test_tabular_precision import recipe

from sdm.cache import Cache, KVCacheEntry
from sdm.execution import tabular_execution
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_rows import Rows
from sdm.nn import TransformerBlock


def owner_journey(rank, world, directory, task, policy, count):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=40),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, query = fixture(task)
        x, y = x[:count], y[:count]
        native = copy.deepcopy(model)
        before = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        batches = [query[:2], query[2:4], query[4:6], query[6:]]
        # Original modules, only CPU AUTO hardware leaf controlled. Native
        # and candidate use the same base limit; candidate may subdivide rows.
        with torch.autocast("cpu", dtype=torch.float16):
            with patch.object(
                TransformerBlock, "auto_batch_size_limit", return_value=8
            ):
                native.fit(x, y, recipe=recipe(), num_estimators=1)
            expected = [native.predict(batch) for batch in batches]
        run = tabular_execution(
            model,
            task=task,
            group=dist.group.WORLD,
            control_group=dist.group.WORLD,
            precision="float16",
            owner_local_frontend=policy[0],
            shard_row_workspace=policy[1],
        )
        for repeat in range(3):
            if repeat == 2:
                run.clear()
                assert run._row_cache is run._icl_cache is None
            with patch.object(
                TransformerBlock, "auto_batch_size_limit", return_value=8
            ):
                run.fit(
                    x if rank == 0 else None,
                    y if rank == 0 else None,
                    recipe=recipe(),
                )
            assert not torch.is_autocast_enabled("cpu")
            for cache in (run._row_cache, run._icl_cache):
                assert cache is not None
                for key in cache:
                    actual = Cache.__getitem__(cache, key)
                    full = native._cache[0][key]
                    if isinstance(actual, KVCacheEntry):
                        for left, right in zip(actual, full, strict=True):
                            assert left.dtype == right.dtype == torch.float16
                            torch.testing.assert_close(
                                left,
                                right.narrow(0, cache.start, cache.length),
                                atol=2e-6,
                                rtol=1e-5,
                            )
                    elif isinstance(actual, torch.Tensor):
                        assert actual.dtype == full.dtype == torch.float32
                        torch.testing.assert_close(
                            actual, full, atol=2e-6, rtol=1e-5
                        )
            outputs = run.predict_batches(batches if rank == 0 else None)
            if rank == 0:
                assert outputs is not None
                for actual, full in zip(outputs, expected, strict=True):
                    assert actual.columns == full.columns
                    assert (
                        actual.numerical.dtype
                        == full.numerical.dtype
                        == torch.float32
                    )
                    torch.testing.assert_close(
                        actual.numerical, full.numerical, atol=2e-6, rtol=1e-5
                    )
                    if task == "regression":
                        assert actual.size(-1) == 999
                    else:
                        assert torch.equal(
                            actual.numerical.argmax(-1),
                            full.numerical.argmax(-1),
                        )
            else:
                assert outputs is None
        run.close()
        assert model._execution is None
        assert dist.is_initialized()
        for key, value in model.state_dict().items():
            assert value.dtype == torch.float32
            assert torch.equal(value, before[key])
        with torch.autocast("cpu", dtype=torch.float16):
            with patch.object(
                TransformerBlock, "auto_batch_size_limit", return_value=8
            ):
                model.fit(x, y, recipe=recipe(), num_estimators=1)
            torch.testing.assert_close(
                model.predict(batches[0]).numerical,
                expected[0].numerical,
                atol=2e-6,
                rtol=1e-5,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "policy", [(False, False), (True, False), (True, True)]
)
def test_fp16_owner_policies(tmp_path, task, world, policy):
    mp.spawn(
        owner_journey,
        args=(world, str(tmp_path), task, policy, 11),
        nprocs=world,
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fp16_empty_owners(tmp_path, task):
    mp.spawn(
        owner_journey, args=(8, str(tmp_path), task, (True, True), 1), nprocs=8
    )


def row_worker(rank, world, directory, fail):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    assert dist.group.WORLD is not None
    try:
        model, _, _, _ = fixture("regression")
        encoder = next(iter(model.models.values())).row_embedding
        exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
            for count in (1, 29):
                raw = torch.arange(count * 3).float().reshape(count, 3).sin()
                rows = Rows(
                    encoder,
                    raw,
                    torch.zeros(count),
                    torch.zeros(3, dtype=torch.bool),
                    exchange,
                    1 << 30,
                    owner_local_frontend=True,
                    shard_row_workspace=True,
                )
                assert rows.dtype == torch.float16
                assert rows.mean.dtype == torch.float32
                block = encoder.row_blocks[-1]
                for tokens in (2, 5):
                    for separate in (False, True):
                        full = (
                            torch.arange(count * 5 * 8)
                            .float()
                            .reshape(count, 5, 8)
                            .cos()
                            .half()
                        )
                        expected = full.clone()
                        block(
                            query=expected[:, :tokens],
                            key_value=expected,
                            out=expected[:, :tokens],
                            batch_size_limit=8,
                        )
                        local = full[rows.start : rows.stop].clone()
                        out = (
                            local.new_empty(local.size(0), tokens, 8)
                            if separate
                            else local[:, :tokens]
                        )
                        observed = []
                        original = block._forward
                        error = RuntimeError("FP16 owner-local row failure")

                        def checked(
                            *,
                            query,
                            key_value,
                            out,
                            _observed=observed,
                            _error=error,
                            _original=original,
                            **kwargs,
                        ):
                            _observed.append(key_value.size(0))
                            assert (
                                key_value.dtype == out.dtype == torch.float16
                            )
                            assert key_value.size(0) <= max(1, 8 // world)
                            if fail and rank == world - 1:
                                raise _error
                            return _original(
                                query=query,
                                key_value=key_value,
                                out=out,
                                **kwargs,
                            )

                        with patch.object(block, "_forward", checked):
                            if fail:
                                with pytest.raises(
                                    RuntimeError,
                                    match="FP16 owner-local row failure",
                                ) as caught:
                                    rows.row(
                                        block,
                                        8,
                                        tokens,
                                        key_value=local,
                                        out=out,
                                    )
                                if rank == world - 1:
                                    assert caught.value is error
                                dist.barrier()
                                return
                            result = rows.row(
                                block, 8, tokens, key_value=local, out=out
                            )
                        assert result is out
                        assert bool(observed) == bool(local.size(0))
                        torch.testing.assert_close(
                            out,
                            expected[rows.start : rows.stop, :tokens],
                            atol=2e-6,
                            rtol=1e-5,
                        )
                        if separate:
                            assert torch.equal(
                                local, full[rows.start : rows.stop]
                            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("failure", [False, True])
def test_fp16_actual_inner_row_bound_alias_and_failure(
    tmp_path, world, failure
):
    mp.spawn(row_worker, args=(world, str(tmp_path), failure), nprocs=world)
