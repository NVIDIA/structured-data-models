# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU dtype/protocol checks, not CUDA FP16 numerical qualification."""

import copy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture
from test.execution.test_tabular import group as group

import sdm
import sdm.processing as sp
from sdm import Recipe, Task
from sdm.cache import Cache, KVCacheEntry
from sdm.execution import tabular_execution
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_icl import workspace_bytes
from sdm.execution._tabular_rows import Rows
from sdm.nn import TransformerBlock


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU source test attempted CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def make_run(model, group, *, task="regression", precision="float16"):
    return tabular_execution(
        model, task=task, group=group, control_group=group, precision=precision
    )


def recipe():
    return Recipe(
        target=sp.StypeDispatch(numerical=sp.Standardize()),
        output=[sp.AverageEstimators()],
    )


def journey(rank, world, directory, task, mismatch=False):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=15),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, query = fixture(task)
        if mismatch:
            with pytest.raises(ValueError, match="policies differ"):
                make_run(
                    model,
                    dist.group.WORLD,
                    task=task,
                    precision="float32" if rank else "float16",
                )
            assert model._execution is None
            return
        native = copy.deepcopy(model)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        batches = [query[:2], query[2:4], query[4:6], query[6:]]
        with torch.autocast("cpu", dtype=torch.float16):
            native.fit(x, y, recipe=recipe(), num_estimators=1)
            expected = [native.predict(q) for q in batches]
        run = make_run(model, dist.group.WORLD, task=task)
        # Neither owner nor workers require an ambient autocast context.
        for _ in range(2):
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe(),
            )
            assert not torch.is_autocast_enabled("cpu")
            for cache in (run._row_cache, run._icl_cache):
                for value in cache._items.values():
                    if isinstance(value, KVCacheEntry):
                        assert (
                            value.key.dtype
                            == value.value.dtype
                            == torch.float16
                        )
                    elif isinstance(value, torch.Tensor):
                        assert value.dtype == torch.float32  # Native mean.
            actual = run.predict_batches(batches if rank == 0 else None)
            if rank == 0:
                for left, right in zip(actual, expected, strict=True):
                    assert left.columns == right.columns
                    assert (
                        left.numerical.dtype
                        == right.numerical.dtype
                        == torch.float32
                    )
                    assert torch.isfinite(left.numerical).all()
                    torch.testing.assert_close(
                        left.numerical, right.numerical, atol=2e-6, rtol=1e-5
                    )
                    if task == "regression":
                        assert left.numerical.shape[-1] == 999
        run.clear()
        assert run._icl_cache is run._row_cache is None
        run.close()
        assert dist.is_initialized()
        assert model._execution is None
        for key, value in model.state_dict().items():
            assert value.dtype == torch.float32
            assert torch.equal(value, weights[key])
        # Healthy release permits original native caller use, including AMP.
        with torch.autocast("cpu", dtype=torch.float16):
            model.fit(x, y, recipe=recipe(), num_estimators=1)
            torch.testing.assert_close(
                model.predict(batches[0]).numerical,
                expected[0].numerical,
                atol=2e-6,
                rtol=1e-5,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2])
def test_actual_cpu_public_amp(tmp_path, task, world):
    if world == 1:
        journey(0, world, str(tmp_path), task)
    else:
        mp.spawn(journey, args=(world, str(tmp_path), task), nprocs=world)


def test_precision_must_agree_across_ranks(tmp_path):
    mp.spawn(journey, args=(2, str(tmp_path), "regression", True), nprocs=2)


@pytest.mark.parametrize("limit", [2, 65_535])
def test_native_amp_inducing_residual_is_fp32(group, limit):
    model, x, y, _ = fixture("regression")
    y = y.squeeze(-1)  # Original task core's scalar target seam.
    encoder = model.models[Task.regression].row_embedding
    mask = torch.zeros(x.size(1), dtype=torch.bool)
    seen = []
    handles = [
        block.inducing_block.register_forward_hook(
            lambda module, args, output: seen.append(output.dtype)
        )
        for block in encoder.col_blocks
    ]
    try:
        # Mocked CPU AUTO hardware leaf only; genuine modules and raw data.
        with (
            torch.no_grad(),
            torch.autocast("cpu", dtype=torch.float16),
            patch.object(
                TransformerBlock, "auto_batch_size_limit", return_value=limit
            ),
        ):
            native_cache, owned_cache = Cache(), Cache()
            expected = encoder(x, y, mask, cache=native_cache)
            rows = Rows(encoder, x, y, mask, Exchange(group, group), 8 << 30)
            assert rows.dtype == torch.float16
            assert rows.mean.dtype == torch.float32
            actual = encoder(x, y, mask, cache=owned_cache, _runner=rows)
        assert seen
        assert set(seen) == {torch.float32}
        assert actual.dtype == expected.dtype == torch.float16
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        for key, value in native_cache.items():
            other = owned_cache[key]
            if isinstance(value, KVCacheEntry):
                assert isinstance(other, KVCacheEntry)
                for left, right in zip(value, other, strict=True):
                    assert left.dtype == right.dtype == torch.float16
                    torch.testing.assert_close(
                        left, right, atol=2e-6, rtol=1e-5
                    )
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("precision", ["bfloat16", "half", None])
def test_unsupported_precision_rejected_before_lease(group, precision):
    model, _, _, _ = fixture("regression")
    with pytest.raises(ValueError, match="precision must be"):
        make_run(model, group, precision=precision)
    assert model._execution is None


def test_conflicting_ambient_rejected(group):
    model, _, _, _ = fixture("regression")
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="Ambient autocast differs"),
    ):
        make_run(model, group)
    assert model._execution is None


def test_conflicting_ambient_after_fit_retires(group):
    model, x, y, query = fixture("regression")
    run = make_run(model, group)
    run.fit(x, y, recipe=recipe())
    cache = run._icl_cache
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="Ambient autocast differs") as caught,
    ):
        run.predict(query)
    with pytest.raises(ValueError, match="Ambient autocast differs") as again:
        run.close()
    assert caught.value is again.value
    assert run._icl_cache is cache


@pytest.mark.parametrize("kind", ["weight", "cache"])
def test_amp_generation_version_guard_remains(group, kind):
    model, x, y, query = fixture("regression")
    run = make_run(model, group)
    run.fit(x, y, recipe=recipe())
    value = (
        next(model.parameters())
        if kind == "weight"
        else next(run._icl_cache._tensors())
    )
    with torch.no_grad():
        value.add_(1)
    with pytest.raises(RuntimeError, match="changed") as caught:
        run.predict(query)
    with pytest.raises(RuntimeError, match="changed") as again:
        run.close()
    assert caught.value is again.value
    assert model._execution is run


def test_fp16_does_not_admit_half_stored_model(group):
    model, _, _, _ = fixture("regression")
    model.half()
    with pytest.raises(ValueError, match="FP32 device"):
        make_run(model, group)
    assert model._execution is None


def test_source_origins_are_this_external_copy():
    source = Path(__file__).resolve().parents[2]
    assert Path(sdm.__file__).resolve().is_relative_to(source)
    assert Path(fixture.__code__.co_filename).resolve().is_relative_to(source)


def test_matching_ambient_restored(group):
    model, x, y, _ = fixture("regression")
    with torch.autocast("cpu", dtype=torch.float16, cache_enabled=False):
        run = make_run(model, group)
        run.fit(x, y, recipe=recipe())
        assert torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_cache_enabled()
        run.close()
    assert not torch.is_autocast_enabled("cpu")


def test_precision_mutation_retires_fitted_generation(group):
    model, x, y, query = fixture("regression")
    run = make_run(model, group)
    run.fit(x, y, recipe=recipe())
    cache = run._icl_cache
    run.precision = "float32"
    with pytest.raises(RuntimeError, match="policy changed") as caught:
        run.predict(query)
    with pytest.raises(RuntimeError, match="policy changed") as repeated:
        run.close()
    assert caught.value is repeated.value
    assert run._icl_cache is cache
    assert model._execution is run
    assert not torch.is_autocast_enabled("cpu")


def test_failure_restores_amp_and_preserves_primary(group):
    model, x, y, _ = fixture("regression")
    run = make_run(model, group)
    primary = RuntimeError("injected neural failure")
    original = run._fit_core

    def fail(*args, **kwargs):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.float16
        raise primary

    run._fit_core = fail
    with pytest.raises(RuntimeError, match="injected neural") as caught:
        run.fit(x, y, recipe=recipe())
    run._fit_core = original
    assert caught.value is primary
    assert not torch.is_autocast_enabled("cpu")
    with pytest.raises(RuntimeError, match="injected neural") as repeated:
        run.close()
    assert repeated.value is primary
    assert model._execution is run


def test_transport_budget_uses_cached_dtype_not_halved_attention(group):
    model, x, y, _ = fixture("regression")
    run = make_run(model, group)
    run.fit(x, y, recipe=recipe())
    cache, block = run._icl_cache, run._core.icl_block
    half = workspace_bytes(block, 3, cache)
    original = dict(cache._items)
    cache._items = {
        key: KVCacheEntry(*(t.float() for t in pair))
        for key, pair in original.items()
    }
    full = workspace_bytes(block, 3, cache)
    cache._items = original
    compact = 11 * block.kv_heads * 2 * block.layers[0].attn.head_dim
    assert full - half == compact * 2
    assert half > full // 2
    run.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_fp32_default_matches_explicit_and_native(group, explicit):
    model, x, y, query = fixture("regression")
    native = copy.deepcopy(model)
    native.fit(x, y, recipe=recipe(), num_estimators=1)
    kwargs = {"precision": "float32"} if explicit else {}
    run = tabular_execution(
        model, task="regression", group=group, control_group=group, **kwargs
    )
    run.fit(x, y, recipe=recipe())
    actual, expected = run.predict(query), native.predict(query)
    assert actual is not None
    assert actual.numerical.dtype == torch.float32
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=2e-6, rtol=1e-5
    )
    run.close()
