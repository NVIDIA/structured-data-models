# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sdm.execution._tabular_icl as icl
import sdm.execution.tabular as tabular
import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, Stype, TableTensor, Task
from sdm.cache import Cache, KVCacheEntry
from sdm.execution import tabular_execution
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.nn import SDPA, TransformerBlock


def fixture(task: str):
    model = KumoTabular(task=task, pretrained=False, device="meta").eval()
    core = _KumoTabular(
        num_classes=10 if task == "classification" else 0,
        num_quantiles=999,
        cell_channels=8,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=3,
        num_readout_tokens=2,
        group_size=3,
        num_frequencies=3,
        icl_channels=32,
        num_icl_layers=3,
        num_icl_heads=8,
        num_icl_key_value_heads_for_query=2,
    ).eval()
    with torch.no_grad():
        for index, parameter in enumerate(core.parameters()):
            values = torch.arange(parameter.numel()).reshape(parameter.shape)
            parameter.copy_(0.09 * (values * 0.31 + index * 0.17).sin())
    model.models[Task(task)] = core
    x = torch.arange(11 * 3, dtype=torch.float32).reshape(11, 3).sin()
    query = torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3).cos()
    if task == "classification":
        y = TableTensor(
            columns={Stype.categorical: ["label"]},
            categorical=CategoricalTensor(
                code=(torch.arange(11) % 3).view(-1, 1),
                categories=(torch.tensor([8, 3, 9]),),
            ),
        )
    else:
        y = torch.arange(11, dtype=torch.float32).sin().view(-1, 1)
    return model, x, y, query


def journey(
    rank: int,
    world: int,
    directory: str,
    task: str,
    chunked: bool = False,
    fit_head_gather_window: int | None = None,
    owner_local_frontend: bool = False,
    context_rows: int = 11,
    shard_row_workspace: bool = False,
    default_recipe: bool = False,
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
        x, y = x[:context_rows], y[:context_rows]
        native = copy.deepcopy(model)
        recipe = Recipe(
            target=sp.StypeDispatch(numerical=sp.Standardize()),
            output=[sp.AverageEstimators()],
        )
        if default_recipe:
            recipe = None

        def fit(target, *args, **kwargs):
            generator = None
            if default_recipe:
                generator = torch.Generator().manual_seed(42)
                kwargs["generator"] = (
                    generator if target is native or rank == 0 else None
                )
            # Exercise original fit windows; query retains its native AUTO.
            with (
                patch.object(
                    TransformerBlock, "auto_batch_size_limit", return_value=2
                )
                if chunked
                else nullcontext()
            ):
                target.fit(*args, **kwargs)
            return generator

        native_generator = fit(native, x, y, recipe=recipe, num_estimators=1)
        batches = [query[:2], query[2:4], query[4:6], query[6:]]
        expected = [native.predict(batch) for batch in batches]
        run = tabular_execution(
            model,
            task=task,
            group=dist.group.WORLD,
            control_group=dist.group.WORLD,
            fit_head_gather_window=fit_head_gather_window,
            owner_local_frontend=owner_local_frontend,
            shard_row_workspace=shard_row_workspace,
        )
        for _ in range(2):
            owner_generator = fit(
                run,
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=recipe,
            )
            if owner_local_frontend:
                for cache in (run._row_cache, run._icl_cache):
                    assert cache is not None
                    assert cache.is_replaying
                    for key in cache:
                        actual_cache = Cache.__getitem__(cache, key)
                        native_cache = native._cache[0][key]
                        if isinstance(actual_cache, KVCacheEntry):
                            for actual_tensor, native_tensor in zip(
                                actual_cache, native_cache, strict=True
                            ):
                                torch.testing.assert_close(
                                    actual_tensor,
                                    native_tensor.narrow(
                                        0, cache.start, cache.length
                                    ),
                                    atol=2e-6,
                                    rtol=1e-5,
                                )
                        elif isinstance(actual_cache, torch.Tensor):
                            torch.testing.assert_close(
                                actual_cache,
                                native_cache,
                                atol=2e-6,
                                rtol=1e-5,
                            )
            actual = run.predict_batches(batches if rank == 0 else None)
            if rank == 0:
                assert actual is not None
                if default_recipe:
                    assert torch.equal(
                        native_generator.get_state(),
                        owner_generator.get_state(),
                    )
                for left, right in zip(actual, expected, strict=True):
                    assert left.columns == right.columns
                    if task == "regression":
                        assert left.shape[-1] == 999
                    torch.testing.assert_close(
                        left.numerical, right.numerical, atol=2e-6, rtol=1e-5
                    )
                    if owner_local_frontend and task == "classification":
                        assert torch.equal(
                            left.numerical.argmax(-1),
                            right.numerical.argmax(-1),
                        )
            else:
                assert actual is None
        run.clear()
        fit(
            run,
            x if rank == 0 else None,
            y if rank == 0 else None,
            recipe=recipe,
        )
        last = run.predict(query if rank == 0 else None)
        if rank != 0:
            assert last is None
        run.close()
        assert dist.is_initialized()
        fit(model, x, y, recipe=recipe, num_estimators=1)
        if rank == 0:
            assert last is not None
            torch.testing.assert_close(
                last.numerical,
                model.predict(query).numerical,
                atol=2e-6,
                rtol=1e-5,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_public_journey(tmp_path: Path, task: str, world: int) -> None:
    if world == 1:
        journey(0, world, str(tmp_path), task)
    else:
        mp.spawn(
            journey, args=(world, str(tmp_path), task), nprocs=world, join=True
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_chunked_native_windows(tmp_path: Path, task: str) -> None:
    mp.spawn(journey, args=(2, str(tmp_path), task, True), nprocs=2, join=True)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_two_owner_default_recipe_journey(tmp_path: Path, task: str) -> None:
    mp.spawn(
        journey,
        args=(2, str(tmp_path), task, False, None, False, 11, False, True),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("window", [1, 2, 4])
def test_fit_head_gather_window_journey(
    tmp_path: Path, task: str, world: int, window: int
) -> None:
    if world == 1:
        journey(0, world, str(tmp_path), task, False, window)
    else:
        mp.spawn(
            journey,
            args=(world, str(tmp_path), task, False, window),
            nprocs=world,
            join=True,
        )


@pytest.fixture
def group(tmp_path: Path) -> Iterator[dist.ProcessGroup]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/single", rank=0, world_size=1
    )
    assert dist.group.WORLD is not None
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()
        torch.set_num_threads(previous)


def execution(model: KumoTabular, group: dist.ProcessGroup):
    return tabular_execution(
        model, task="regression", group=group, control_group=group
    )


def test_fit_head_gather_window_defaults_to_none(group) -> None:
    model, _, _, _ = fixture("regression")
    run = execution(model, group)
    assert run.fit_head_gather_window is None
    run.close()


@pytest.mark.parametrize(
    "window",
    [True, False, 0, -1, 2.0, 2.5, float("nan"), float("inf"), "2"],
)
def test_invalid_fit_head_gather_window_before_lease(group, window) -> None:
    model, x, y, query = fixture("regression")
    with pytest.raises(ValueError, match="positive integer"):
        tabular_execution(
            model,
            task="regression",
            group=group,
            control_group=group,
            fit_head_gather_window=window,
        )
    assert model._execution is None
    model.fit(x, y, recipe=Recipe(), num_estimators=1)
    assert model.predict(query).shape[-1] == 999


def test_fit_head_gather_window_mutation_retires_first_error(group) -> None:
    model, x, y, query = fixture("regression")
    run = tabular_execution(
        model,
        task="regression",
        group=group,
        control_group=group,
        fit_head_gather_window=2,
    )
    run.fit(x, y, recipe=Recipe())
    cache = model._cache
    run.fit_head_gather_window = 4
    with pytest.raises(RuntimeError, match="policy changed") as caught:
        run.predict(query)
    # Restoring the policy cannot release uncertain work or renew the lease.
    run.fit_head_gather_window = 2
    for operation in (run.clear, run.close, lambda: run.fit(x, y)):
        with pytest.raises(RuntimeError) as repeated:
            operation()
        assert repeated.value is caught.value
    assert model._cache is cache
    assert model._execution is run


def mismatched_fit_head_gather_window(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/mismatch",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
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
                fit_head_gather_window=rank + 1,
            )
        assert model._execution is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_rank_fit_head_gather_window_mismatch(tmp_path: Path) -> None:
    mp.spawn(
        mismatched_fit_head_gather_window,
        args=(str(tmp_path),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("kind", ["pre", "post", "local", "compiled"])
def test_unsupported_hooks_before_lease(group, kind: str) -> None:
    model, x, y, query = fixture("regression")
    handle = None
    if kind == "pre":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda module, args: None
        )
    elif kind == "post":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda module, args, result: None
        )
    elif kind == "local":
        handle = model.register_forward_hook(lambda module, args, result: None)
    else:
        model._compiled_call_impl = lambda *args, **kwargs: None
    try:
        with pytest.raises(ValueError, match=r"hooks|unmodified"):
            execution(model, group)
        assert model._execution is None
    finally:
        if handle is not None:
            handle.remove()
        model._compiled_call_impl = None
    model.fit(x, y, recipe=Recipe())
    assert model.predict(query).shape[-1] == 999


def test_borrowed_model_mutations_rejected_before_change(group) -> None:
    model, x, y, query = fixture("regression")
    run = execution(model, group)
    for operation in (
        model.clear,
        model.train,
        lambda: model.to(torch.float64),
        lambda: model.fit(x, y),
        lambda: model.predict(query),
        lambda: model.load_state_dict(model.state_dict()),
        lambda: copy.deepcopy(model),
    ):
        with pytest.raises(RuntimeError, match="owning tabular executor"):
            operation()
    assert not model.training
    assert next(model.parameters()).dtype == torch.float32
    run.close()
    model.train()
    assert model.training


@pytest.mark.parametrize("mutation", ["parameter", "cache", "kv", "hook"])
def test_generation_rejects_stale_state(group, mutation: str) -> None:
    model, x, y, query = fixture("regression")
    run = execution(model, group)
    run.fit(x, y, recipe=Recipe())
    handle = None
    if mutation == "parameter":
        with torch.no_grad():
            next(model.parameters()).add_(1)
    elif mutation == "cache":
        model._cache = copy.copy(model._cache)
    elif mutation == "kv":
        with torch.no_grad():
            next(run._icl_cache._tensors()).add_(1)
    else:
        handle = model.register_forward_hook(lambda module, args, output: None)
    try:
        with pytest.raises(
            (RuntimeError, ValueError), match=r"changed|hooks"
        ) as caught:
            run.predict(query)
        for operation in (run.clear, run.close, lambda: run.fit(x, y)):
            with pytest.raises(type(caught.value)) as again:
                operation()
            assert again.value is caught.value
    finally:
        if handle is not None:
            handle.remove()


def test_failed_fence_retains_generation_and_first_error(
    group, monkeypatch
) -> None:
    model, x, y, query = fixture("regression")
    run = execution(model, group)
    run.fit(x, y, recipe=Recipe())
    cache, rows, icl = model._cache, run._row_cache, run._icl_cache
    primary = RuntimeError("first failed completion fence")

    def fail() -> None:
        raise primary

    monkeypatch.setattr(run, "_fence", fail)
    with pytest.raises(RuntimeError, match="first failed") as caught:
        run.clear()
    assert caught.value is primary
    assert model._cache is cache
    assert run._row_cache is rows
    assert run._icl_cache is icl
    monkeypatch.setattr(run, "_fence", lambda: None)
    for operation in (run.clear, run.close, lambda: run.predict(query)):
        with pytest.raises(RuntimeError) as again:
            operation()
        assert again.value is primary


def test_default_recipe_rng_and_native_output(group) -> None:
    model, x, y, query = fixture("regression")
    native = copy.deepcopy(model)
    initial = torch.Generator().manual_seed(42).get_state()
    native_generator = torch.Generator().set_state(initial)
    owner_generator = torch.Generator().set_state(initial)
    native.fit(x, y, generator=native_generator, num_estimators=1)
    expected = native.predict(query)
    run = execution(model, group)
    run.fit(x, y, generator=owner_generator)
    actual = run.predict(query)
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=2e-6, rtol=1e-5
    )
    assert torch.equal(
        native_generator.get_state(), owner_generator.get_state()
    )
    assert actual.columns == expected.columns
    run.close()


def test_two_models_have_independent_instance_dispatch(group) -> None:
    first, x, y, query = fixture("regression")
    second = copy.deepcopy(first)
    original = type(first)._forward
    a, b = execution(first, group), execution(second, group)
    a.fit(x, y, recipe=Recipe())
    b.fit(x, y + 2, recipe=Recipe())
    expected = a.predict(query).clone()
    b.close()
    torch.testing.assert_close(a.predict(query).numerical, expected.numerical)
    assert type(first)._forward is original
    assert "forward" not in vars(first.models[Task.regression])
    a.close()


def test_import_has_no_research_or_cuda() -> None:
    code = """
import sys
import torch
def reject(*args, **kwargs):
    raise AssertionError('CUDA initialized during execution import')
torch.cuda._lazy_init = reject
import sdm.execution
assert not any(name.startswith('research') for name in sys.modules)
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_owner_thread_reentry_and_duplicate_lease(group) -> None:
    model, x, y, query = fixture("regression")
    run = execution(model, group)
    with pytest.raises(ValueError, match="unleased"):
        execution(model, group)
    with ThreadPoolExecutor(max_workers=1) as thread:
        future = thread.submit(run.clear)
        with pytest.raises(RuntimeError, match="another thread"):
            future.result()
    with run._call("fit"), pytest.raises(RuntimeError, match="overlap"):
        run.clear()
    run.fit(x, y, recipe=Recipe())
    assert run.predict(query).shape[-1] == 999
    run.close()


@pytest.mark.parametrize("kind", ["missing", "grad", "output", "budget"])
def test_unsupported_fit_retires_without_silent_fallback(group, kind) -> None:
    model, x, y, query = fixture("regression")
    run = tabular_execution(
        model,
        task="regression",
        group=group,
        control_group=group,
        column_staging_bytes=1 if kind == "budget" else 8 << 30,
    )
    if kind == "missing":
        y = None
    if kind == "grad":
        x.requires_grad_()
    recipe = (
        Recipe(output=sp.Callable(lambda value: value))
        if kind == "output"
        else Recipe()
    )
    with pytest.raises(
        ValueError, match=r"inputs|processors|staging"
    ) as caught:
        run.fit(x, y, recipe=recipe)
    with pytest.raises(
        ValueError, match=r"inputs|processors|staging"
    ) as again:
        run.predict(query)
    assert again.value is caught.value


def test_expanded_ecoc_is_rejected(group) -> None:
    model, x, _, _ = fixture("classification")
    y = TableTensor(
        columns={Stype.categorical: ["label"]},
        categorical=CategoricalTensor(
            code=torch.arange(11).view(-1, 1),
            categories=(torch.arange(11),),
        ),
    )
    run = tabular_execution(
        model, task="classification", group=group, control_group=group
    )
    with pytest.raises(ValueError, match="ECOC"):
        run.fit(x, y, recipe=Recipe())


def test_autocast_and_policy_changes_rejected(group) -> None:
    model, _, _, _ = fixture("regression")
    with torch.autocast("cpu"), pytest.raises(ValueError, match="autocast"):
        execution(model, group)
    run = execution(model, group)
    run.attention_workspace_bytes += 1
    with pytest.raises(RuntimeError, match="policy changed"):
        run.clear()


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("limit", [None, 2])
def test_attention_dispatch_preserves_native_forward_and_cache(grad, limit):
    model, _, _, _ = fixture("regression")
    block = model.models[Task.regression].icl_block.layers[0]
    query = torch.arange(3 * 4 * 32).view(3, 4, 32).float().sin()
    calls = []

    def attention(sdpa: SDPA, **kwargs):
        calls.append(sdpa)
        return sdpa(**kwargs)

    with torch.set_grad_enabled(grad):
        expected, expected_kv = block(
            query=query.clone(), return_key_value=True, batch_size_limit=limit
        )
        actual, actual_kv = block(
            query=query.clone(),
            return_key_value=True,
            batch_size_limit=limit,
            _attention=attention,
        )
    assert calls
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for left, right in zip(actual_kv, expected_kv, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_explicit_scaling_does_not_mutate_native_sdpa() -> None:
    model, _, _, _ = fixture("regression")
    sdpa = model.models[Task.regression].icl_block.layers[0].attn.sdpa
    query = torch.arange(5 * 8 * 4).view(5, 8, 4).float().sin()
    key, value = query.cos(), query.sin()
    scaling = sdpa.query_scaling
    with torch.no_grad():
        expected = sdpa(query, key, value)
        scaled = scaling(query, key_len=key.size(-3))
        actual = sdpa(scaled, key, value, _already_scaled=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert sdpa.query_scaling is scaling


def test_native_serialization_without_execution_field() -> None:
    model, x, y, query = fixture("regression")
    state = model.__getstate__()
    state.pop("_execution")
    restored = object.__new__(KumoTabular)
    restored.__setstate__(state)
    restored.fit(x, y, recipe=Recipe())
    assert restored.predict(query).shape[-1] == 999


def rejected_fit(rank: int, directory: str, failure: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/failed-group",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, _ = fixture("regression")
        run = tabular_execution(
            model,
            task="regression",
            group=dist.group.WORLD,
            control_group=dist.group.WORLD,
            column_staging_bytes=1 if failure == "budget" else 8 << 30,
        )
        original = (
            torch.Tensor.nanmean
            if failure == "mean"
            else torch.Tensor.clone
            if failure == "row"
            else torch.Tensor.contiguous
            if failure == "root"
            else torch.empty
        )
        first = RuntimeError("rank-local allocation failed")

        def allocate(*args, **kwargs):
            shape = (
                args[0].shape
                if failure in ("mean", "row", "root")
                else (args[0],)
                if len(args) == 1 and isinstance(args[0], int)
                else args[0]
                if len(args) == 1
                else args
            )
            target = (
                (11, 3)
                if failure == "mean"
                else (2, 5, 8)
                if failure == "row"
                else (3,)
                if failure == "root"
                else (6, 5, 8)
            )
            failing_rank = 0 if failure == "root" else 1
            if rank == failing_rank and tuple(shape) == target:
                raise first
            return original(*args, **kwargs)

        injection = (
            patch.object(torch.Tensor, "nanmean", allocate)
            if failure == "mean"
            else patch.object(torch.Tensor, "clone", allocate)
            if failure == "row"
            else patch.object(torch.Tensor, "contiguous", allocate)
            if failure == "root"
            else patch.object(torch, "empty", allocate)
            if failure == "buffer"
            else nullcontext()
        )
        with (
            injection,
            patch.object(
                TransformerBlock, "auto_batch_size_limit", return_value=2
            )
            if failure == "row"
            else nullcontext(),
            pytest.raises(
                (ValueError, RuntimeError), match=r"staging budget|rank-local"
            ) as primary,
        ):
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=Recipe(),
            )
        if rank == (0 if failure == "root" else 1) and failure != "budget":
            assert primary.value is first
        with pytest.raises(
            (ValueError, RuntimeError), match=r"staging budget|rank-local"
        ) as repeated:
            run.close()
        assert repeated.value is primary.value
        # The executor must not destroy caller-owned groups even on failure.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "failure", ["budget", "mean", "buffer", "row", "root"]
)
def test_collective_preallocation_failure(
    tmp_path: Path, failure: str
) -> None:
    mp.spawn(rejected_fit, args=(str(tmp_path), failure), nprocs=2, join=True)


def test_local_primary_survives_control_collective_failure(group, monkeypatch):
    model, _, _, _ = fixture("regression")
    run = execution(model, group)
    first = RuntimeError("original allocation failure")

    def fail_operation():
        raise first

    def fail_collective(*args, **kwargs):
        raise RuntimeError("secondary transport failure")

    monkeypatch.setattr(dist, "all_gather_object", fail_collective)
    with pytest.raises(RuntimeError, match="original allocation") as caught:
        run.exchange.attempt(fail_operation)
    assert caught.value is first
    with pytest.raises(RuntimeError, match="original allocation") as repeated:
        run.exchange.attempt(lambda: None)
    assert repeated.value is first


def rejected_query_packing(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/query-packing",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, query = fixture("regression")
        run = execution(model, dist.group.WORLD)
        run.fit(
            x if rank == 0 else None,
            y if rank == 0 else None,
            recipe=Recipe(),
        )
        mask = model._cache[0]["categorical_mask"] if rank == 0 else None
        original = torch.Tensor.contiguous
        failure = RuntimeError("root query packing allocation failed")

        def contiguous(value, *args, **kwargs):
            if value is mask:
                raise failure
            return original(value, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "contiguous", contiguous),
            pytest.raises(RuntimeError, match="query packing") as caught,
        ):
            run.predict(query if rank == 0 else None)
        if rank == 0:
            assert caught.value is failure
        with pytest.raises(RuntimeError, match="query packing") as repeated:
            run.close()
        assert caught.value is repeated.value
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_root_query_packing_failure_reaches_peers(tmp_path: Path) -> None:
    mp.spawn(
        rejected_query_packing, args=(str(tmp_path),), nprocs=2, join=True
    )


def rejected_rank_allocation(rank: int, directory: str, stage: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rank-allocation",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    assert dist.group.WORLD is not None
    try:
        model, x, y, query = fixture("regression")
        run = execution(model, dist.group.WORLD)
        failure = RuntimeError(f"rank-local {stage} allocation failed")
        original_clone = torch.Tensor.clone
        original_empty = torch.Tensor.new_empty
        original_contiguous = torch.Tensor.contiguous
        original_predict = icl.predict_icl

        def inject_clone(block, state, cache):
            def clone(value, *args, **kwargs):
                if rank == 0 and value is state:
                    raise failure
                return original_clone(value, *args, **kwargs)

            with patch.object(torch.Tensor, "clone", clone):
                return original_predict(block, state, cache)

        def new_empty(value, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_filename
            if (
                rank == 0
                and caller.endswith("/tabular.py")
                and args == ((0,),)
            ):
                raise failure
            return original_empty(value, *args, **kwargs)

        def contiguous(value, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_filename
            if (
                rank == 0
                and caller.endswith("/_tabular_icl.py")
                and tuple(value.shape) == (11, 8, 4)
            ):
                raise failure
            return original_contiguous(value, *args, **kwargs)

        def fit():
            run.fit(
                x if rank == 0 else None,
                y if rank == 0 else None,
                recipe=Recipe(),
            )

        if stage != "fit-gather":
            fit()
        scope = (
            patch.object(tabular, "predict_icl", inject_clone)
            if stage == "query-clone"
            else patch.object(torch.Tensor, "new_empty", new_empty)
            if stage == "empty-labels"
            else patch.object(torch.Tensor, "contiguous", contiguous)
        )
        operation = (
            fit
            if stage == "fit-gather"
            else lambda: run.predict(query if rank == 0 else None)
        )
        with scope, pytest.raises(RuntimeError, match=stage) as caught:
            operation()
        if rank == 0:
            assert caught.value is failure
        with pytest.raises(RuntimeError, match=stage) as repeated:
            run.close()
        assert caught.value is repeated.value
        # A peer timeout is not acceptable error propagation.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "stage", ["query-clone", "empty-labels", "fit-gather"]
)
def test_rank_local_allocation_reaches_peers(
    tmp_path: Path, stage: str
) -> None:
    mp.spawn(
        rejected_rank_allocation,
        args=(str(tmp_path), stage),
        nprocs=2,
        join=True,
    )
