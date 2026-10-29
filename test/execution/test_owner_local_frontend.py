# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-local frontend output, window-boundary and failure controls."""

from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture, journey

from sdm import Task
from sdm.cache import Cache
from sdm.execution import tabular_execution
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_rows import Rows
from sdm.models.tabfm.cell_embedding import CellEmbedding


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("count", [1, 11])
def test_owner_local_public_lifecycle(
    tmp_path: Path, task: str, world: int, count: int
) -> None:
    args = (world, str(tmp_path), task, count > 1, None, True, count)
    if world == 1:
        journey(0, *args)
    else:
        mp.spawn(journey, args=args, nprocs=world, join=True)


def component_worker(rank: int, world: int, directory: str) -> None:
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
        with torch.no_grad():
            for count in (1, 11):
                model, _, _, _ = fixture("regression")
                encoder = model.models[Task.regression].row_embedding
                raw = (
                    torch.arange(count * 3, dtype=torch.float32)
                    .reshape(count, 3)
                    .sin()
                )
                raw[::2, 0] = torch.nan
                raw[:, 2] = torch.nan
                mask = torch.tensor([False, True, False])
                labels = torch.arange(count, dtype=torch.float32)
                baseline = Rows(
                    encoder,
                    raw,
                    labels,
                    mask,
                    Exchange(dist.group.WORLD, dist.group.WORLD),
                    8 << 30,
                )
                candidate = Rows(
                    encoder,
                    raw,
                    labels,
                    mask,
                    Exchange(dist.group.WORLD, dist.group.WORLD),
                    8 << 30,
                    owner_local_frontend=True,
                )
                baseline.cell_rows = candidate.cell_rows = 3
                n, features, readouts, channels = (
                    candidate.stop - candidate.start,
                    candidate.columns,
                    candidate.readouts,
                    candidate.channels,
                )
                before = torch.full((n, features + readouts, channels), -123.0)
                expected_back, actual_back = before.clone(), before.clone()
                expected, actual = (
                    expected_back[:, readouts:],
                    actual_back[:, readouts:],
                )
                expected_cache, actual_cache = Cache(), Cache()
                assert (
                    baseline.cell(
                        x=raw[baseline.start : baseline.stop],
                        categorical_mask=mask,
                        cache=expected_cache,
                        out=expected,
                    )
                    is expected
                )
                calls = []
                forward = CellEmbedding.forward

                def observed(
                    module,
                    x,
                    categorical_mask,
                    *,
                    _calls=calls,
                    _forward=forward,
                    **kwargs,
                ):
                    _calls.append((x.size(0), kwargs["out"].stride(0)))
                    return _forward(module, x, categorical_mask, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            candidate.exchange,
                            "broadcast",
                            side_effect=AssertionError(
                                "NO_BOUNDARY_BROADCAST"
                            ),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            candidate.exchange,
                            "exchange",
                            side_effect=AssertionError("NO_BOUNDARY_EXCHANGE"),
                        )
                    )
                    stack.enter_context(
                        patch.object(CellEmbedding, "forward", observed)
                    )
                    assert (
                        candidate.cell(
                            x=raw[candidate.start : candidate.stop],
                            categorical_mask=mask,
                            cache=actual_cache,
                            out=actual,
                        )
                        is actual
                    )
                expected_rows = [
                    min(start + 3, count, candidate.stop)
                    - max(start, candidate.start)
                    for start in range(0, count, 3)
                    if max(start, candidate.start)
                    < min(start + 3, count, candidate.stop)
                ]
                assert calls == [
                    (rows, (features + readouts) * channels)
                    for rows in expected_rows
                ]
                torch.testing.assert_close(
                    actual_back, expected_back, atol=2e-6, rtol=1e-5
                )
                torch.testing.assert_close(
                    actual_cache["cell_impute_mean"],
                    expected_cache["cell_impute_mean"],
                    atol=0,
                    rtol=0,
                )
                mean = actual_cache["cell_impute_mean"]
                assert isinstance(mean, torch.Tensor)
                assert mean[0, 2] == 0
                for tokens in (readouts, features + readouts):
                    for separate in (False, True):
                        full = (
                            torch.arange(
                                count * (features + readouts) * channels,
                                dtype=torch.float32,
                            )
                            .reshape(count, features + readouts, channels)
                            .cos()
                        )
                        original = full[candidate.start : candidate.stop]
                        expected_kv, actual_kv = (
                            original.clone(),
                            original.clone(),
                        )
                        expected_out = (
                            torch.empty(n, tokens, channels)
                            if separate
                            else expected_kv[:, :tokens]
                        )
                        actual_out = (
                            torch.empty(n, tokens, channels)
                            if separate
                            else actual_kv[:, :tokens]
                        )
                        block = encoder.row_blocks[-1]
                        assert (
                            baseline.row(
                                block,
                                4,
                                tokens,
                                key_value=expected_kv,
                                out=expected_out,
                            )
                            is expected_out
                        )
                        row_calls = []
                        block_forward = block.forward

                        def observed_row(
                            *,
                            _calls=row_calls,
                            _forward=block_forward,
                            **kwargs,
                        ):
                            _calls.append(kwargs["key_value"].size(0))
                            return _forward(**kwargs)

                        with ExitStack() as stack:
                            stack.enter_context(
                                patch.object(
                                    candidate.exchange,
                                    "broadcast",
                                    side_effect=AssertionError(
                                        "NO_BOUNDARY_BROADCAST"
                                    ),
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    candidate.exchange,
                                    "exchange",
                                    side_effect=AssertionError(
                                        "NO_BOUNDARY_EXCHANGE"
                                    ),
                                )
                            )
                            stack.enter_context(
                                patch.object(block, "forward", observed_row)
                            )
                            assert (
                                candidate.row(
                                    block,
                                    4,
                                    tokens,
                                    key_value=actual_kv,
                                    out=actual_out,
                                )
                                is actual_out
                            )
                        expected_windows = [
                            min(start + 4, count, candidate.stop)
                            - max(start, candidate.start)
                            for start in range(0, count, 4)
                            if max(start, candidate.start)
                            < min(start + 4, count, candidate.stop)
                        ]
                        assert row_calls == expected_windows
                        torch.testing.assert_close(
                            actual_out, expected_out, atol=2e-6, rtol=1e-5
                        )
                        torch.testing.assert_close(
                            actual_kv, expected_kv, atol=2e-6, rtol=1e-5
                        )
                        assert actual_out.stride() == expected_out.stride()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_local_intersections_pitch_cache_and_aliases(
    tmp_path: Path, world: int
) -> None:
    mp.spawn(
        component_worker, args=(world, str(tmp_path)), nprocs=world, join=True
    )


def fault_worker(rank: int, directory: str, stage: str) -> None:
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
        model, x, _, _ = fixture("regression")
        if stage == "policy":
            with pytest.raises(ValueError, match="policies differ"):
                tabular_execution(
                    model,
                    task="regression",
                    group=dist.group.WORLD,
                    control_group=dist.group.WORLD,
                    owner_local_frontend=bool(rank),
                )
            assert model._execution is None
        elif stage == "mutation":
            run = tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                owner_local_frontend=True,
            )
            run.owner_local_frontend = False
            with pytest.raises(RuntimeError, match="policy changed") as first:
                run.clear()
            with pytest.raises(
                RuntimeError, match="policy changed"
            ) as repeated:
                run.clear()
            assert repeated.value is first.value
            assert model._execution is run
        else:
            encoder = model.models[Task.regression].row_embedding
            exchange = Exchange(dist.group.WORLD, dist.group.WORLD)
            rows = Rows(
                encoder,
                x,
                torch.zeros(11),
                torch.zeros(3, dtype=torch.bool),
                exchange,
                8 << 30,
                owner_local_frontend=True,
            )
            n = rows.stop - rows.start
            state = x.new_zeros(n, rows.columns + rows.readouts, rows.channels)
            injected = ValueError(
                f"CONTROLLED_OWNER_LOCAL_{stage.upper()}_RANK1"
            )
            original = (
                CellEmbedding.forward
                if stage == "cell"
                else encoder.row_blocks[0].forward
            )

            def fail(*args, **kwargs):
                if rank == 1:
                    raise injected
                return original(*args, **kwargs)

            target = (
                CellEmbedding if stage == "cell" else encoder.row_blocks[0]
            )

            def execute_local():
                if stage == "cell":
                    rows.cell(
                        x=x[rows.start : rows.stop],
                        categorical_mask=rows.mask,
                        cache=Cache(),
                        out=state[:, rows.readouts :],
                    )
                else:
                    rows.row(
                        encoder.row_blocks[0],
                        4,
                        rows.readouts,
                        key_value=state,
                        out=state[:, : rows.readouts],
                    )

            with patch.object(target, "forward", fail), torch.no_grad():
                with pytest.raises(
                    (ValueError, RuntimeError), match=str(injected)
                ) as first:
                    execute_local()
                with pytest.raises(
                    (ValueError, RuntimeError), match=str(injected)
                ) as repeated:
                    exchange.attempt(lambda: None)
                assert repeated.value is first.value
                if rank == 1:
                    assert first.value is injected
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage", ["cell", "row", "policy", "mutation"])
def test_local_failures_and_rank_policy_agreement(
    tmp_path: Path, stage: str
) -> None:
    mp.spawn(fault_worker, args=(str(tmp_path), stage), nprocs=2, join=True)
