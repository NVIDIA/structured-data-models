# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import sys
import weakref
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_bounded_kv_reuse import recipe
from test.execution.test_tabular import fixture

from sdm.execution import tabular_execution
from sdm.execution._tabular_cache import OwnedCache


def worker(
    rank: int, world: int, directory: str, append_failure: bool
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=15),
    )
    try:
        base, x, y, query = fixture("regression")

        def make(budget):
            assert dist.group.WORLD is not None
            model = copy.deepcopy(base)
            run = tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                precision="float16",
                gathered_kv_cache_bytes=budget,
            )
            return model, run

        def fit(run, rows):
            run.fit(
                x[:rows] if rank == 0 else None,
                y[:rows] if rank == 0 else None,
                recipe=recipe(),
            )

        def predict(run):
            result = run.predict_batches([query[:3]] if rank == 0 else None)
            return result[0].numerical.clone() if rank == 0 else None

        if append_failure:
            model, run = make(1 << 20)
            fit(run, 8)
            cache = run._icl_cache
            first = MemoryError("rank-local replica append failure")
            lines, start = inspect.getsourcelines(
                OwnedCache.materialize_replicas
            )
            target = start + next(
                index
                for index, line in enumerate(lines)
                if "full.append(gathered)" in line
            )
            filename = inspect.getsourcefile(OwnedCache)

            def trace(frame, event, arg):
                # Exercise a real Python-list append allocation boundary,
                # including when it lives in the attempted lambda's frame.
                if (
                    event == "line"
                    and frame.f_code.co_filename == filename
                    and frame.f_lineno == target
                ):
                    raise first
                return trace

            if rank == 1:
                sys.settrace(trace)
            try:
                with pytest.raises(
                    (MemoryError, RuntimeError), match="replica append failure"
                ) as caught:
                    predict(run)
            finally:
                sys.settrace(None)
            if rank == 1:
                assert caught.value is first
            assert run._icl_cache is cache
            assert model._execution is run
            assert cache._replicas is None
            assert len(cache._building_tensors) == 1
            refs = [weakref.ref(tensor) for tensor in cache._building_tensors]
            with (
                patch.object(
                    dist,
                    "all_reduce",
                    side_effect=AssertionError("late control"),
                ),
                patch.object(
                    dist,
                    "all_gather_into_tensor",
                    side_effect=AssertionError("late tensor"),
                ),
            ):
                for action in (lambda: predict(run), run.clear, run.close):
                    with pytest.raises(
                        (MemoryError, RuntimeError),
                        match="replica append failure",
                    ) as repeated:
                        action()
                    assert repeated.value is caught.value
            assert all(ref() is not None for ref in refs)
        else:
            _, probe = make(0)
            expected, sizes = {}, {}
            for rows in (8, 11, 2):
                fit(probe, rows)
                sizes[rows] = probe._icl_cache.replica_plan()[1]
                expected[rows] = predict(probe)
            probe.close()
            for budget, contexts in (
                (sizes[8], (8, 11)),
                (sizes[8] - 1, (8, 2)),
            ):
                _, run = make(budget)
                previous = None
                for rows in contexts:
                    fit(run, rows)
                    cache = run._icl_cache
                    assert cache is not previous
                    owned_bytes = cache.size()
                    assert run.predict_batches([] if rank == 0 else None) == (
                        [] if rank == 0 else None
                    )
                    assert cache._replicas is None
                    assert not run._kv_decision_made
                    actual = predict(run)
                    if rank == 0:
                        torch.testing.assert_close(
                            actual, expected[rows], rtol=0, atol=0
                        )
                        assert actual.size(-1) == 999
                    admitted = sizes[rows] <= budget
                    assert (cache._replicas is not None) == admitted
                    assert cache._building_tensors is None
                    assert cache.size() == owned_bytes + (
                        sizes[rows] if admitted else 0
                    )
                    assert run._kv_decision_made
                    again = predict(run)
                    if rank == 0:
                        assert torch.equal(actual, again)
                    previous = cache
                run.clear()
                assert run._icl_cache is None
                run.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_budget_boundary_and_context_refit(tmp_path: Path, world: int) -> None:
    mp.spawn(worker, args=(world, str(tmp_path), False), nprocs=world)


def test_rank_local_replica_append_failure(tmp_path: Path) -> None:
    mp.spawn(worker, args=(2, str(tmp_path), True), nprocs=2)
