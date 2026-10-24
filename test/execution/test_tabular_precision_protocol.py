# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-asymmetric precision/mutation/failure behavior with borrowed groups."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_tabular import fixture
from test.execution.test_tabular_precision import no_cuda as no_cuda
from test.execution.test_tabular_precision import recipe

from sdm.execution import tabular_execution
from sdm.nn import TransformerBlock


def worker(rank, world, directory):
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
        with pytest.raises(ValueError, match="policies differ"):
            tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                precision="float32" if rank else "float16",
                owner_local_frontend=True,
                shard_row_workspace=True,
            )
        assert model._execution is None
        dist.barrier()

        for problem in ("precision", "cache", "row_failure"):
            model, x, y, query = fixture("regression")
            run = tabular_execution(
                model,
                task="regression",
                group=dist.group.WORLD,
                control_group=dist.group.WORLD,
                precision="float16",
                owner_local_frontend=True,
                shard_row_workspace=True,
            )
            if problem != "row_failure":
                run.fit(
                    x if rank == 0 else None,
                    y if rank == 0 else None,
                    recipe=recipe(),
                )
                cache = run._icl_cache
                assert cache is not None
                if rank == 0:
                    if problem == "precision":
                        run.precision = "float32"
                    else:
                        with torch.no_grad():
                            next(cache._tensors()).add_(1)
                with pytest.raises(RuntimeError, match="changed") as caught:
                    run.predict(query if rank == 0 else None)
                assert run._icl_cache is cache
            else:
                original = torch.add
                error = RuntimeError("rank-local FP16 row residual failure")
                seen = []

                def fail(
                    left,
                    *args,
                    _original=original,
                    _error=error,
                    _seen=seen,
                    **kwargs,
                ):
                    if (
                        rank == world - 1
                        and isinstance(left, torch.Tensor)
                        and left.dtype == torch.float16
                        and left.shape[-2:] == (5, 8)
                    ):
                        _seen.append(tuple(left.shape))
                        raise _error
                    return _original(left, *args, **kwargs)

                with (
                    patch.object(torch, "add", fail),
                    patch.object(
                        TransformerBlock,
                        "auto_batch_size_limit",
                        return_value=8,
                    ),
                    pytest.raises(
                        RuntimeError,
                        match="rank-local FP16 row residual failure",
                    ) as caught,
                ):
                    run.fit(
                        x if rank == 0 else None,
                        y if rank == 0 else None,
                        recipe=recipe(),
                    )
                if rank == world - 1:
                    assert seen
                    assert caught.value is error
            assert not torch.is_autocast_enabled("cpu")
            with pytest.raises(RuntimeError) as repeated:
                run.close()
            assert repeated.value is caught.value
            assert model._execution is run
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_rank_precision_cache_and_row_failure(tmp_path, world):
    mp.spawn(worker, args=(world, str(tmp_path)), nprocs=world)
