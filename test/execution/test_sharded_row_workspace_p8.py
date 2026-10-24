# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eight actual Gloo owners, including empty row and column owners."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test.execution.test_sharded_row_workspace import (
    component_worker,
    policy_worker,
)
from test.execution.test_tabular import fixture, journey

from sdm.execution import tabular_execution
from sdm.nn import Attention


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("count", [1, 11])
@pytest.mark.parametrize(
    "policy", [(False, False), (True, False), (True, True)]
)
def test_p8_public(
    tmp_path: Path, task: str, count: int, policy: tuple[bool, bool]
) -> None:
    frontend, workspace = policy
    mp.spawn(
        journey,
        args=(
            8,
            str(tmp_path),
            task,
            count > 1,
            None,
            frontend,
            count,
            workspace,
        ),
        nprocs=8,
        join=True,
    )


@pytest.mark.parametrize("failure", [False, True])
def test_p8_actual_row_bound_and_failure(
    tmp_path: Path, failure: bool
) -> None:
    mp.spawn(
        component_worker,
        args=(8, str(tmp_path), failure),
        nprocs=8,
        join=True,
    )


def test_p8_policy_and_retirement(tmp_path: Path) -> None:
    mp.spawn(policy_worker, args=(8, str(tmp_path)), nprocs=8, join=True)


def head_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/group",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=35),
    )
    assert dist.group.WORLD is not None
    try:
        # Real, internally consistent attention modules: separately reject
        # four query heads and four KV heads before executor ownership.
        for query_heads, kv_heads in ((4, 4), (8, 4)):
            model, _, _, _ = fixture("regression")
            core = next(iter(model.models.values()))
            core.icl_block.layers[0].attn = Attention(
                channels=32,
                num_query_heads=query_heads,
                num_key_value_heads=kv_heads,
            ).eval()
            with pytest.raises(ValueError, match="heads must divide"):
                tabular_execution(
                    model,
                    task="regression",
                    group=dist.group.WORLD,
                    control_group=dist.group.WORLD,
                    owner_local_frontend=True,
                    shard_row_workspace=True,
                )
            assert model._execution is None
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_p8_fit_head_divisibility(tmp_path: Path) -> None:
    mp.spawn(head_worker, args=(str(tmp_path),), nprocs=8, join=True)
