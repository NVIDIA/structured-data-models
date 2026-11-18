# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Task
from sdm.prior import should_drop


def test_should_drop_disabled() -> None:
    assert not should_drop(
        x=torch.zeros(4, 2), y=torch.zeros(4), task_type=Task.regression
    )


@pytest.mark.parametrize("task_type", [Task.classification, Task.regression])
@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize(
    ("drop_trivial", "drop_unlearnable"),
    [(True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("max_rows", [None, 64])
def test_should_drop(
    task_type: Task,
    learnable: bool,
    max_rows: int | None,
    drop_trivial: bool,
    drop_unlearnable: bool,
) -> None:
    pytest.importorskip("sklearn")
    y = torch.arange(256) % 2
    x = y.float() if learnable else torch.zeros(256)
    x = torch.stack((x, torch.full_like(x, torch.nan)), dim=-1)

    actual = should_drop(
        x=x,
        y=y,
        task_type=task_type,
        drop_trivial=drop_trivial,
        drop_unlearnable=drop_unlearnable,
        max_rows=max_rows,
    )

    assert actual is (
        (drop_trivial and learnable) or (drop_unlearnable and not learnable)
    )
    assert x[:, 1].isnan().all()
