# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import Recipe, TableTensor
from sdm.processing.execution import RecipeExecution
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("infer_members", [False, True])
def test_target_clamper(device: torch.device, infer_members: bool) -> None:
    target = torch.tensor(
        [
            [[torch.nan, -1.0, torch.nan], [0.0, 1.0, torch.nan]],
            [[1.0, -2.0, 0.0], [2.0, 0.0, torch.nan]],
        ],
        device=device,
    )
    x = torch.empty(2, 2, 0, device=device)
    if infer_members:
        target = target.unsqueeze(0).repeat(2, 1, 1, 1)
        target[1, 1, 0, 0] = -1.0
        x = x.unsqueeze(0).expand(2, -1, -1, -1)

    execution = RecipeExecution(
        Recipe(
            target=sp.FlipSign(probability=1.0),
            output=[sp.AverageEstimators(), sp.TargetClamper()],
        )
    )
    execution.fit_transform(
        x=x,
        y=target,
        related_tables=None,
        num_members=None if infer_members else 2,
    )
    columns = [f"{name}__q{q}" for name in ("a", "b", "c") for q in (10, 90)]
    outputs = [
        TableTensor.from_tensor(
            target.new_tensor(values).view(1, 1, -1).expand(2, 3, -1),
            columns=columns,
        )
        for values in (
            [-6.0, -2.0, -4.0, -1.0, -6.0, -2.0],
            [2.0, 4.0, 0.0, 3.0, 2.0, 4.0],
        )
    ]
    result = execution.transform_output(outputs)
    expected = (
        target.new_tensor(
            [[0.0, 1.0, -2.0, 1.0, -2.0, 1.0], [0.0, 1.0, -2.0, 1.0, 0.0, 1.0]]
        )
        .unsqueeze(-2)
        .expand(-1, 3, -1)
    )
    if infer_members:
        expected = expected.clone()
        expected[1, :, 0] = -2.0
    torch.testing.assert_close(result.numerical, expected)
