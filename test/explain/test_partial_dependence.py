# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm
from sdm.explain import ICEExplainer, PartialDependenceExplainer
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_partial_dependence(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    x = sdm.TableTensor(
        columns={"numerical": ["A"], "categorical": ["B"]},
        numerical=torch.randn(32, 1, device=device),
        categorical=sdm.CategoricalTensor(
            code=torch.randint(0, 2, (32, 1), device=device),
            categories=(torch.arange(2, device=device),),
        ),
    )
    y = x["A"] if dtype.is_floating_point else x["B"]

    model = sdm.models.TabICLv2(pretrained=False, device=device)
    model.fit(x, y)

    out = ICEExplainer(grid_resolution=5).explain(
        model,
        x,
        output="q500" if dtype.is_floating_point else "0",
    )
    assert len(out) == 2
    grid, curve = out["A"]
    assert grid.device == device
    assert grid.size() == (5,)
    assert curve.device == device
    assert curve.size() == (32, 5)
    grid, curve = out["B"]
    assert grid.device == device
    assert grid.size() == (2,)
    assert curve.device == device
    assert curve.size() == (32, 2)

    out = PartialDependenceExplainer(grid_resolution=5).explain(
        model,
        x,
        output="q500" if dtype.is_floating_point else "0",
    )
    assert len(out) == 2
    grid, curve = out["A"]
    assert grid.device == device
    assert grid.size() == (5,)
    assert curve.device == device
    assert curve.size() == (5,)
    grid, curve = out["B"]
    assert grid.device == device
    assert grid.size() == (2,)
    assert curve.device == device
    assert curve.size() == (2,)
