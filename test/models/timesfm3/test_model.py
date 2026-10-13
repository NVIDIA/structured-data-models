# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Recipe, TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.model import _TimesFM3
from sdm.testing import withCUDA


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3 * 9)
    assert out.numerical.isfinite().all()
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(
        x=x_context,
        y=y_context,
        context_only_columns=["x1", "x3"],
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)


@withCUDA
@pytest.mark.parametrize("refinement", [False, True])
@pytest.mark.parametrize("horizon", [0, 1, 4, 10])
@pytest.mark.parametrize("values", [(1.0, 3.0), (2.0, 2.0)])
def test_refinement(
    device: torch.device,
    refinement: bool,
    horizon: int,
    values: tuple[float, float],
) -> None:
    model = TimesFM3(pretrained=False, device="meta", refinement=refinement)
    model.model = _TimesFM3(
        input_patch_size=2,
        output_patch_size=4,
        channels=8,
        num_layers=1,
        num_heads=2,
        device=device,
        refinement=refinement,
    )
    # Predict one in normalized space for every quantile and patch.
    with torch.no_grad():
        model.model.icl_block.head.weight.zero_()
        model.model.icl_block.head.bias.fill_(1.0)

    history = torch.tensor(values, device=device)
    initial = history.mean() + history.std(correction=0)
    if refinement and horizon > 4:
        history = torch.cat([history, initial.expand(2)])
        a = history.mean() + history.std(correction=0)
        history = torch.cat([history, initial.expand(2)])
        b = history.mean() + history.std(correction=0)
        history = torch.cat([history, b.expand(2)])
        c = history.mean() + history.std(correction=0)
        history = torch.cat([history, b.expand(2)])
        d = history.mean() + history.std(correction=0)
        expected = torch.stack(
            [initial] * 3 + [a] * 2 + [b] * 2 + [c] * 2 + [d]
        )
    else:
        expected = initial.expand(10)

    # Multiple targets and leading batch dimensions follow the same recurrence.
    scales = history.new_tensor([1.0, 2.0])
    y = torch.tensor(values, device=device).unsqueeze(-1) * scales
    y = y.expand(2, 3, 2, 2)
    out = model(
        x_context=TableTensor.from_tensor(y.new_empty(2, 3, 2, 0)),
        y_context=TableTensor.from_tensor(y),
        x_query=TableTensor.from_tensor(y.new_empty(2, 3, horizon, 0)),
        recipe=Recipe(),
    )
    expected = expected[:horizon, None, None] * scales[:, None]
    expected = expected.expand(horizon, 2, 9).flatten(-2)
    torch.testing.assert_close(
        out.numerical, expected.expand(2, 3, horizon, 18)
    )
