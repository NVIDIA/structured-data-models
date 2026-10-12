# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.model import _TimesFM3
from sdm.nn import Attention
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
    assert out.numerical.count_nonzero() > 0
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(
        x=x_context,
        y=y_context,
        context_only_columns=["x1", "x3"],
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)


@withCUDA
def test_core(
    device: torch.device,
) -> None:
    model = _TimesFM3(
        input_patch_size=2,
        output_patch_size=4,
        channels=8,
        num_layers=2,
        num_heads=2,
        device=device,
    )

    out = model(
        x_context=torch.randn(5, 3, device=device),
        x_query=torch.randn(8, 3, device=device),
        x_context_only=torch.randn(5, 1, device=device),
        y=torch.randn(5, 2, device=device),
    )
    assert out.device == device
    assert out.size() == (8, 2, 9)


def _small_model(device: torch.device) -> TimesFM3:
    with torch.device("meta"):
        model = TimesFM3(pretrained=False)
    model.model = _TimesFM3(
        input_patch_size=2,
        output_patch_size=4,
        channels=8,
        num_layers=2,
        num_heads=2,
        device=device,
    )
    # Exercise covariate dependence with nonzero attention projections.
    for module in model.modules():
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.1)
    return model.eval()


@withCUDA
@pytest.mark.parametrize(
    ("num_known", "num_past"), [(0, 0), (2, 0), (0, 2), (2, 2)]
)
def test_public_forecast_matches_core(
    device: torch.device,
    num_known: int,
    num_past: int,
) -> None:
    model = _small_model(device)
    known = [f"known_{i}" for i in range(num_known)]
    past = [f"past_{i}" for i in range(num_past)]
    context = TableTensor.from_tensor(
        torch.randn(
            5, num_known + num_past, device=device, dtype=torch.float64
        ),
        columns=[*known, *past],
    )
    query = TableTensor.from_tensor(
        torch.randn(5, num_known, device=device, dtype=torch.float64),
        columns=known,
    )
    target = TableTensor.from_tensor(
        torch.randn(5, 2, device=device, dtype=torch.float64),
        columns=["a", "b"],
    )
    with torch.inference_mode():
        expected = model.model(
            x_context=context.select_columns(known).numerical.float(),
            x_query=query.numerical.float(),
            x_context_only=context.select_columns(past).numerical.float(),
            y=target.numerical.float(),
        ).flatten(-2)

    out = model(context, target, query)
    assert out.size() == (5, 18)
    assert out.device == device
    assert out.dtype == torch.float64
    assert out.columns[Stype.numerical] == tuple(
        f"{name}__q{i}" for name in ("a", "b") for i in range(10, 100, 10)
    )
    torch.testing.assert_close(out.numerical, expected.double())
    model.fit(x=context, y=target, context_only_columns=past)
    torch.testing.assert_close(model.predict(query).numerical, out.numerical)


@withCUDA
@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_cached_forecasts_with_new_future_covariates(
    device: torch.device,
    estimator_batch_size: int | None,
) -> None:
    model = _small_model(device)
    context = TableTensor.from_tensor(
        torch.randn(2, 5, 2, device=device),
        columns=["known", "past"],
    )
    target = TableTensor.from_tensor(
        torch.randn(2, 5, 2, device=device),
        columns=["a", "b"],
    )
    query = TableTensor.from_tensor(
        torch.randn(2, 5, 1, device=device),
        columns=["known"],
    )
    changed_query = query.replace_blocks(numerical=query.numerical + 10)
    expected = model(
        x_context=context,
        y_context=target,
        x_query=query,
        estimator_batch_size=estimator_batch_size,
    )
    changed_expected = model(
        x_context=context,
        y_context=target,
        x_query=changed_query,
        estimator_batch_size=estimator_batch_size,
    )
    assert not torch.equal(expected.numerical, changed_expected.numerical)
    model.fit(
        x=context,
        y=target,
        context_only_columns=["past"],
        estimator_batch_size=estimator_batch_size,
    )
    torch.testing.assert_close(
        model.predict(query).numerical, expected.numerical
    )
    context.numerical.add_(100)
    target.numerical.add_(100)
    torch.testing.assert_close(
        model.predict(changed_query).numerical,
        changed_expected.numerical,
    )
    torch.testing.assert_close(
        model.predict(query).numerical, expected.numerical
    )
