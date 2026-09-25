# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm import model as tabfm_module
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize(
    ("forward_estimator_batch_size", "predict_estimator_batch_size"),
    [(1, 1), (1, 8), (8, 1), (2, None), (None, 2)],
)
def test_forward(
    device: torch.device,
    dtype: torch.dtype,
    forward_estimator_batch_size: int | None,
    predict_estimator_batch_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tabfm_module,
        "_TabFM",
        functools.partial(
            tabfm_module._TabFM,
            channels=64,
            num_inducing_points=128,
            num_readout_tokens=4,
            num_icl_layers=4,
        ),
    )

    model = TabFM(
        task="regression" if dtype.is_floating_point else "classification",
        pretrained=False,
        device=device,
    )
    # Make predictions sensitive to cached attention and feature permutations.
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    if device.type == "cpu":
        assert repr(model) == "TabFM()"
    else:
        assert repr(model) == "TabFM(device=cuda:0)"

    x = TableTensor(
        numerical=torch.randn(8, 3, device=device),
        categorical=CategoricalTensor.from_tensor(
            torch.randint(0, 2, (8, 3), device=device)
        ),
    )
    x_context, x_query = x.split(5, dim=0)

    if dtype.is_floating_point:
        y_context = torch.randn(5, 1, device=device)
    else:
        y_context = torch.tensor([0, 1, 0, 1, 0], device=device).unsqueeze(-1)

    generator = torch.Generator(device=device).manual_seed(1)
    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=9,
        estimator_batch_size=forward_estimator_batch_size,
        generator=generator,
    )
    assert out.dtype == x_context.dtype
    assert out.device == device
    assert torch.is_inference(out)
    if dtype.is_floating_point:
        assert out.size() == (3, 1)
    else:
        assert out.size() == (3, 2)

    generator = torch.Generator(device=device).manual_seed(1)
    model.fit(
        x=x_context,
        y=y_context,
        num_estimators=9,
        generator=generator,
    )
    assert model._cache is not None
    assert model._cache.size() > 0
    assert model.predict(
        x=x_query,
        estimator_batch_size=predict_estimator_batch_size,
    ).allclose(out, atol=1e-4, rtol=1e-4)
    model.clear()
