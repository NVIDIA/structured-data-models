# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, TableTensor
from sdm.models import ICLModel, KumoTabular, TabFM, TabICLv2
from sdm.models.kumo.tabular import model as kumo_module
from sdm.models.tabfm import model as tabfm_module
from sdm.models.tabiclv2 import model as tabicl_module
from sdm.testing import withCUDA


def _build(
    name: str,
    task: Literal["classification", "regression"],
    monkeypatch: pytest.MonkeyPatch,
) -> ICLModel:
    if name == "tabiclv2":
        monkeypatch.setattr(
            tabicl_module,
            "_TabICLv2",
            functools.partial(
                tabicl_module._TabICLv2,
                channels=16,
                num_embedding_layers=2,
                num_embedding_heads=2,
                num_inducing_points=4,
                num_readout_tokens=2,
                num_icl_layers=2,
                num_icl_heads=2,
            ),
        )
        model = TabICLv2(task=task, pretrained=False)
    elif name == "tabfm":
        monkeypatch.setattr(
            tabfm_module,
            "_TabFM",
            functools.partial(
                tabfm_module._TabFM,
                channels=16,
                num_embedding_layers=2,
                num_embedding_col_heads=2,
                num_embedding_row_heads=2,
                num_inducing_points=4,
                num_readout_tokens=2,
                num_icl_layers=2,
                num_icl_heads=2,
            ),
        )
        model = TabFM(task=task, pretrained=False)
    else:
        assert name == "kumo"
        monkeypatch.setitem(
            kumo_module.MODEL_KWARGS,
            "small",
            {
                "cell_channels": 16,
                "num_embedding_layers": 2,
                "num_embedding_heads": 2,
                "num_inducing_points": 4,
                "num_readout_tokens": 2,
                "icl_channels": 32,
                "num_icl_layers": 2,
                "num_icl_heads": 2,
            },
        )
        model = KumoTabular(task=task, pretrained=False)

    # Zero-initialized residuals would hide errors in feature permutations.
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    return model


@withCUDA
@pytest.mark.parametrize("name", ["tabiclv2", "tabfm", "kumo"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_estimator_batching(
    device: torch.device,
    name: str,
    task: Literal["classification", "regression"],
    batch_shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build(name, task, monkeypatch).to(device)
    x = TableTensor(
        numerical=torch.randn(*batch_shape, 12, 3, device=device),
        categorical=CategoricalTensor.from_tensor(
            (torch.arange(12, device=device) % 2)
            .view(12, 1)
            .expand(*batch_shape, 12, 1)
        ),
    )
    context, query = x.split(8, dim=-2)
    if task == "classification":
        y = (10 + 10 * (torch.arange(8, device=device) % 3)).view(8, 1)
        y = y.expand(*batch_shape, 8, 1)
    else:
        y = torch.randn(*batch_shape, 8, 1, device=device)

    default = model.default_recipe()
    # Keep every estimator's output so averaging cannot hide misalignment.
    recipe = Recipe(features=default.features, target=default.target)
    if batch_shape:
        # AlignCategories in the default recipes currently only supports one
        # leading batch dimension. Exercise nested model batches separately.
        recipe = Recipe(
            features=[sp.ToNumerical(), sp.Standardize(), sp.ShuffleColumns()],
            target=sp.StypeDispatch(
                categorical=sp.ShuffleCategories(),
                numerical=sp.Standardize(),
            ),
        )

    def predict(batch_size: int | None) -> tuple[TableTensor, TableTensor]:
        model.fit(
            x=context,
            y=y,
            recipe=recipe,
            num_estimators=5,
            estimator_batch_size=batch_size,
            # Compare execution with identical stochastic preprocessing.
            generator=torch.Generator(device=device).manual_seed(123),
        )
        first = model.predict(query)
        second = model.predict(query[..., :2, :])
        return first, second

    expected, expected_short = predict(1)
    assert expected.size()[: 1 + len(batch_shape)] == (5, *batch_shape)
    for batch_size in (2, None, 10):
        actual, actual_short = predict(batch_size)
        assert actual.schema == expected.schema
        assert actual.dtype == expected.dtype
        assert actual.device == device
        assert torch.is_inference(actual)
        torch.testing.assert_close(
            actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            actual_short.numerical,
            expected_short.numerical,
            atol=1e-4,
            rtol=1e-4,
        )

    for batch_size in (1, 2, None, 10):
        actual = model(
            x_context=context,
            y_context=y,
            x_query=query,
            recipe=recipe,
            num_estimators=5,
            estimator_batch_size=batch_size,
            generator=torch.Generator(device=device).manual_seed(123),
        )
        assert actual.schema == expected.schema
        assert actual.dtype == expected.dtype
        assert actual.device == device
        assert torch.is_inference(actual)
        torch.testing.assert_close(
            actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
        )
