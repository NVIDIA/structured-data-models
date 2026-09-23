# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, RelatedTables, Stype, TableTensor
from sdm.models import ICLModel, KumoTabular, TabFM, TabICLv2
from sdm.models.callback import Callback
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
        model = KumoTabular(task=task, size="small", pretrained=False)

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

    def predict(
        estimator_batch_size: int | None,
    ) -> tuple[TableTensor, TableTensor]:
        model.fit(
            x=context,
            y=y,
            recipe=recipe,
            num_estimators=9,
            estimator_batch_size=estimator_batch_size,
            # Compare execution with identical stochastic preprocessing.
            generator=torch.Generator(device=device).manual_seed(123),
        )
        first = model.predict(query, estimator_batch_size=estimator_batch_size)
        second = model.predict(
            query[..., :2, :], estimator_batch_size=estimator_batch_size
        )
        return first, second

    expected, expected_short = predict(1)
    assert expected.size()[: 1 + len(batch_shape)] == (9, *batch_shape)
    for estimator_batch_size in (2, None, 10):
        actual, actual_short = predict(estimator_batch_size)
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

    # Reuse the fit across prediction groupings, including partial batches.
    for fit_estimator_batch_size in (1, 2, None):
        predict(fit_estimator_batch_size)
        for predict_estimator_batch_size in (None, 8, 2, 1):
            actual = model.predict(
                query, estimator_batch_size=predict_estimator_batch_size
            )
            assert actual.schema == expected.schema
            torch.testing.assert_close(
                actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
            )

    actual = model(
        x_context=context,
        y_context=y,
        x_query=query,
        recipe=recipe,
        num_estimators=9,
        generator=torch.Generator(device=device).manual_seed(123),
    )
    assert actual.schema == expected.schema
    assert actual.dtype == expected.dtype
    assert actual.device == device
    assert torch.is_inference(actual)
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
    )


@withCUDA
@pytest.mark.parametrize("name", ["tabfm", "kumo"])
def test_estimator_batching_callback_columns(
    device: torch.device,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SelectColumns(Callback):
        requires_grad = True

        def on_context_preprocessing_end(
            self,
            model: torch.nn.Module,
            x: TableTensor,
            y: TableTensor,
            related_tables: RelatedTables[TableTensor] | None,
        ) -> tuple[
            TableTensor, TableTensor, RelatedTables[TableTensor] | None
        ]:
            x = x.select_columns(x.columns[Stype.numerical][::2])
            y = y.replace_blocks(
                categorical=CategoricalTensor(
                    code=y.categorical.code,
                    categories=tuple(
                        c + 100 for c in y.categorical.categories
                    ),
                ),
            )
            return x, y, related_tables

        def on_query_preprocessing_end(
            self,
            model: torch.nn.Module,
            x: TableTensor,
            related_tables: RelatedTables[TableTensor] | None,
        ) -> tuple[TableTensor, RelatedTables[TableTensor] | None]:
            x = x.select_columns(x.columns[Stype.numerical][::2])
            self.numerical = x.numerical.detach().requires_grad_(True)
            return x.replace_blocks(numerical=self.numerical), related_tables

        def on_model_forward_end(
            self, model: torch.nn.Module, out: TableTensor
        ) -> TableTensor:
            out = out.select_columns("120")
            (gradient,) = torch.autograd.grad(
                out.numerical.sum(), self.numerical
            )
            assert gradient.isfinite().all()
            return out

    model = _build(name, "classification", monkeypatch).to(device)
    x = TableTensor(
        numerical=torch.randn(9, 2, device=device),
        categorical=CategoricalTensor.from_tensor(
            (torch.arange(9, device=device) % 2).view(-1, 1)
        ),
    )
    y = TableTensor(
        categorical=CategoricalTensor.from_tensor(
            (10 + 10 * (torch.arange(9, device=device) % 3)).view(-1, 1)
        ),
    )
    recipe = Recipe(
        features=[sp.ToNumerical(), sp.ShuffleColumns()],
        target=sp.ShuffleCategories(),
    )
    callbacks = (SelectColumns(),)
    expected = model(
        x_context=x,
        y_context=y,
        x_query=x,
        recipe=recipe,
        num_estimators=3,
        callbacks=callbacks,
        generator=torch.Generator(device=device).manual_seed(123),
    )
    for fit_estimator_batch_size in (1, 2, None):
        model.fit(
            x=x,
            y=y,
            recipe=recipe,
            num_estimators=3,
            estimator_batch_size=fit_estimator_batch_size,
            callbacks=callbacks,
            generator=torch.Generator(device=device).manual_seed(123),
        )
        for predict_estimator_batch_size in (1, 2, None):
            actual = model.predict(
                x,
                estimator_batch_size=predict_estimator_batch_size,
                callbacks=callbacks,
            )
            assert actual.schema == expected.schema
            torch.testing.assert_close(
                actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
            )
