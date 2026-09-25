# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import KumoTabular
from sdm.models.callback import Callback


def _build(
    task: Literal["classification", "regression"],
    size: Literal["small", "medium"],
) -> KumoTabular:
    model = KumoTabular(task=task, size=size, pretrained=False)
    # Residual branches are zero-initialized, so an untrained model maps every
    # row onto the same constant. Randomize them to make the prediction depend
    # on the features it is given.
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    return model


@pytest.fixture
def cls_model(size: Literal["small", "medium"]) -> KumoTabular:
    return _build("classification", size)


@pytest.fixture
def reg_model(size: Literal["small", "medium"]) -> KumoTabular:
    return _build("regression", size)


@pytest.fixture(params=["small", "medium"])
def size(request: pytest.FixtureRequest) -> Literal["small", "medium"]:
    return request.param


def _features(stype: Stype = Stype.categorical) -> tuple[TableTensor, ...]:
    numerical = torch.tensor(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
            [104.0, 204.0],
        ]
    )
    label = torch.tensor([[0], [1], [0], [0], [1]])
    if stype == Stype.categorical:
        x = TableTensor(
            columns={Stype.numerical: ("n0", "n1"), Stype.categorical: ("c",)},
            numerical=numerical,
            categorical=CategoricalTensor.from_tensor(label),
        )
    else:
        x = TableTensor(
            columns={Stype.numerical: ("n0", "n1", "c")},
            numerical=torch.cat((numerical, label.float()), dim=-1),
        )
    return x.split(3, dim=0)


def _cls_target(num_classes: int = 3) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [num_classes - 1], [0]]),
            categories=(torch.arange(num_classes).mul(10),),
        ),
    )


def _reg_target(offset: float = 0.0) -> TableTensor:
    values = torch.tensor([[10.0], [20.0], [30.0]])
    return TableTensor.from_tensor(values + offset)


def _recipe() -> sp.Recipe:
    return sp.Recipe(
        features=[sp.ToNumerical()],
        target=sp.StypeDispatch(numerical=sp.Standardize()),
        output=[sp.AverageEstimators()],
    )


def test_forward(
    cls_model: KumoTabular,
    reg_model: KumoTabular,
) -> None:
    x_context, x_query = _features()

    out = cls_model(x_context, _cls_target(), x_query, recipe=_recipe())

    assert out.size() == (2, 3)
    assert out.columns[Stype.numerical] == ("0", "10", "20")
    assert out.dtype == x_query.dtype
    assert torch.is_inference(out)

    x_context, x_query = _features(Stype.numerical)
    out = reg_model(
        x_context,
        _reg_target(),
        x_query,
        recipe=_recipe(),
    )

    assert out.size() == (2, 999)
    assert out.columns[Stype.numerical] == tuple(
        f"q{i:03d}" for i in range(1, 1000)
    )


def test_categorical_features_are_marked(cls_model: KumoTabular) -> None:
    target = _cls_target()

    x_context, x_query = _features(Stype.categorical)
    categorical = cls_model(x_context, target, x_query, recipe=_recipe())
    # Declaring the same column numerical leaves the features handed to the
    # model untouched, so only the stype it embeds them with differs.
    x_context, x_query = _features(Stype.numerical)
    numerical = cls_model(x_context, target, x_query, recipe=_recipe())

    assert not categorical.allclose(numerical)


@pytest.mark.parametrize("num_estimators", [1, 3])
@pytest.mark.parametrize("kv_cache", [True, False])
def test_fit_predict(
    cls_model: KumoTabular,
    reg_model: KumoTabular,
    kv_cache: bool,
    num_estimators: int,
) -> None:
    x_context, x_query = _features()
    for num_classes in (3, 11):
        target = _cls_target(num_classes)
        if kv_cache or num_classes <= 10:
            expected = cls_model(
                x_context=x_context,
                y_context=target,
                x_query=x_query,
                recipe=_recipe(),
                num_estimators=num_estimators,
                generator=torch.Generator().manual_seed(0),
            )
        cls_model.fit(
            x=x_context,
            y=target,
            recipe=_recipe(),
            num_estimators=num_estimators,
            generator=torch.Generator().manual_seed(0),
            kv_cache=kv_cache,
        )
        if not kv_cache and num_classes > 10:
            # A callback uses sequential prediction with the same ECOC seeds.
            expected = cls_model.predict(x_query, callbacks=[Callback()])
        actual = cls_model.predict(x_query)

        assert actual.size() == (2, num_classes)
        assert actual.allclose(expected, atol=1e-5)
        assert actual.columns == expected.columns
        assert actual.dtype == x_query.dtype

    x_context, x_query = _features(Stype.numerical)
    target = _reg_target()
    recipe = _recipe()
    expected = reg_model(
        x_context=x_context,
        y_context=target,
        x_query=x_query,
        recipe=recipe,
        num_estimators=num_estimators,
    )
    reg_model.fit(
        x=x_context,
        y=target,
        recipe=recipe,
        num_estimators=num_estimators,
        kv_cache=kv_cache,
    )
    actual = reg_model.predict(x_query)

    assert actual.dtype == x_query.dtype
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=1e-4,
        rtol=1e-4,
    )


def test_missing_values_pass_through_fit_predict(
    size: Literal["small", "medium"],
) -> None:
    model = _build("regression", size)
    x_context = TableTensor.from_tensor(
        torch.tensor(
            [
                [100.0, float("inf")],
                [float("nan"), 201.0],
                [102.0, 202.0],
            ]
        )
    )
    x_query = TableTensor.from_tensor(
        torch.tensor(
            [
                [float("nan"), 203.0],
                [float("inf"), 203.0],
                [104.0, float("-inf")],
            ]
        )
    )
    target = _reg_target()

    direct = model(x_context, target, x_query)
    model.fit(x_context, target)
    cached = model.predict(x_query)

    assert direct.numerical.isfinite().all()
    assert cached.numerical.isfinite().all()
    assert cached.shape == direct.shape
    assert (direct.numerical.diff(dim=-1) >= 0).all()
    assert (cached.numerical.diff(dim=-1) >= 0).all()
