# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import KumoTabular
from sdm.testing import onlyMPS


def _build(
    task: Literal["classification", "regression"],
    size: Literal["small", "medium", "large"],
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
def cls_model(size: Literal["small", "medium", "large"]) -> KumoTabular:
    return _build("classification", size)


@pytest.fixture
def reg_model(size: Literal["small", "medium", "large"]) -> KumoTabular:
    return _build("regression", size)


@pytest.fixture(params=["small", "medium", "large"])
def size(
    request: pytest.FixtureRequest,
) -> Literal["small", "medium", "large"]:
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


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("estimator_batch_size", [2, None])
def test_estimator_batching(
    task: Literal["classification", "regression"],
    size: Literal["small", "medium", "large"],
    estimator_batch_size: int | None,
) -> None:
    model = _build(task, size)
    x_context, x_query = _features()
    target = _cls_target() if task == "classification" else _reg_target()
    # Match the shuffled features and target categories in both executions.
    expected = model(
        x_context=x_context,
        y_context=target,
        x_query=x_query,
        num_estimators=5,
        generator=torch.Generator().manual_seed(0),
    )

    actual = model(
        x_context=x_context,
        y_context=target,
        x_query=x_query,
        num_estimators=5,
        estimator_batch_size=estimator_batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    assert actual.columns == expected.columns
    assert actual.dtype == expected.dtype
    assert torch.is_inference(actual)
    torch.testing.assert_close(
        actual=actual.numerical,
        expected=expected.numerical,
        atol=1e-4,
        rtol=1e-4,
    )

    model.fit(
        x=x_context,
        y=target,
        num_estimators=5,
        estimator_batch_size=estimator_batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    actual = model.predict(x_query)
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual=actual.numerical,
        expected=expected.numerical,
        atol=1e-4,
        rtol=1e-4,
    )


def test_fit_predict(
    cls_model: KumoTabular,
    reg_model: KumoTabular,
) -> None:
    x_context, x_query = _features()
    for num_classes in (3, 11):
        target = _cls_target(num_classes)
        expected = cls_model(
            x_context=x_context,
            y_context=target,
            x_query=x_query,
            recipe=_recipe(),
            generator=torch.Generator().manual_seed(0),
        )
        cls_model.fit(
            x=x_context,
            y=target,
            recipe=_recipe(),
            generator=torch.Generator().manual_seed(0),
        )
        actual = cls_model.predict(x_query)

        assert actual.size() == (2, num_classes)
        assert actual.allclose(expected, atol=1e-5)
        assert actual.columns == expected.columns

    x_context, x_query = _features(Stype.numerical)
    target = _reg_target()
    recipe = _recipe()
    expected = reg_model(x_context, target, x_query, recipe=recipe)
    reg_model.fit(x_context, target, recipe=recipe)
    actual = reg_model.predict(x_query)

    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=1e-4,
        rtol=1e-4,
    )


def test_missing_values_pass_through_fit_predict(
    size: Literal["small", "medium", "large"],
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


@pytest.mark.parametrize("estimator_batch_size", [2, None])
def test_estimator_batching_many_classes(
    estimator_batch_size: int | None,
) -> None:
    # More than 10 classes run through ECOC codebooks drawn per estimator.
    model = _build("classification", "small")
    x_context, x_query = TableTensor.from_tensor(torch.randn(28, 4)).split(
        24, dim=0
    )
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.arange(24).remainder(12).unsqueeze(-1),
            categories=(torch.arange(12),),
        ),
    )

    def forward(size: int | None) -> TableTensor:
        return model(
            x_context=x_context,
            y_context=target,
            x_query=x_query,
            num_estimators=4,
            estimator_batch_size=size,
            generator=torch.Generator().manual_seed(0),
        )

    expected = forward(1)
    actual = forward(estimator_batch_size)
    assert actual.columns == expected.columns
    torch.testing.assert_close(actual.numerical, expected.numerical)

    model.fit(
        x=x_context,
        y=target,
        num_estimators=4,
        estimator_batch_size=estimator_batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    actual = model.predict(x_query)
    assert actual.columns == expected.columns
    torch.testing.assert_close(actual.numerical, expected.numerical)


@onlyMPS
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_default_recipe_on_mps(
    task: Literal["classification", "regression"],
) -> None:
    model = _build(task, "small").to("mps")
    if task == "classification":
        x_context, x_query = _features()
        target = _cls_target()
    else:
        x_context, x_query = _features(Stype.numerical)
        target = _reg_target()

    x_context, x_query, target = (
        x_context.to("mps"),
        x_query.to("mps"),
        target.to("mps"),
    )

    out1 = model(x_context, target, x_query, num_estimators=4)
    assert out1.device == torch.device("mps", 0)
    assert out1.numerical.isfinite().all()

    model.fit(x_context, target, num_estimators=4)
    out2 = model.predict(x_query)
    assert out2.device == torch.device("mps", 0)
    assert out2.size() == out1.size()
    assert out2.numerical.isfinite().all()


@pytest.mark.parametrize("num_classes", [0, 2, 10, 11, 100])
@pytest.mark.parametrize(
    ("num_rows", "num_columns"), [(1, 1), (3000, 25), (10000, 500)]
)
def test_batch_estimates_match_tabarena(
    size: Literal["small", "medium", "large"],
    num_classes: int,
    num_rows: int,
    num_columns: int,
) -> None:
    # Freeze the FP16 formulas from ValterH/tabarena PR #1 at ef7ab98.
    task: Literal["classification", "regression"] = (
        "classification" if num_classes else "regression"
    )
    model = KumoTabular(task=task, size=size, pretrained=False, device="meta")
    cell_channels, icl_channels, num_layers = {
        "small": (128, 512, 12),
        "medium": (256, 512, 24),
        "large": (256, 1024, 24),
    }[size]
    tasks = (
        max(
            math.ceil(num_classes / 9),
            4 * math.ceil(math.log(num_classes, 10)),
        )
        if num_classes > 10
        else 1
    )
    workspace = (
        tasks
        * 2
        * (
            4 * (min(2 * num_columns, 500) + 4) * cell_channels
            + 15 * icl_channels
        )
    )
    cache = tasks * 2 * 2 * icl_channels * num_layers
    for memory_budget in (0, 2**29 + 1, 2**30):
        expected = max(
            1,
            min(16, (memory_budget // 2) // (num_rows * (workspace + cache))),
        )
        assert (
            model.estimate_estimator_batch_size(
                num_rows=num_rows,
                num_columns=num_columns,
                num_estimators=16,
                memory_budget=memory_budget,
                num_classes=num_classes,
            )
            == expected
        )
        for estimator_batch_size in (1, 4, 16, 32):
            row_bytes = workspace * estimator_batch_size + 16 * 8 * (
                8 * num_columns + 4 * (num_classes or 999)
            )
            expected = max(1, (memory_budget // 2) // row_bytes)
            assert (
                model.estimate_query_batch_size(
                    num_columns=num_columns,
                    num_estimators=16,
                    estimator_batch_size=estimator_batch_size,
                    memory_budget=memory_budget,
                    num_classes=num_classes,
                )
                == expected
            )
