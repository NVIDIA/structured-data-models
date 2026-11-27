# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

import sdm
from sdm import TableTensor
from sdm.explain import RecontextualizedShapleyExplainer, ShapleyExplainer
from sdm.models import ICLModel
from sdm.testing import withCUDA

pytest.importorskip("shapiq")


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("imputer", ["marginal", "baseline"])
def test_shapley_explainer(
    imputer: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    pytest.importorskip("shapiq")

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

    output = "q500" if dtype.is_floating_point else "0"
    prediction = model.predict(x[:2])[output].numerical.squeeze(-1).cpu()

    out = ShapleyExplainer(imputer=imputer).explain(
        model,
        x[:2],
        output=output,
        background=x[:4],
        budget=4,
    )
    assert len(out) == 2
    for value, expected in zip(out, prediction):
        attributions = value.get_n_order_values(1)
        assert attributions.shape == (2,)
        assert np.isfinite(attributions).all()
        np.testing.assert_allclose(
            attributions.sum() + value.baseline_value,
            expected.numpy(),
            atol=1e-5,
        )


WEIGHT = {"a": 2.0, "b": -3.0}
FILL = {"a": 1.0, "b": 2.0}


def _linear(columns: dict[str, Tensor]) -> Tensor | float:
    # Absent columns add nothing, and missing values count as `FILL`:
    return sum(
        WEIGHT[name] * value.nan_to_num(FILL[name])
        for name, value in columns.items()
    )


def _product(columns: dict[str, Tensor]) -> Tensor | float:
    # Columns "a" and "b" only matter together, and absent columns are 0:
    return columns.get("a", 0.0) * columns.get("b", 0.0) + columns.get(
        "c", 0.0
    )


def _table(rows: Any, device: torch.device | None = None) -> TableTensor:
    # Return rows `[R, D]` as a table with the columns "a", "b" and "c".
    tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
    return TableTensor.from_tensor(
        tensor, columns=["a", "b", "c"][: tensor.size(-1)]
    )


def _fit(model: ICLModel, x: TableTensor) -> ICLModel:
    model.fit(x, torch.zeros(len(x), 1, device=x.device))
    return model


def _first_order(values: Sequence[Any]) -> Tensor:  # [R, D]
    return torch.tensor(
        np.stack([value.get_n_order_values(1) for value in values]),
        dtype=torch.float32,
    )


def _base(values: Sequence[Any]) -> Tensor:  # [R]
    return torch.tensor([value.baseline_value for value in values])


@withCUDA
def test_recontextualized_shapley_values_drop_columns(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = _table(torch.randn(3, 2), device)
    generator = torch.Generator(device)
    state = generator.get_state()
    explainer = RecontextualizedShapleyExplainer(output="pred", budget=4)

    values = explainer.explain(
        score_model(_linear, noise=True),
        x,
        x_context=x,
        y_context=TableTensor.from_tensor(torch.zeros(3, 1, device=device)),
        generator=generator,
    )

    # A dropped column adds nothing, and every model call adds the next draw
    # of the unchanged generator:
    assert torch.equal(generator.get_state(), state)
    weight = torch.tensor(list(WEIGHT.values()))
    torch.testing.assert_close(
        _first_order(values),
        weight * x.numerical.cpu(),
    )
    noise = torch.rand(size=(), generator=generator, device=device)
    torch.testing.assert_close(_base(values), noise.cpu().expand(3))


@pytest.mark.parametrize("index", [None, "FSII"])
def test_recontextualized_shapley_interaction_keeps_fitted_model(
    score_model: Callable[..., ICLModel],
    index: str | None,
) -> None:
    x = _table([[2.0, 3.0]])
    y = TableTensor.from_tensor(torch.zeros(1, 1))
    model = _fit(score_model(_product), x)
    explainer = RecontextualizedShapleyExplainer(
        output="pred",
        budget=4,
        index=index,
        max_order=2,
    )

    (value,) = explainer.explain(model, x, x_context=x, y_context=y)

    assert value.index == (index or "k-SII")
    assert value[(0, 1)] == pytest.approx(6.0)
    assert value[(0,)] == pytest.approx(0.0, abs=1e-5)
    assert value[(1,)] == pytest.approx(0.0, abs=1e-5)
    assert value.baseline_value == pytest.approx(0.0, abs=1e-5)
    assert model.predict(x)["pred"].numerical.item() == 6.0


def test_recontextualized_shapley_values_of_sampled_column_subsets(
    score_model: Callable[..., ICLModel],
) -> None:
    # Two equal rows, whose columns interact:
    x = TableTensor.from_tensor(torch.arange(1.0, 7.0).repeat(2, 1))
    model = score_model(lambda columns: sum(columns.values()) ** 3)
    # Sample a quarter of the column subsets:
    explainer = RecontextualizedShapleyExplainer(output="pred", budget=16)

    def explain(seed: int | None = None) -> Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        values = explainer.explain(
            model,
            x,
            x_context=x,
            y_context=TableTensor.from_tensor(torch.zeros(2, 1)),
            generator=generator,
        )
        return _first_order(values)

    unseeded = explain()
    assert torch.equal(unseeded[0], unseeded[1])  # Rows share subsets.
    assert torch.equal(explain(0), explain(0))
    assert not torch.equal(explain(0), explain(1))
