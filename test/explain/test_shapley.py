# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain import ShapleyExplainer
from sdm.models import ICLModel
from sdm.testing import withCUDA

pytest.importorskip("shapiq")

WEIGHT = {"a": 2.0, "b": -3.0}
FILL = {"a": 1.0, "b": 2.0}


def _linear(columns: dict[str, Tensor]) -> Tensor | float:
    # Missing values count as `FILL`:
    return sum(
        WEIGHT[name] * value.nan_to_num(FILL[name])
        for name, value in columns.items()
    )


def _product(columns: dict[str, Tensor]) -> Tensor | float:
    # Columns "a" and "b" only matter together, and "c" may be absent:
    return columns["a"] * columns["b"] + columns.get("c", 0.0)


def _mixed(columns: dict[str, Tensor]) -> Tensor:
    # A missing "n" counts as 4:
    return columns["n"].nan_to_num(4.0) + 10.0 * columns["c"]


def _mixed_tables(device: torch.device) -> tuple[TableTensor, TableTensor]:
    # Category "u" has code 0 and category "v" has code 1:
    x = TableTensor.from_pandas(
        df=pd.DataFrame({"n": [1.0, 2.0, 6.0], "c": ["u", "v", "v"]}),
        stypes={"n": "numerical", "c": "categorical"},
        device=device,
    )
    y = torch.tensor([[1.0], [12.0], [16.0]], device=device)
    return x, TableTensor.from_tensor(y)  # The targets equal `_mixed`.


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
@pytest.mark.parametrize(
    ("imputer", "reference"),
    [
        ("baseline", [0.5, -1.0]),  # The mean of the background rows.
        ("marginal", [0.5, -1.0]),
        ("missing", [1.0, 2.0]),  # The values the model fills in.
    ],
)
def test_shapley_values_of_linear_model(
    device: torch.device,
    score_model: Callable[..., ICLModel],
    imputer: Literal["baseline", "marginal", "missing"],
    reference: list[float],
) -> None:
    x = _table(torch.randn(5, 2), device)
    background = _table([[0.0, 1.0], [1.0, -3.0]], device)
    model = _fit(score_model(_linear), background)
    explainer = ShapleyExplainer(output="pred", budget=4, imputer=imputer)

    values = explainer.explain(
        model,
        x,
        background=None if imputer == "missing" else background,
    )

    weight = torch.tensor(list(WEIGHT.values()))
    torch.testing.assert_close(
        _first_order(values),
        weight * (x.numerical.cpu() - torch.tensor(reference)),
    )
    torch.testing.assert_close(
        _base(values),
        (weight * torch.tensor(reference)).sum().expand(5),
    )
    # The values of a row sum to its prediction minus the base value:
    torch.testing.assert_close(
        _first_order(values).sum(dim=-1) + _base(values),
        model.predict(x)["pred"].numerical.squeeze(-1).cpu(),
    )


@withCUDA
def test_shapley_values_marginalize_over_background_rows(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = _table([[1.0, 1.0]], device)
    background = _table([[-2.0, -2.0], [2.0, 2.0]], device)
    model = _fit(score_model(_product), background)
    explainer = ShapleyExplainer(output="pred", budget=4, imputer="marginal")

    values = explainer.explain(model, x, background=background)

    # Both columns of a background row replace jointly, unlike their means:
    assert _base(values).tolist() == pytest.approx([4.0])
    assert _first_order(values).tolist() == [pytest.approx([-1.5, -1.5])]


@withCUDA
@pytest.mark.parametrize(
    ("background", "base", "expected"),
    [
        # The base value is the prediction of a single background row:
        ([[1.0, 1.0]], -1.0, [4.0, -9.0]),
        # Column "a" is replaced by the mean 0.5 of its observed values:
        ([[0.0, 1.0], [float("nan"), -3.0], [1.0, -1.0]], 4.0, [5.0, -15.0]),
    ],
)
def test_shapley_values_of_background_rows(
    device: torch.device,
    score_model: Callable[..., ICLModel],
    background: list[list[float]],
    base: float,
    expected: list[float],
) -> None:
    x = _table([[3.0, 4.0]], device)
    model = _fit(score_model(_linear), _table(background, device))
    explainer = ShapleyExplainer(output="pred", budget=4)

    values = explainer.explain(model, x, background=_table(background, device))

    assert _base(values).tolist() == pytest.approx([base])
    assert _first_order(values).tolist() == [pytest.approx(expected)]


@withCUDA
@pytest.mark.parametrize(
    ("imputer", "base", "expected"),
    [
        ("baseline", 13.0, [-2.0, -10.0]),  # Mean 3 and most frequent "v".
        ("marginal", 29.0 / 3.0, [-2.0, -20.0 / 3.0]),  # Mean 3 and code 2/3.
        ("missing", -6.0, [-3.0, 10.0]),  # Filled 4 and missing code -1.
    ],
)
def test_shapley_values_of_categorical_column(
    device: torch.device,
    score_model: Callable[..., ICLModel],
    imputer: Literal["baseline", "marginal", "missing"],
    base: float,
    expected: list[float],
) -> None:
    x, y = _mixed_tables(device)
    model = score_model(_mixed)
    model.fit(x, y)
    explainer = ShapleyExplainer(output="pred", budget=4, imputer=imputer)

    values = explainer.explain(
        model,
        x[:1],
        background=None if imputer == "missing" else x,
    )

    assert _base(values).tolist() == pytest.approx([base])
    assert _first_order(values).tolist() == [pytest.approx(expected)]


@withCUDA
def test_shapley_values_match_categories_of_background(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    background, y = _mixed_tables(device)
    # Category "v" has code 0 in this table, but code 1 in the background,
    # and the numerical column comes first in the values:
    x = TableTensor.from_pandas(
        df=pd.DataFrame({"c": ["v"], "n": [5.0]}),
        stypes={"c": "categorical", "n": "numerical"},
        device=device,
    )
    model = score_model(_mixed)
    model.fit(background, y)
    explainer = ShapleyExplainer(output="pred", budget=4)

    values = explainer.explain(model, x, background=background)

    # The row holds the most frequent category, and 2 above the mean of 3:
    assert _base(values).tolist() == pytest.approx([13.0])
    assert _first_order(values).tolist() == [
        pytest.approx([2.0, 0.0], abs=1e-5)
    ]


def test_shapley_values_reject_background_of_other_schema(
    score_model: Callable[..., ICLModel],
) -> None:
    x = _table(torch.randn(1, 2))
    background = TableTensor.from_tensor(torch.randn(2, 2), columns=["b", "a"])
    explainer = ShapleyExplainer(output="pred", budget=4)

    with pytest.raises(ValueError, match="share the same schema"):
        explainer.explain(
            _fit(score_model(_linear), x),
            x,
            background=background,
        )


@withCUDA
@pytest.mark.parametrize("index", [None, "FSII"])
def test_shapley_interaction_of_product_model(
    device: torch.device,
    score_model: Callable[..., ICLModel],
    index: str | None,
) -> None:
    x = _table([[2.0, 3.0, 5.0]], device)
    background = _table(torch.zeros(2, 3), device)
    model = _fit(score_model(_product), background)
    explainer = ShapleyExplainer(
        output="pred",
        budget=8,
        index=index,
        max_order=2,
    )

    (value,) = explainer.explain(model, x, background=background)

    assert value.index == (index or "k-SII")
    assert value[(0, 1)] == pytest.approx(6.0)
    assert value[(0, 2)] == pytest.approx(0.0, abs=1e-5)
    assert value[(1, 2)] == pytest.approx(0.0, abs=1e-5)
    assert value[(2,)] == pytest.approx(5.0)


@withCUDA
@pytest.mark.parametrize("budget", [8, 16])  # Half or all column subsets.
def test_shapley_values_are_reproducible(
    device: torch.device,
    score_model: Callable[..., ICLModel],
    budget: int,
) -> None:
    x = TableTensor.from_tensor(torch.randn(2, 4, device=device))
    background = TableTensor.from_tensor(torch.randn(8, 4, device=device))
    model = score_model(lambda c: c["0"] * c["1"] + c["2"] * c["3"])
    model = _fit(model, background)
    # Two sampled background rows vary even with all column subsets:
    explainer = ShapleyExplainer(
        output="pred",
        budget=budget,
        imputer="marginal",
        sample_size=2,
    )

    def explain(seed: int) -> Tensor:
        values = explainer.explain(
            model,
            x,
            background=background,
            generator=torch.Generator(device).manual_seed(seed),
        )
        return _first_order(values)

    assert torch.equal(explain(0), explain(0))
    assert not torch.equal(explain(0), explain(1))


def test_shapley_values_of_selected_class(
    score_model: Callable[..., ICLModel],
) -> None:
    x = _table(torch.rand(4, 2))
    y = TableTensor.from_pandas(
        df=pd.DataFrame({"y": ["no", "yes", "yes", "no"]}),
        stypes={"y": "categorical"},
    )
    # The model predicts the probability of "yes":
    model = score_model(lambda c: 0.5 * c["a"] + 0.25 * c["b"])
    model.fit(x, y)

    yes = ShapleyExplainer(output="yes", budget=4)
    yes = yes.explain(model, x, background=x)
    no = ShapleyExplainer(output="no", budget=4)
    no = no.explain(model, x, background=x)

    weight = torch.tensor([0.5, 0.25])
    expected = weight * (x.numerical - x.numerical.mean(dim=0))
    torch.testing.assert_close(_first_order(yes), expected)
    torch.testing.assert_close(_first_order(no), -expected)
    torch.testing.assert_close(_base(yes), 1.0 - _base(no))
