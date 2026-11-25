# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pandas as pd
import torch

from sdm import TableTensor
from sdm.explain import ICEExplainer, PartialDependenceExplainer
from sdm.models import ICLModel
from sdm.testing import withCUDA


@withCUDA
def test_ice_grid_spans_quantiles(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    # The finite values 0, ..., 100 have the 5% and 95% quantiles 5 and 95:
    a = torch.arange(102.0, device=device)
    a[-1] = torch.nan
    b = torch.randn(102, device=device)
    x = TableTensor.from_tensor(
        tensor=torch.stack([a, b], dim=-1),
        columns=["a", "b"],
    )
    model = score_model(lambda columns: 2.0 * columns["a"] - columns["b"])
    model.fit(x, torch.zeros(102, 1, device=device))
    explainer = ICEExplainer(output="pred", grid_resolution=10)

    out = explainer.explain(model, x)

    assert list(out) == ["a", "b"]
    grid, curves = out["a"]
    expected = torch.linspace(5.0, 95.0, 10, device=device)
    assert grid.numerical_columns == ("a",)
    torch.testing.assert_close(grid.numerical, expected.unsqueeze(-1))
    # Every row is predicted with each grid value in place of its own:
    torch.testing.assert_close(
        curves,
        2.0 * expected.unsqueeze(-1) - b.unsqueeze(0),
    )


@withCUDA
def test_ice_grid_holds_observed_categories(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    table = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "n": [1.0, 2.0, 3.0, 4.0, 5.0],
                "c": ["u", "w", None, "u", "v"],
            }
        ),
        stypes={"n": "numerical", "c": "categorical"},
        device=device,
    )
    x = table[:4]  # Category "v" with code 2 is not observed.
    model = score_model(lambda columns: columns["n"] + 10.0 * columns["c"])
    model.fit(x, torch.zeros(4, 1, device=device))
    explainer = ICEExplainer(output="pred")

    grid, curves = explainer.explain(model, x)["c"]

    assert grid.to_pandas()["c"].tolist() == ["u", "w"]
    code = torch.tensor([[0.0], [1.0]], device=device)
    torch.testing.assert_close(curves, x.numerical.T + 10.0 * code)


def test_ice_of_selected_features(
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_tensor(torch.randn(8, 3))  # Columns "0" to "2".
    model = score_model(lambda columns: columns["0"] + columns["2"])
    model.fit(x, torch.zeros(8, 1))
    explainer = ICEExplainer(
        output="pred",
        features=["2", "0"],
        grid_resolution=4,
    )

    out = explainer.explain(model, x)

    assert list(out) == ["2", "0"]
    grid, curves = out["2"]
    torch.testing.assert_close(curves, grid.numerical + x.numerical[:, 0])


def test_ice_of_single_feature_name(
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_tensor(torch.randn(8, 11))  # Columns "0" to "10".
    model = score_model(lambda columns: columns["10"])
    model.fit(x, torch.zeros(8, 1))
    explainer = ICEExplainer(output="pred", features="10")

    assert list(explainer.explain(model, x)) == ["10"]


@withCUDA
def test_partial_dependence_averages_over_rows(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_tensor(
        tensor=torch.randn(16, 2, device=device),
        columns=["a", "b"],
    )
    model = score_model(lambda columns: columns["a"] * columns["b"])
    model.fit(x, torch.zeros(16, 1, device=device))
    explainer = PartialDependenceExplainer(
        output="pred",
        features="a",
        grid_resolution=5,
    )

    out = explainer.explain(model, x)

    assert list(out) == ["a"]
    grid, average = out["a"]
    assert grid.size() == (5, 1)
    # A grid value is multiplied by column "b" of every row:
    torch.testing.assert_close(
        average,
        grid.numerical[:, 0] * x.numerical[:, 1].mean(),
    )
