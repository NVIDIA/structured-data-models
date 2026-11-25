# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pandas as pd
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain import LOCOExplainer
from sdm.models import ICLModel
from sdm.testing import withCUDA


def _mixed(columns: dict[str, Tensor]) -> Tensor | float:
    # Absent columns add nothing:
    return columns.get("n", 0.0) + 10.0 * columns.get("c", 0.0)


def _mixed_tables(device: torch.device) -> tuple[TableTensor, TableTensor]:
    # Category "u" has code 0 and category "v" has code 1:
    x = TableTensor.from_pandas(
        df=pd.DataFrame({"n": [1.0, 2.0, 6.0], "c": ["u", "v", "v"]}),
        stypes={"n": "numerical", "c": "categorical"},
        device=device,
    )
    y = torch.tensor([[1.0], [12.0], [16.0]], device=device)
    return x, TableTensor.from_tensor(y)  # The targets equal `_mixed`.


@withCUDA
def test_loco_is_loss_without_column(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x, y = _mixed_tables(device)

    out = LOCOExplainer().explain(
        score_model(_mixed),
        x,
        y_query=y,
        x_context=x,
        y_context=y,
    )

    assert out.numerical_columns == ("n", "c")
    # A model without a column misses its contribution to the target. The
    # mean of the predicted columns has no error with all columns:
    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[41.0 / 3.0, 200.0 / 3.0]], device=device),
    )


@withCUDA
def test_loco_with_custom_score(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x, y = _mixed_tables(device)

    def accuracy(pred: TableTensor, target: TableTensor) -> float:
        correct = pred["pred"].numerical == target.numerical
        return float(correct.float().mean())

    out = LOCOExplainer(score=accuracy).explain(
        score_model(_mixed),
        x,
        y_query=y,
        x_context=x,
        y_context=y,
    )

    # All rows are correct with all columns, none without "n", and only the
    # first row without "c":
    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[1.0, 2.0 / 3.0]], device=device),
    )


@withCUDA
def test_loco_shares_generator_state(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x, y = _mixed_tables(device)
    generator = torch.Generator(device)

    out = LOCOExplainer().explain(
        score_model(_mixed, noise=True),
        x,
        y_query=y,
        x_context=x,
        y_context=y,
        generator=generator,
    )

    # Every model call adds the next draw of the unchanged generator:
    noise = torch.rand(size=(), generator=generator, device=device)
    n, c = x.numerical[:, 0], 10.0 * x.categorical.code[:, 0]
    expected = torch.stack(
        [(noise - n).square().mean(), (noise - c).square().mean()]
    )
    torch.testing.assert_close(out.numerical[0], expected - noise.square())


def test_loco_keeps_fitted_model(
    score_model: Callable[..., ICLModel],
) -> None:
    x, y = _mixed_tables(torch.device("cpu"))
    model = score_model(_mixed)
    model.fit(x, y)

    LOCOExplainer().explain(
        model,
        x,
        y_query=y,
        x_context=x,
        y_context=y,
    )

    torch.testing.assert_close(model.predict(x)["pred"].numerical, y.numerical)
