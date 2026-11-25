# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable

import pandas as pd
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain import PermutationExplainer
from sdm.models import ICLModel
from sdm.testing import withCUDA


@withCUDA
def test_permutation_importance_of_used_and_unused_column(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_pandas(
        df=pd.DataFrame({"n": range(32), "c": ["u", "v"] * 16}),
        stypes={"n": "numerical", "c": "categorical"},
        device=device,
    )
    y = TableTensor.from_tensor(10.0 * x.categorical.code)
    model = score_model(lambda columns: 10.0 * columns["c"])
    model.fit(x, y)

    out = PermutationExplainer(num_repeats=3).explain(model, x, y_query=y)

    assert out.size() == (3, 2)
    assert out.numerical_columns == ("n", "c")
    assert (out["n"].numerical == 0.0).all()
    assert (out["c"].numerical > 0.0).all()


@withCUDA
def test_permutation_importance_is_reproducible(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_tensor(torch.randn(16, 2, device=device))
    y = TableTensor.from_tensor(torch.randn(16, 1, device=device))
    model = score_model(lambda columns: columns["0"] - columns["1"])
    model.fit(x, y)
    explainer = PermutationExplainer()

    def explain(seed: int) -> Tensor:
        generator = torch.Generator(device).manual_seed(seed)
        out = explainer.explain(model, x, y_query=y, generator=generator)
        return out.numerical

    assert torch.equal(explain(0), explain(0))
    assert not torch.equal(explain(0), explain(1))
    assert not torch.equal(explain(0)[0], explain(0)[1])  # Two shuffles.


@withCUDA
def test_permutation_importance_uses_log_loss(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    # The model predicts the probability of "yes":
    x = TableTensor.from_tensor(
        tensor=torch.tensor([[0.2], [0.8]], device=device),
        columns=["p"],
    )
    y = TableTensor.from_pandas(
        df=pd.DataFrame({"y": ["no", "yes"]}),
        stypes={"y": "categorical"},
        device=device,
    )
    model = score_model(lambda columns: columns["p"])
    model.fit(x, y)

    out = PermutationExplainer(num_repeats=8).explain(model, x, y_query=y)

    # A shuffle keeps both rows, or swaps them to the probability 0.2 each:
    swapped = out.numerical > 1.0
    increase = math.log(0.8) - math.log(0.2)
    torch.testing.assert_close(
        out.numerical,
        increase * swapped.to(out.numerical.dtype),
    )


@withCUDA
def test_permutation_importance_with_custom_score(
    device: torch.device,
    score_model: Callable[..., ICLModel],
) -> None:
    x = TableTensor.from_tensor(
        tensor=torch.tensor([[0.0], [2.0]], device=device),
        columns=["a"],
    )
    model = score_model(lambda columns: columns["a"])
    model.fit(x, x)

    def accuracy(pred: TableTensor, target: TableTensor) -> float:
        correct = pred["pred"].numerical == target.numerical
        return float(correct.float().mean())

    explainer = PermutationExplainer(num_repeats=8, score=accuracy)
    out = explainer.explain(model, x, y_query=x)

    assert out.device == device
    # A shuffle keeps both rows, or swaps them to an accuracy of 0 (the
    # default score would decrease by the squared error of 4):
    swapped = out.numerical > 0.5
    torch.testing.assert_close(out.numerical, swapped.to(out.numerical.dtype))
