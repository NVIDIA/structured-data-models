# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import Timestral
from sdm.testing import withCUDA

WEIGHT_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


@withCUDA
@pytest.mark.parametrize("targets", [1, 2])
def test_forward_and_predict(device: torch.device, targets: int) -> None:
    model = Timestral(pretrained=False, device=device)
    x = TableTensor.from_pandas(
        pd.DataFrame(
            {
                "past": [0.0, 1, 2, 3, 4, 5, 6],
                "event": pd.Categorical(["a", "b", None, "a", "b", "a", "b"]),
            }
        ),
        stypes={"past": Stype.numerical, "event": Stype.categorical},
        device=device,
    )
    query = TableTensor.from_pandas(
        pd.DataFrame(
            {
                "event": pd.Categorical(
                    ["c", "b", "a"], categories=["c", "b", "a"]
                )
            }
        ),
        stypes={"event": Stype.categorical},
        device=device,
    )
    y = TableTensor.from_tensor(
        torch.arange(7 * targets, device=device).float().view(7, targets),
        columns=[f"y{i}" for i in range(targets)],
    )
    out = model(x, y, query)
    assert out.size() == (3, targets * 21)
    assert out.device == device
    assert out.numerical.isfinite().all()
    quantiles = out.numerical.unflatten(-1, (targets, 21))
    assert (quantiles[..., 1:] >= quantiles[..., :-1]).all()
    assert (out.numerical >= 0).all()
    assert "y0__q1" in out.column_names
    assert "y0__q50" in out.column_names
    assert "y0__q99" in out.column_names
    if targets == 1:
        # Missing learns 2.5, while unseen c uses the context mean of 3.
        encoded_context = TableTensor.from_tensor(
            torch.tensor(
                [
                    [0, 2.75],
                    [1, 3.5],
                    [2, 2.5],
                    [3, 2.75],
                    [4, 3.5],
                    [5, 2.75],
                    [6, 3.5],
                ],
                device=device,
            ),
            columns=["past", "event"],
        )
        encoded_query = TableTensor.from_tensor(
            torch.tensor([[3], [3.5], [2.75]], device=device),
            columns=["event"],
        )
        torch.testing.assert_close(
            model(encoded_context, y, encoded_query).numerical,
            out.numerical,
        )
    model.fit(x, y, context_only_columns=["past"])
    torch.testing.assert_close(model.predict(query).numerical, out.numerical)


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 1)])
def test_batch_independence(
    device: torch.device, batch_shape: tuple[int, ...]
) -> None:
    model = Timestral(pretrained=False, device=device)
    target = torch.randn(*batch_shape, 9, 2, device=device)
    x = TableTensor(size=(*batch_shape, 9), device=device)
    query = TableTensor(size=(*batch_shape, 3), device=device)
    out = model(x, TableTensor.from_tensor(target), query, num_estimators=1)
    expected = []
    for series in target.reshape(-1, 9, 2):
        expected.append(
            model(
                TableTensor(size=(9,), device=device),
                TableTensor.from_tensor(series),
                TableTensor(size=(3,), device=device),
            ).numerical
        )
    torch.testing.assert_close(
        out.numerical, torch.stack(expected).reshape(*batch_shape, 3, 42)
    )


@withCUDA
@pytest.mark.parametrize(
    "case",
    [
        "univariate",
        "multivariate",
        "missing",
        "all_missing",
        "constant",
        "positive",
        "numeric",
        "indicator",
        "categorical",
        "categorical_multi",
        "long_horizon",
    ],
)
def test_pretrained_parity(device: torch.device, case: str) -> None:
    upstream = pytest.importorskip("timestral")
    # Run the parity script first to cache weights for offline tests.
    try:
        hf_hub_download(
            "mistralai/Timestral-5M",
            "model.safetensors",
            revision=WEIGHT_REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip(
            "Run the Timestral parity script to cache the 5M checkpoint"
        )
    model = Timestral(device=device)
    reference = upstream.TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision=WEIGHT_REVISION,
        device=device,
        local_files_only=True,
    )
    t, horizon = 79, 1030 if case == "long_horizon" else 65
    v = 2 if case in {"multivariate", "categorical_multi"} else 1
    time = torch.arange(t + horizon, device=device).float()
    target = (time[:t] / 9).sin().expand(1, v, t).clone()
    if case == "missing":
        target[..., :8] = torch.nan
        target[..., 12:15] = torch.nan
    if case == "all_missing":
        target.fill_(torch.nan)
    if case == "constant":
        target.fill_(2)
    if case == "positive":
        target = target.abs() + 1
    x = TableTensor(size=(1, t), device=device)
    query = TableTensor(size=(1, horizon), device=device)
    covariates = {}
    if case in {"numeric", "indicator"}:
        known = (time / 7).sin()[None, None]
        if case == "indicator":
            known = (known > 0).float()
        covariates = {"known_covariates": known}
        x = TableTensor.from_tensor(known[..., :t].transpose(-2, -1))
        query = TableTensor.from_tensor(known[..., t:].transpose(-2, -1))
    if case.startswith("categorical"):
        codes = (time.long() % 4).view(1, -1, 1)
        codes[:, 3:6] = -1
        codes[:, t + 4 :] = 4
        known = TableTensor(
            columns={Stype.categorical: ["event"]},
            categorical=CategoricalTensor(
                codes, [torch.arange(5, device=device)]
            ),
        )
        x, query = known[..., :t, :], known[..., t:, :]
        covariates = {
            "categorical_known_covariates": codes.transpose(-2, -1)
            .float()
            .masked_fill(codes.transpose(-2, -1) < 0, torch.nan)
        }
    y = TableTensor.from_tensor(target.transpose(-2, -1))
    expected = (
        reference.forecast(
            y.numerical.transpose(-2, -1), query.size(-2), **covariates
        )
        .quantiles.transpose(-3, -2)
        .flatten(-2)
    )
    actual = model(x, y, query, num_estimators=1).numerical
    tol = 1e-4 if device.type == "cuda" else 2e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
