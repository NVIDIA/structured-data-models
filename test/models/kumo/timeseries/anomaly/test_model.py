# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm import Recipe, TableTensor
from sdm.models import KumoAnomaly
from sdm.models.kumo.timeseries.anomaly.threshold import (
    macs_threshold,
    scs_threshold,
)
from sdm.processing import Callable
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(pretrained=False, device=device)
    x = torch.randn(2, 130, 3, device=device)

    out = model(x, num_samples=1, num_steps=1)
    assert out.size() == (2, 130, 3)
    assert out.columns["numerical"] == ("mae",)
    assert out.columns["categorical"] == ("scs", "macs")
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert (out.numerical >= 0).all()
    assert out.categorical.code.shape == (2, 130, 2)
    for categories in out.categorical.categories:
        torch.testing.assert_close(
            categories, torch.tensor([False, True], device=device)
        )


@withCUDA
def test_threshold_preprocessed_series(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = torch.arange(600, device=device, dtype=torch.float64)
    x = torch.stack([(time / 20).sin(), (time / 17).cos()], dim=-1) * 0.1
    scores = 0.1 + 0.02 * (time / 7).sin().square()
    scores[[0, 50, 199, 200, 201, 299, 398, 400, 450, 599]] += 2
    model = KumoAnomaly(pretrained=False, device=device)

    def reconstruct(x: Tensor, **kwargs: object) -> Tensor:
        return scores.expand(x.shape[:-1])

    monkeypatch.setattr(model.model, "forward", reconstruct)
    recipe = Recipe(
        features=Callable(
            lambda table: TableTensor.from_tensor(table.numerical + 10)
        ),
        output=Callable(lambda table: table[..., :300, :]),
    )
    out = model(x, recipe=recipe)

    torch.testing.assert_close(out.numerical.squeeze(-1), scores[:300])
    expected_scs = scs_threshold(scores, x + 10)
    expected_macs = macs_threshold(scores)
    expected = torch.stack([expected_scs, expected_macs], dim=-1).long()
    torch.testing.assert_close(out.categorical.code, expected[:300])
    assert not torch.equal(expected_scs, scs_threshold(scores, x))
