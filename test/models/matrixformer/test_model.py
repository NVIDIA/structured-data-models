# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Literal

import pytest
import torch

from sdm import TableTensor
from sdm.models import MatrixFormer
from sdm.models.matrixformer import model as matrixformer_module
from sdm.testing import onlyFullTest, withCUDA


@pytest.fixture
def small_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        matrixformer_module,
        "_MatrixFormer",
        functools.partial(
            matrixformer_module._MatrixFormer,
            channels=32,
            hidden_channels=64,
            num_layers=2,
            num_heads=4,
            num_cls=2,
            num_bins=16,
        ),
    )


@withCUDA
@pytest.mark.usefixtures("small_model")
@pytest.mark.parametrize("variant", ["plus", "impute", "recsys"])
def test_forward(
    device: torch.device,
    variant: Literal["plus", "impute", "recsys"],
) -> None:
    model = MatrixFormer(variant=variant, pretrained=False, device=device)
    # Make predictions sensitive to the observed entries.
    for parameter in model.parameters():
        if parameter.requires_grad and not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    if device.type == "cpu":
        assert repr(model) == f"MatrixFormer(variant={variant!r})"
    else:
        assert (
            repr(model) == f"MatrixFormer(variant={variant!r}, device=cuda:0)"
        )

    x = torch.randn(2, 9, 5, device=device)
    x[torch.rand(2, 9, 5, device=device) < 0.3] = torch.nan
    missing = x.isnan()

    out = model(x)
    assert isinstance(out, TableTensor)
    assert out.size() == (2, 9, 5)
    assert out.dtype == x.dtype
    assert out.device == device
    assert torch.is_inference(out)
    assert out.numerical[~missing].equal(x[~missing])
    assert out.numerical.isfinite().all()

    # Each matrix in a batch is completed independently.
    single = model(x[1])
    assert single.numerical.allclose(out.numerical[1], atol=1e-5)

    # Overlapping windows still complete every entry.
    windowed = model(x[0], max_rows=4, max_cols=3)
    assert windowed.numerical[~missing[0]].equal(x[0][~missing[0]])
    assert windowed.numerical.isfinite().all()


@onlyFullTest
def test_pretrained() -> None:
    model = MatrixFormer(variant="plus")

    generator = torch.Generator().manual_seed(0)
    u = torch.randn(60, 3, generator=generator)
    v = torch.randn(3, 12, generator=generator)
    complete = u @ v + 0.1 * torch.randn(60, 12, generator=generator)
    missing = torch.rand(60, 12, generator=generator) < 0.3
    x = complete.masked_fill(missing, torch.nan)

    out = model(x).numerical
    mean = x.nanmean(dim=0, keepdim=True).expand_as(x)
    error = (out - complete)[missing].square().mean()
    baseline = (mean - complete)[missing].square().mean()
    assert error < 0.5 * baseline
