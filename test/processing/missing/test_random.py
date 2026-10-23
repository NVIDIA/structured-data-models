# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from io import BytesIO
from typing import cast

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.testing import onlyCUDA, withCUDA


def test_masks_cells_at_the_configured_rate() -> None:
    rate = 0.05
    table = TableTensor(
        numerical=torch.randn(4, 4000, 3),
        categorical=CategoricalTensor(
            code=torch.randint(3, (4, 4000, 2)),
            categories=(torch.arange(3),) * 2,
        ),
    )

    output = sp.MaskRandom(rate=(rate, rate)).fit_transform(table)

    numerical_missing = output.numerical.isnan()
    categorical_missing = output.categorical.code == -1
    missing = torch.cat((numerical_missing, categorical_missing), dim=-1)
    actual_rate = missing.float().mean(dim=(-2, -1))
    torch.testing.assert_close(
        actual_rate,
        torch.full_like(actual_rate, rate),
        rtol=0.0,
        atol=0.01,
    )
    assert output.numerical[~numerical_missing].equal(
        table.numerical[~numerical_missing]
    )
    assert output.categorical.code[~categorical_missing].equal(
        table.categorical.code[~categorical_missing]
    )
    assert all(
        actual.equal(expected)
        for actual, expected in zip(
            output.categorical.categories,
            table.categorical.categories,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "processor",
    [
        sp.MaskRandom(rate=(0.1, 0.3)),
        sp.Sequential(sp.MaskRandom(rate=(0.1, 0.3)), sp.Identity()),
        sp.Choice(sp.MaskRandom(rate=(0.1, 0.3)), method="round_robin"),
        sp.StypeDispatch(numerical=sp.MaskRandom(rate=(0.1, 0.3))),
    ],
    ids=lambda processor: type(processor).__name__,
)
@withCUDA
def test_transforms_advance_and_refitting_reproduces_the_stream(
    processor: sp.Processor,
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(torch.ones(1000, 4, device=device))
    processor = deepcopy(processor)
    processor.fit(
        table, generator=torch.Generator(device=device).manual_seed(0)
    )

    first = processor.transform(table)
    second = processor.transform(table)
    shorter = processor.transform(table[:97])

    assert not first.numerical.isnan().equal(second.numerical.isnan())
    assert shorter.size() == table[:97].size()

    processor.fit(
        table, generator=torch.Generator(device=device).manual_seed(0)
    )
    repeated = processor.transform(table)
    assert first.allclose(repeated, rtol=0, atol=0, equal_nan=True)


@withCUDA
def test_save_and_load_continues_the_random_stream(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(torch.ones(1000, 4, device=device))
    processor = sp.MaskRandom(rate=(0.1, 0.3)).fit(table)
    processor.transform(table)

    checkpoint = BytesIO()
    torch.save(processor.state_dict(), checkpoint)
    expected = processor.transform(table)

    checkpoint.seek(0)
    restored = sp.MaskRandom(rate=(0.1, 0.3))
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    actual = restored.transform(table)
    assert actual.allclose(expected, rtol=0, atol=0, equal_nan=True)


@onlyCUDA
def test_cuda_random_stream_continues_on_cpu() -> None:
    table = TableTensor.from_tensor(torch.ones(1000, 4, device="cuda"))
    processor = sp.MaskRandom(rate=(0.1, 0.3)).fit(table)
    processor.transform(table)
    state = deepcopy(processor.state_dict())
    expected = processor.transform(table)

    processor.load_state_dict(state)
    actual = processor.cpu().transform(cast(TableTensor, table.cpu()))
    assert actual.allclose(expected.cpu(), rtol=0, atol=0, equal_nan=True)
