# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.models.kumo.timeseries.anomaly import MACSThreshold, SCSThreshold
from sdm.testing import withCUDA

KINDS = ("flat", "regimes", "oscillating", "offset_regimes")
SPIKES = [
    0,
    50,
    149,
    150,
    151,
    199,
    200,
    201,
    299,
    374,
    375,
    376,
    398,
    400,
    450,
    599,
]
# Generated with NVIDIA/Kumo-TS af74798, using the SDK strategy defaults.
SCS_INDICES = (
    [50, 149, 150, 151, 199, 299, 374, 375, 376, 398, 450, 599],
    [50, 149, 150, 151, 199, 299, 374, 375, 376, 398, 450, 599],
    SPIKES[1:],
    [50, 149, 199, 200, 201, 299, 374, 398, 400, 450, 599],
)


def _series(
    kind: str, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    time = torch.arange(600, device=device, dtype=dtype)
    target = torch.stack(
        [1 + 0.1 * (time / 20).sin(), 1 + 0.1 * (time / 17).cos()], dim=-1
    )
    scores = 0.1 + 0.02 * (time / 7).sin().square()
    if kind in ("regimes", "offset_regimes"):
        start, end = (200, 400) if kind == "regimes" else (150, 375)
        target[start:end] += 9
        target[end:] += 2
        scores[start:end] += 0.4
        scores[end:] += 0.1
    elif kind == "oscillating":
        target -= 1
    scores[SPIKES] += 2
    return scores, target


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("as_table", [False, True])
def test_upstream_decisions(
    device: torch.device, dtype: torch.dtype, kind: str, as_table: bool
) -> None:
    scores, target = _series(kind, device, dtype)
    scs = SCSThreshold()(
        scores, TableTensor.from_tensor(target) if as_table else target
    )
    macs = MACSThreshold()(scores)
    for actual, indices in (
        (scs, SCS_INDICES[KINDS.index(kind)]),
        (macs, SPIKES[1:]),
    ):
        expected = torch.zeros(600, device=device, dtype=torch.bool)
        expected[indices] = True
        assert actual.device == device
        torch.testing.assert_close(actual, expected)


@withCUDA
def test_batched_decisions(device: torch.device) -> None:
    series = [_series(kind, device, torch.float64) for kind in KINDS]
    scores = torch.stack([scores for scores, _ in series]).reshape(2, 2, 600)
    target = torch.stack([target for _, target in series]).reshape(
        2, 2, 600, 2
    )
    scs = SCSThreshold()(scores, target)
    macs = MACSThreshold()(scores)
    for index, (score, data) in enumerate(series):
        torch.testing.assert_close(
            scs[index // 2, index % 2], SCSThreshold()(score, data)
        )
        torch.testing.assert_close(
            macs[index // 2, index % 2], MACSThreshold()(score)
        )


@withCUDA
@pytest.mark.parametrize("length", [0, 1, 2, 100, 401])
@pytest.mark.parametrize("value", [0.0, 1.0, 1e7])
def test_constant_scores(
    device: torch.device, length: int, value: float
) -> None:
    scores = torch.full((2, length), value, device=device)
    target = torch.ones(2, length, 2, device=device)
    for flags in (SCSThreshold()(scores, target), MACSThreshold()(scores)):
        assert flags.shape == scores.shape
        assert flags.dtype == torch.bool
        assert not flags.any()


@withCUDA
@pytest.mark.parametrize(
    ("confidence", "first"), [(0.85, 2), (0.9, 7), (0.95, 7), (0.99, 20)]
)
def test_confidence_width(
    device: torch.device, confidence: float, first: int
) -> None:
    scores = torch.arange(20, device=device, dtype=torch.float64)
    expected = scores >= first
    scs = SCSThreshold(confidence_level=confidence, percentile_filter=0)
    macs = MACSThreshold(confidence_level=confidence, percentile_filter=0)
    torch.testing.assert_close(
        scs(scores, torch.ones(20, 1, device=device)), expected
    )
    torch.testing.assert_close(macs(scores), expected)


@withCUDA
def test_percentile_ties(device: torch.device) -> None:
    scores = torch.ones(100, device=device)
    scores[20:40] = 10
    target = torch.ones(100, 2, device=device)
    assert not SCSThreshold()(scores, target).any()
    assert not MACSThreshold()(scores).any()
    assert SCSThreshold(percentile_filter=70)(scores, target).any()
    assert MACSThreshold(percentile_filter=70)(scores).any()
