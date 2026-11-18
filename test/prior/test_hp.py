# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import LogUniform, LogUniformInt, Uniform, UniformInt


@pytest.mark.parametrize(
    "distribution", [Uniform, UniformInt, LogUniform, LogUniformInt]
)
def test_reversed_bounds(
    distribution: type[Uniform]
    | type[UniformInt]
    | type[LogUniform]
    | type[LogUniformInt],
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        distribution(5, 2)


@pytest.mark.parametrize("distribution", [Uniform, LogUniform])
@pytest.mark.parametrize(
    ("low", "high"), [(float("nan"), 2.0), (1.0, float("nan"))]
)
def test_nan_bounds(
    distribution: type[Uniform] | type[LogUniform], low: float, high: float
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        distribution(low, high)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "distribution",
    [
        UniformInt(13, 20),
        UniformInt(2**24 + 1, 2**24 + 4),
        LogUniformInt(13, 20),
        LogUniformInt(13, 13),
    ],
)
def test_integer_transform_endpoints(
    distribution: UniformInt | LogUniformInt, dtype: torch.dtype
) -> None:
    x = torch.tensor([0.0, 1.0], dtype=dtype)
    expected = torch.tensor([distribution.low, distribution.high])

    torch.testing.assert_close(distribution.transform(x), expected)


@pytest.mark.parametrize("distribution", [LogUniform, LogUniformInt])
@pytest.mark.parametrize("low", [0, -1])
def test_nonpositive_log_bounds(
    distribution: type[LogUniform] | type[LogUniformInt], low: int
) -> None:
    sampler = distribution(low, 10)

    with pytest.raises(ValueError):  # noqa: PT011
        sampler.sample()
    with pytest.raises(ValueError):  # noqa: PT011
        sampler.transform(torch.tensor([0.0, 1.0]))
