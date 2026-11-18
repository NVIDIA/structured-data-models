# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import HP, Correlated, CorrelatedContext, LogUniform, Uniform
from sdm.prior.root_samplers import (
    BallRootSampler,
    CorrelatedRootSampler,
    MixtureRootSampler,
    NormalRootSampler,
    RootSampler,
    UniformRootSampler,
)
from sdm.prior.util import random_weight


@pytest.mark.parametrize(
    ("mean", "std"),
    [(2.0, 0.5), (Uniform(-2.0, 2.0), LogUniform(0.1, 2.0))],
)
def test_normal_root_sampler(
    mean: HP[float] | float, std: HP[float] | float
) -> None:
    sampler = NormalRootSampler(mean=mean, std=std)
    actual = sampler.sample(
        num_rows=16, channels=3, generator=torch.Generator()
    )

    generator = torch.Generator()
    expected_mean = HP.from_value(mean).sample(generator=generator)
    expected_std = HP.from_value(std).sample(generator=generator)
    expected = expected_mean + expected_std * torch.randn(
        size=(16, 3), device="cpu", generator=generator
    )
    assert actual.shape == (16, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("low", "high"),
    [(-2.0, 3.0), (Uniform(-3.0, -1.0), Uniform(1.0, 3.0))],
)
def test_uniform_root_sampler(
    low: HP[float] | float, high: HP[float] | float
) -> None:
    actual = UniformRootSampler(low=low, high=high).sample(
        num_rows=16, channels=3, generator=torch.Generator()
    )
    generator = torch.Generator()
    lower = HP.from_value(low).sample(generator=generator)
    upper = HP.from_value(high).sample(generator=generator)
    expected = lower + torch.rand((16, 3), generator=generator) * (
        upper - lower
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert ((actual >= lower) & (actual < upper)).all()


@pytest.mark.parametrize("channels", [1, 3, 16])
def test_ball_root_sampler(channels: int) -> None:
    points = BallRootSampler().sample(num_rows=8192, channels=channels)
    radii = points.norm(dim=-1)
    assert (radii <= 1.0 + 1e-6).all()
    # Uniform volume means radius ** channels is uniform on [0, 1].
    torch.testing.assert_close(
        radii.pow(channels).mean(), torch.tensor(0.5), rtol=0, atol=0.03
    )
    assert (points.mean(dim=0).abs() < 0.05).all()


def test_correlated_root_sampler() -> None:
    base = UniformRootSampler()
    actual = CorrelatedRootSampler(base).sample(
        num_rows=16, channels=4, generator=torch.Generator()
    )

    generator = torch.Generator()
    points = base.sample(num_rows=16, channels=4, generator=generator)
    mixing = torch.randn((4, 4), generator=generator)
    importance = random_weight(size=(4,), generator=generator)
    # Weight input channels before projecting them into output channels.
    expected = (points * importance) @ mixing.T
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "sampler",
    [
        UniformRootSampler(low=Correlated(Uniform(-2.0, -1.0))),
        BallRootSampler(),
        CorrelatedRootSampler(UniformRootSampler()),
        MixtureRootSampler(
            [NormalRootSampler(), UniformRootSampler(), BallRootSampler()],
            key="roots",
        ),
    ],
)
def test_root_sampler_replay(sampler: RootSampler) -> None:
    def sample(generator: torch.Generator) -> list[torch.Tensor]:
        with CorrelatedContext(generator=generator):
            return [
                sampler.sample(
                    num_rows=rows, channels=channels, generator=generator
                )
                for rows, channels in [(1, 1), (16, 3), (7, 8)]
            ]

    actual = sample(torch.Generator())
    repeated = sample(torch.Generator())
    for points, replay, shape in zip(
        actual, repeated, [(1, 1), (16, 3), (7, 8)]
    ):
        assert points.shape == shape
        assert points.device.type == "cpu"
        assert points.isfinite().all()
        torch.testing.assert_close(points, replay, rtol=0, atol=0)


@pytest.mark.parametrize("correlated", [False, True])
def test_mixture_root_sampler(correlated: bool) -> None:
    sampler = MixtureRootSampler(
        samplers=[
            (NormalRootSampler(mean=torch.nan), 0.0),
            (UniformRootSampler(low=-2.0, high=-2.0), 1.0),
            (UniformRootSampler(low=3.0, high=3.0), 1.0),
        ],
        correlated=correlated,
    )
    with CorrelatedContext():
        for _ in range(8):
            points = sampler.sample(num_rows=16, channels=3)
            assert (points == -2.0).all() or (points == 3.0).all()
