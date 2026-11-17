# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import Correlated, CorrelatedContext, LogUniform, LogUniformInt
from sdm.prior.activations import CallableActivation, softmax
from sdm.prior.matrix_samplers import (
    ActivationMatrixSampler,
    GaussianMatrixSampler,
    KernelMatrixSampler,
    MatrixSampler,
    MixtureMatrixSampler,
    PowerLawMatrixSampler,
    SingularValuesMatrixSampler,
)


@pytest.mark.parametrize("size", [(4, 7), (3, 4, 7)])
def test_random_gaussian_matrix(size: tuple[int, ...]) -> None:
    actual = GaussianMatrixSampler().sample(
        size=size, dtype=torch.float64, generator=torch.Generator()
    )

    generator = torch.Generator()
    expected = torch.randn(size, dtype=torch.float64, generator=generator)
    noise = torch.randn(size, dtype=torch.float64, generator=generator)
    expected.add_(noise, alpha=1e-6)
    expected /= expected.norm(dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "size", [(1, 1), (4, 7), (7, 4), (2, 4, 7), (1, 3, 4, 7)]
)
@pytest.mark.parametrize(
    "sampler",
    [
        GaussianMatrixSampler(),
        PowerLawMatrixSampler(),
        SingularValuesMatrixSampler(),
        KernelMatrixSampler(num_centers=3, gamma=1.0),
        ActivationMatrixSampler(
            matrix_sampler=GaussianMatrixSampler(normalize=False),
            activation=CallableActivation(torch.Tensor.tanh),
        ),
        MixtureMatrixSampler(
            [
                GaussianMatrixSampler(),
                KernelMatrixSampler(num_centers=3, gamma=1.0),
            ]
        ),
    ],
)
def test_matrix_shape_norm_and_replay(
    sampler: MatrixSampler, size: tuple[int, ...], dtype: torch.dtype
) -> None:
    generator = torch.Generator()
    state = generator.get_state()
    with CorrelatedContext(generator=generator):
        actual = sampler.sample(size, dtype=dtype, generator=generator)
    generator.set_state(state)
    with CorrelatedContext(generator=generator):
        replay = sampler.sample(size, dtype=dtype, generator=generator)

    assert actual.shape == size
    assert actual.dtype == dtype
    assert actual.device.type == "cpu"
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, replay, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.norm(dim=-1), torch.ones(size[:-1], dtype=dtype)
    )


@pytest.mark.parametrize("size", [(4, 7), (2, 3, 4, 7)])
def test_activation_matrix_uses_raw_flattened_entries(
    size: tuple[int, ...],
) -> None:
    actual_generator = torch.Generator()
    actual = ActivationMatrixSampler(
        matrix_sampler=GaussianMatrixSampler(normalize=False),
        activation=CallableActivation(softmax),
    ).sample(size, dtype=torch.float64, generator=actual_generator)

    generator = torch.Generator()
    raw = torch.randn(size, dtype=torch.float64, generator=generator)
    expected = raw.flatten(-2).softmax(-1).view_as(raw)
    expected.add_(
        torch.randn(size, dtype=torch.float64, generator=generator), alpha=1e-3
    )
    expected.add_(
        torch.randn(size, dtype=torch.float64, generator=generator), alpha=1e-6
    )
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_generator.get_state(), generator.get_state())


@pytest.mark.parametrize("normalize", [False, True])
def test_nested_matrix_mixture_preserves_component(normalize: bool) -> None:
    actual_generator = torch.Generator()
    inner = MixtureMatrixSampler([GaussianMatrixSampler(normalize=normalize)])
    sampler = MixtureMatrixSampler(
        [(KernelMatrixSampler(num_centers=3, gamma=1.0), 0), (inner, 1)]
    )
    actual = sampler.sample(
        size=(4, 7), dtype=torch.float32, generator=actual_generator
    )

    generator = torch.Generator()
    sampler.pick(generator=generator)
    expected = inner.pick(generator=generator).sample(
        size=(4, 7), dtype=torch.float32, generator=generator
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_generator.get_state(), generator.get_state())


def test_kernel_zero_gamma() -> None:
    actual = KernelMatrixSampler(num_centers=3, gamma=0.0, channels=7).sample(
        size=(4, 7), dtype=torch.float64
    )
    torch.testing.assert_close(
        actual.abs(), torch.full_like(actual, 7**-0.5), atol=1e-5, rtol=0
    )


def test_kernel_instances_share_correlation() -> None:
    def make_kernel() -> KernelMatrixSampler:
        return KernelMatrixSampler(
            num_centers=Correlated(LogUniformInt(1, 15), key="num_centers"),
            gamma=Correlated(LogUniform(0.1, 10.0), key="gamma"),
        )

    sizes = [(1, 3, 4), (1, 5, 4)]
    generator = torch.Generator()
    state = generator.get_state()
    outputs = []
    shared = make_kernel()
    for samplers in (
        [make_kernel(), make_kernel()],
        [shared, shared],
    ):
        generator.set_state(state)
        with CorrelatedContext(generator=generator):
            outputs.append(
                [
                    sampler.sample(
                        size=size, dtype=torch.float32, generator=generator
                    )
                    for sampler, size in zip(samplers, sizes)
                    for _ in range(2)
                ]
            )
    for actual, replay in zip(*outputs):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, replay, rtol=0, atol=0)
