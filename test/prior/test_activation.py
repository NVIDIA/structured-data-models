# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import sdm.prior.activations as activations
from sdm.prior import (
    HP,
    Constant,
    Correlated,
    CorrelatedContext,
    LogUniform,
    Uniform,
)
from sdm.prior.activations import (
    CallableActivation,
    MixtureActivation,
    ReLUPowActivation,
    StandardizedActivation,
)


@pytest.mark.parametrize("scale", [None, 2.0, Uniform(1.0, 3.0)])
def test_standardized_activation(scale: HP[float] | float | None) -> None:
    x = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    activation = StandardizedActivation(
        CallableActivation(Tensor.tanh), scale=scale
    )
    actual = activation.forward(x, generator=torch.Generator())

    generator = torch.Generator()
    std, mean = torch.std_mean(x, dim=0, keepdim=True, correction=0)
    expected = (x - mean) / (std + 1e-4)
    scale = HP.from_value(
        Correlated(LogUniform(1.0, 10.0)) if scale is None else scale
    ).sample(generator=generator)
    rows = torch.randint(high=32, size=(1, 4), generator=generator)
    center = expected.gather(dim=0, index=rows)
    expected = (scale * (expected - center)).tanh()
    std, mean = torch.std_mean(expected, dim=0, keepdim=True, correction=0)
    expected = (expected - mean) / (std + 1e-4)

    torch.testing.assert_close(actual, expected)


def test_callable_activation() -> None:
    x = torch.randn(16, 4)
    actual = CallableActivation(F.elu).forward(x, generator=torch.Generator())
    torch.testing.assert_close(actual, F.elu(x))


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (activations.identity, lambda x: x),
        (activations.heaviside, lambda x: (x > 0).to(x.dtype)),
        (activations.gaussian, lambda x: (-x.square()).exp()),
        (
            activations.unit_indicator,
            lambda x: ((x >= 0) & (x <= 1)).to(x.dtype),
        ),
        (activations.softmax, lambda x: x.softmax(-1)),
        (
            activations.one_hot_argmax,
            lambda x: F.one_hot(x.argmax(-1), num_classes=x.size(-1)).to(
                x.dtype
            ),
        ),
        (activations.argsort, lambda x: x.argsort(-1).to(x.dtype)),
        (activations.log_abs, lambda x: x.abs().clamp(min=1e-6).log()),
        (activations.rank, lambda x: x.argsort(-1).argsort(-1).to(x.dtype)),
        (activations.modulo, lambda x: x % 1),
    ],
)
def test_fixed_functions(
    function: Callable[[Tensor], Tensor],
    expected: Callable[[Tensor], Tensor],
) -> None:
    x = torch.tensor([[-2.0, -0.1, 0, 0.7], [1.0, 1.5, 2.0, 3.0]])
    torch.testing.assert_close(function(x), expected(x))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ReLUPow", lambda x: x.relu().square()),
        ("SignPow", lambda x: x.sign() * x.square()),
        ("InversePow", lambda x: (x.abs() + 1e-3).pow(-2)),
        ("IntegerPow", lambda x: x.square()),
    ],
)
def test_parametric_activations(
    name: str, expected: Callable[[Tensor], Tensor]
) -> None:
    x = torch.randn(16, 4)
    actual = getattr(activations, name + "Activation")(
        exponent=Constant(2)
    ).forward(x)
    torch.testing.assert_close(actual, expected(x))


def _forward(activation: activations.Activation, x: Tensor) -> Tensor:
    generator = torch.Generator()
    with CorrelatedContext(generator=generator):
        return activation.forward(x, generator=generator)


def test_mixture_activation_replays_after_pickle() -> None:
    x = torch.randn(32, 4)
    activation = MixtureActivation(
        activations=[
            (CallableActivation(Tensor.tanh), 2.0),
            CallableActivation(activations.gaussian),
            ReLUPowActivation(exponent=Correlated(LogUniform(0.1, 10.0))),
        ],
    )
    expected = _forward(activation, x)
    torch.testing.assert_close(_forward(activation, x), expected)
    restored = pickle.loads(pickle.dumps(activation))
    torch.testing.assert_close(_forward(restored, x), expected)
