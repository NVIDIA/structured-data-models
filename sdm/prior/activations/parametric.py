# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Correlated, LogUniform, UniformInt
from sdm.prior.activations import Activation


class ReLUPowActivation(Activation):
    """Apply ``relu(x) ** exponent`` elementwise.

    Args:
        exponent: Exponent sampled on every call. Defaults to
            ``Correlated(LogUniform(0.1, 10.0))``.
    """

    def __init__(
        self, exponent: HP[float] | Correlated[float] | None = None
    ) -> None:
        self.exponent = exponent or Correlated(LogUniform(0.1, 10.0))

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return x.relu().pow(self.exponent.sample(generator=generator))


class SignPowActivation(Activation):
    """Apply ``sign(x) * abs(x) ** exponent`` elementwise.

    Args:
        exponent: Exponent sampled on every call. Defaults to
            ``Correlated(LogUniform(0.1, 10.0))``.
    """

    def __init__(
        self, exponent: HP[float] | Correlated[float] | None = None
    ) -> None:
        self.exponent = exponent or Correlated(LogUniform(0.1, 10.0))

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return x.sign() * x.abs().pow(
            self.exponent.sample(generator=generator)
        )


class InversePowActivation(Activation):
    """Apply ``(abs(x) + 1e-3) ** -exponent`` elementwise.

    Args:
        exponent: Exponent sampled on every call. Defaults to
            ``Correlated(LogUniform(0.1, 10.0))``.
    """

    def __init__(
        self, exponent: HP[float] | Correlated[float] | None = None
    ) -> None:
        self.exponent = exponent or Correlated(LogUniform(0.1, 10.0))

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return (x.abs() + 1e-3).pow(-self.exponent.sample(generator=generator))


class IntegerPowActivation(Activation):
    """Apply ``x ** exponent`` elementwise for an integer exponent.

    Args:
        exponent: Exponent sampled on every call. Defaults to
            ``Correlated(UniformInt(2, 5))``.
    """

    def __init__(
        self, exponent: HP[int] | Correlated[int] | None = None
    ) -> None:
        self.exponent = exponent or Correlated(UniformInt(2, 5))

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return x.pow(self.exponent.sample(generator=generator))
