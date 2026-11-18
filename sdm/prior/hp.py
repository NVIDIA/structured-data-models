# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import torch
from torch import Tensor

T = TypeVar("T")


def _check_bounds(low: float, high: float) -> None:
    if math.isnan(low) or math.isnan(high) or low > high:
        raise ValueError(f"Expected low <= high, got low={low}, high={high}.")


class HP(ABC, Generic[T]):
    """Base class for sampling hyperparameters."""

    def sample(self, generator: torch.Generator | None = None) -> T:
        """Sample one Python value.

        Args:
            generator: Pseudorandom generator.
        """
        return cast(T, self.sample_batch(size=1, generator=generator).item())

    @abstractmethod
    def sample_batch(
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a tensor.

        Args:
            size: Sample shape or vector length.
            generator: Pseudorandom generator.
        """
        ...

    @staticmethod
    def from_value(x: "HP[T] | T") -> "HP[T]":
        """Return a sampler unchanged or wrap a value in :class:`Constant`.

        Args:
            x: Hyperparameter or fixed value.
        """
        return x if isinstance(x, HP) else Constant(x)


@dataclass(frozen=True)
class Constant(HP[T]):
    """Return a fixed hyperparameter.

    Args:
        value: Fixed value. Tensor sampling requires a numerical scalar.
    """

    value: T

    def sample(  # noqa: D102
        self, generator: torch.Generator | None = None
    ) -> T:
        return self.value

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return torch.full(  # type: ignore
            size=(size,) if isinstance(size, int) else size,
            fill_value=self.value,
            device="cpu",
        )


@dataclass(frozen=True)
class Uniform(HP[float]):
    """Sample real values uniformly between two bounds.

    Args:
        low: Lower bound.
        high: Upper bound.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        _check_bounds(self.low, self.high)

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = torch.rand(size, device="cpu", generator=generator)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        """Map quantile levels to values.

        Args:
            x: Quantile levels in ``[0, 1]``, with shape ``[...]``.
        """
        return self.low + (self.high - self.low) * x


@dataclass(frozen=True)
class UniformInt(HP[int]):
    """Sample integers uniformly between inclusive bounds.

    Args:
        low: Inclusive lower bound.
        high: Inclusive upper bound.
    """

    low: int
    high: int

    def __post_init__(self) -> None:
        _check_bounds(self.low, self.high)

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = torch.rand(size, device="cpu", generator=generator)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        """Map quantile levels to values.

        Args:
            x: Quantile levels in ``[0, 1]``, with shape ``[...]``.
        """
        x = (self.low + (self.high + 1 - self.low) * x).floor().long()
        return x.clamp(min=self.low, max=self.high)


@dataclass(frozen=True)
class LogUniform(HP[float]):
    """Sample positive real values uniformly in logarithmic space.

    Args:
        low: Positive lower bound.
        high: Positive upper bound.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        _check_bounds(self.low, self.high)

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = torch.rand(size, device="cpu", generator=generator)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        """Map quantile levels to values.

        Args:
            x: Quantile levels in ``[0, 1]``, with shape ``[...]``.
        """
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return (log_low + (log_high - log_low) * x).exp()


@dataclass(frozen=True)
class LogUniformInt(HP[int]):
    """Sample integers by truncating log-uniform values.

    Args:
        low: Positive inclusive lower bound.
        high: Inclusive upper bound.
    """

    low: int
    high: int

    def __post_init__(self) -> None:
        _check_bounds(self.low, self.high)

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = torch.rand(size, device="cpu", generator=generator)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        """Map quantile levels to values.

        Args:
            x: Quantile levels in ``[0, 1]``, with shape ``[...]``.
        """
        log_low = math.log(self.low)
        log_high = math.log(self.high + 1)
        x = (log_low + (log_high - log_low) * x).exp().long()
        return x.clamp(min=self.low, max=self.high)


@dataclass(frozen=True, init=False)
class Choice(HP[T]):
    """Choose uniformly among fixed options.

    Args:
        *options: Values to choose from. Tensor sampling requires numerical
            options.
    """

    options: tuple[T, ...]

    def __init__(self, *options: T) -> None:
        assert len(options) > 0
        object.__setattr__(self, "options", options)

    def sample(  # noqa: D102
        self, generator: torch.Generator | None = None
    ) -> T:
        index = torch.randint(
            low=0,
            high=len(self.options),
            size=(),
            device="cpu",
            generator=generator,
        )
        return self.options[int(index)]

    def sample_batch(  # noqa: D102
        self,
        size: int | torch.Size | tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> Tensor:
        index = torch.randint(
            low=0,
            high=len(self.options),
            size=(size,) if isinstance(size, int) else size,
            device="cpu",
            generator=generator,
        )
        return torch.tensor(self.options, device="cpu")[index]
