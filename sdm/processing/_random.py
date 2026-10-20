# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor


class SeedMixin:
    r"""Store a seed at fit, from which transform draws its row values."""

    _seed = 0

    def get_extra_state(self) -> object:
        r""":meta private:"""  # noqa: D415
        return self._seed

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._seed = cast(int, state)

    def _draw_seed(self, generator: torch.Generator | None) -> None:
        device = None if generator is None else generator.device
        self._seed = int(
            torch.randint(2**62, (), device=device, generator=generator)
        )

    def _transform_generator(self) -> torch.Generator:
        # A CPU generator makes the output independent of the device.
        return torch.Generator().manual_seed(self._seed)


def uniform(
    low: float,
    high: float,
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.empty(size, device=device, dtype=dtype).uniform_(
        low, high, generator=generator
    )


def randint(
    low: Tensor | int,
    high: Tensor | int,
    size: Sequence[int],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Draw integers in ``[low, high]`` with broadcast bounds."""
    u = torch.rand(size, device=device, generator=generator)
    return (low + u * (high - low + 1)).floor().long()


def count(
    total: Tensor,
    low: float,
    high: float,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Draw a count between the fractions ``low`` and ``high`` of ``total``.

    The count is at least 1 and at most ``total``, if ``total`` is positive.
    """
    total = total.double()
    lower = (total * low).ceil().clamp(min=1)
    upper = torch.maximum(lower, torch.minimum(total, (total * high).floor()))
    u = torch.rand(total.shape, device=total.device, generator=generator)
    return (
        (lower + u * (upper - lower + 1)).floor().long().minimum(total.long())
    )


def log_uniform(
    low: float,
    high: float,
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Draw values whose logarithms are uniformly distributed."""
    return uniform(
        math.log(low),
        math.log(high),
        size,
        device=device,
        dtype=dtype,
        generator=generator,
    ).exp()


def regime(
    size: Sequence[int],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Draw values in ``[0, 1]`` that share a random regime.

    The values along the last dimension follow one
    :math:`\mathrm{Beta}(st, s(1 - t))` with :math:`t \sim U(0, 1)` and
    :math:`s \sim \mathrm{LogUniform}(0.1, 10^4)`. A large :math:`s` puts the
    values close together, and a small :math:`s` pushes them to 0 or 1.
    """
    size = tuple(size)
    t = torch.rand(
        size[:-1], device=device, dtype=torch.float64, generator=generator
    ).clamp(1e-6, 1 - 1e-6)
    s = log_uniform(
        1e-1,
        1e4,
        size[:-1],
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    # Float64 keeps small concentrations from underflowing to 0 / 0.
    # PyTorch has no public generator-aware gamma sampling primitive.
    a = torch._standard_gamma(
        (s * t).unsqueeze(-1).expand(size), generator=generator
    )
    b = torch._standard_gamma(
        (s * (1 - t)).unsqueeze(-1).expand(size), generator=generator
    )
    return (a / (a + b)).float()


def rank(x: Tensor, dim: int) -> Tensor:
    r"""Return the rank of each entry along ``dim``, with stable ties."""
    order = x.argsort(dim=dim, stable=True)
    position = torch.arange(x.size(dim), device=x.device)
    position = position.view(-1, *([1] * (x.dim() - 1 - dim % x.dim())))
    return torch.empty_like(order).scatter_(
        dim, order, position.expand_as(order)
    )


def subset(
    eligible: Tensor,
    count: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Select ``count`` random eligible entries along the last dimension.

    Args:
        eligible: Boolean mask of shape ``[..., N]``.
        count: Number of entries to select, of shape ``[...]``.
        generator: Pseudorandom number generator used for sampling.
    """
    score = torch.rand(
        eligible.shape, device=eligible.device, generator=generator
    )
    position = rank(score.masked_fill(~eligible, 2.0), dim=-1)
    return (position < count.unsqueeze(-1)) & eligible
