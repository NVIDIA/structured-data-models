# Copyright (c) 2024 Auton Lab, Carnegie Mellon University
# Licensed under the MIT License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reversible time-series normalization."""

from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.nn import Parameter


class RevINState(NamedTuple):
    """Statistics needed to invert :class:`RevIN`.

    Args:
        mean: Per-series observed mean with shape ``[B, V, 1]``.
        stdev: Per-series observed standard deviation with shape
            ``[B, V, 1]``.
    """

    mean: Tensor
    stdev: Tensor


class RevIN(torch.nn.Module):
    r"""Reversible instance normalization from the `"RevIN"`_ paper.

    .. _"RevIN": https://openreview.net/forum?id=cGDAkQo1C0p

    Args:
        num_features: Number of affine feature parameters. Use ``1`` to share
            affine parameters across variates.
        eps: Value added to standard deviations before division.
        affine: Whether to learn an affine transformation.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = Parameter(
                torch.ones(1, num_features, 1, **factory_kwargs)
            )
            self.affine_bias = Parameter(
                torch.zeros(1, num_features, 1, **factory_kwargs)
            )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, RevINState]:
        """Normalize a time series and return its inverse state.

        Args:
            x: Time-series tensor with shape ``[B, V, T]``.
            mask: Observed-time mask with shape ``[B, T]``. ``None`` marks
                every time step as observed.

        Returns:
            The normalized tensor with shape ``[B, V, T]`` and its detached
            normalization statistics.
        """
        if mask is None:
            mask = torch.ones(
                x.size(0),
                x.size(-1),
                dtype=torch.bool,
                device=x.device,
            )

        observed = mask.bool().unsqueeze(-2)
        masked = x.masked_fill(~observed, float("nan"))
        mean = masked.nanmean(dim=-1, keepdim=True).detach()
        variance = (masked - mean).square().nanmean(dim=-1, keepdim=True)
        stdev = variance.sqrt().detach() + self.eps

        out = (x - mean) / stdev
        if self.affine:
            out = out * self.affine_weight + self.affine_bias
        return out, RevINState(mean=mean, stdev=stdev)

    def inverse(self, x: Tensor, state: RevINState) -> Tensor:
        """Invert a normalization operation.

        Args:
            x: Normalized tensor with shape ``[B, V, T]``.
            state: Statistics returned by :meth:`forward`.

        Returns:
            Tensor restored to its original scale with shape ``[B, V, T]``.
        """
        if self.affine:
            x = (x - self.affine_bias) / (
                self.affine_weight + self.eps * self.eps
            )
        return x * state.stdev + state.mean
