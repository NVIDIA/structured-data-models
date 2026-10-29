# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor


class RevIN(torch.nn.Module):
    r"""Reversible instance normalization from the
    `"Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift" <https://openreview.net/forum?id=cGDAkQo1C0p>`_
    paper.

    Args:
        affine_channels: The number of channels for learned affine parameters.
            If ``None``, no affine transformation is applied.
        eps: The value added to the supplied scale for numerical stability.
        device: The device.
        dtype: The dtype.
    """  # noqa: D205

    def __init__(
        self,
        affine_channels: int | None = None,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.eps = eps

        self.weight = self.bias = None
        if affine_channels is not None:
            self.weight = torch.nn.Parameter(
                torch.ones(affine_channels, 1, **factory_kwargs)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(affine_channels, 1, **factory_kwargs)
            )

    def stats(
        self,
        x: Tensor,  # [..., R, D]
        mask: Tensor | None = None,  # [..., R, D]
        correction: int = 1,
    ) -> tuple[Tensor, Tensor]:  # [..., R, 1]
        """Compute mean and standard deviation over the final dimension.

        NaNs and masked values are excluded. Mean and standard deviation
        are computed and returned in float64.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mask: Boolean mask with shape ``[..., R, D]``. Values marked
                ``True`` are excluded.
            correction: Difference between the sample size and sample degrees
                of freedom.

        Returns:
            Mean and standard deviation with shape ``[..., R, 1]`` and
            dtype ``torch.float64``.
        """
        x = x.detach().double()
        if mask is not None:
            x = x.masked_fill(mask, torch.nan)

        mean = x.nanmean(dim=-1, keepdim=True)
        var = (x - mean).square().nanmean(dim=-1, keepdim=True)
        if correction != 0:
            count = (~x.isnan()).sum(dim=-1, keepdim=True)
            var = var * (count.double() / (count - correction).clamp(min=1))
        std = var.sqrt()

        mask = mean.isnan()
        return mean.masked_fill(mask, 0.0), std.masked_fill(mask, 1.0)

    @torch.no_grad()
    def running_stats(
        self,
        x: Tensor,  # [..., R, D]
        mask: Tensor | None = None,  # [..., R, D]
        correction: int = 1,
        state: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:  # [..., R, 1]
        """Compute cumulative statistics through each patch.

        Each patch uses all observed values through its final value. NaNs
        and masked values are excluded. Means and standard deviations are
        computed and returned in float64. Prefixes without observations
        return zero mean and standard deviation.

        Args:
            x: Tensor with shape ``[..., R, D]``, where ``R`` is the number
                of patches and ``D`` is the number of values per patch.
            mask: Boolean mask with the same shape as ``x``. Values marked
                ``True`` are excluded.
            correction: Degrees of freedom subtracted from the observed count.
                The variance divisor is clamped to at least one.
            state: Previous mean, standard deviation, and count, each with
                shape ``[..., 1, 1]``, to continue accumulation. Use the
                final patch returned by a previous call with the same
                ``correction``. If ``None``, start without observations.

        Returns:
            Cumulative mean, standard deviation, and observed count, each
            with shape ``[..., R, 1]``. Counts have dtype ``torch.int64``;
            means and standard deviations have dtype ``torch.float64``.
        """
        observed = ~x.isnan()
        if mask is not None:
            observed = observed & ~mask
        observed = observed.flatten(-2)
        values = x.flatten(-2).double().masked_fill(~observed, 0)
        count = observed.cumsum(dim=-1)
        mean = values.cumsum(dim=-1) / count.clamp(min=1)

        # Welford increments accumulate squared deviations over time.
        previous = torch.nn.functional.pad(mean[..., :-1], (1, 0))
        increments = ((values - previous) * (values - mean)).masked_fill(
            ~observed,
            0,
        )
        m2 = increments.cumsum(dim=-1)

        # Keep statistics at the final value of each patch.
        end = slice(x.size(-1) - 1, None, x.size(-1))
        count = count[..., end].unsqueeze(-1)
        mean = mean[..., end].unsqueeze(-1)
        m2 = m2[..., end].unsqueeze(-1)
        if state is not None:
            state_mean, state_std, state_count = state
            total_count = state_count + count
            weight = count.double() / total_count.clamp(min=1)
            delta = mean - state_mean
            m2 = (
                m2
                + state_std.square() * (state_count - correction).clamp(min=1)
                + delta.square() * state_count * weight
            )
            mean = state_mean + delta * weight
            count = total_count
        std = (m2 / (count - correction).clamp(min=1)).clamp(min=0).sqrt()
        return mean, std, count

    def forward(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """The forward pass.

        Mean and scale are converted to the input dtype.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, *]``.
            scale: The scale with shape ``[..., R, *]``.
        """
        mean, scale = mean.to(x.dtype), scale.to(x.dtype)
        out = (x - mean) / (scale + self.eps)
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def inverse(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """Invert the forward pass.

        Mean and scale are converted to the input dtype.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, *]``.
            scale: The scale with shape ``[..., R, *]``.
        """
        mean, scale = mean.to(x.dtype), scale.to(x.dtype)
        if self.weight is not None and self.bias is not None:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * (scale + self.eps) + mean
