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

    @staticmethod
    @torch.no_grad()
    def stats(
        x: Tensor,  # [..., R, D]
        mask: Tensor | None = None,  # [..., R, D]
        correction: int = 1,
    ) -> tuple[Tensor, Tensor]:  # [..., R, 1]
        """Compute mean and standard deviation over the final dimension.

        NaNs and masked values are excluded. Mean and standard deviation
        are computed and returned in
        :external+torch:ref:`torch.float64 <dtype-doc>`.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mask: Boolean mask with shape ``[..., R, D]``. Values marked
                ``True`` are excluded.
            correction: Difference between the sample size and sample degrees
                of freedom.

        Returns:
            Mean and standard deviation with shape ``[..., R, 1]``.
        """
        if mask is not None:
            x = x.masked_fill(mask, torch.nan)

        x = x.double()
        mean = x.nanmean(dim=-1, keepdim=True)
        var = (x - mean).square().nanmean(dim=-1, keepdim=True)
        if correction != 0:
            count = (~x.isnan()).sum(dim=-1, keepdim=True)
            var = var * (count.double() / (count - correction).clamp(min=1))
        std = var.sqrt()

        mask = mean.isnan()
        return mean.masked_fill(mask, 0.0), std.masked_fill(mask, 1.0)

    @staticmethod
    @torch.no_grad()
    def running_stats(
        x: Tensor,  # [..., R, D]
        mask: Tensor | None = None,  # [..., R, D]
        state: tuple[Tensor, Tensor, Tensor] | None = None,
        correction: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:  # [..., R, 1]
        """Compute cumulative statistics over patched sequences.

        NaNs and masked values are excluded. Mean and standard deviation
        are computed and returned in
        :external+torch:ref:`torch.float64 <dtype-doc>`.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mask: Boolean mask with shape ``[..., R, D]``. Values marked
                ``True`` are excluded.
            state: Previous mean, standard deviation, and count for streaming
                statistics.
            correction: Degrees of freedom subtracted from the observed count.

        Returns:
            Cumulative mean, standard deviation, and observed count for the
            supplied sequence, each with shape ``[..., R, 1]``.
        """
        D = x.size(-1)
        mask = x.isnan() if mask is None else x.isnan() | mask
        mask = mask.flatten(-2)  # [..., R * D]
        x = x.flatten(-2).double().masked_fill(mask, 0)  # [..., R * D]
        count = (~mask).cumsum(dim=-1)
        mean = x.cumsum(dim=-1) / count.clamp(min=1)

        # Welford increments accumulate squared deviations over time.
        prev = torch.nn.functional.pad(mean[..., :-1], (1, 0))
        inc = ((x - prev) * (x - mean)).masked_fill(mask, 0)
        m2 = inc.cumsum(dim=-1)

        # Keep statistics at the final value of each patch.
        count = count[..., D - 1 :: D].unsqueeze(-1)
        mean = mean[..., D - 1 :: D].unsqueeze(-1)
        m2 = m2[..., D - 1 :: D].unsqueeze(-1)

        if state is not None:
            state_mean, state_std, state_count = state
            state_mean = state_mean[..., -1:, :]
            state_std = state_std[..., -1:, :]
            state_count = state_count[..., -1:, :]

            total_count = state_count + count
            weight = count.double() / total_count.clamp(min=1)
            delta = mean - state_mean
            m2 += state_std.square() * (state_count - correction).clamp(min=1)
            m2 += delta.square() * state_count * weight
            mean = state_mean + delta * weight
            count = total_count

        std = (m2 / (count - correction).clamp(min=1)).clamp(min=0).sqrt()
        return mean, std, count

    def forward(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """The forward pass.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, 1]``.
            scale: The scale with shape ``[..., R, 1]``.
        """
        mean, scale = mean.to(x.dtype), scale.to(x.dtype)
        out = (x - mean) / (scale + self.eps)
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def inverse(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """Invert the forward pass.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, 1]``.
            scale: The scale with shape ``[..., R, 1]``.
        """
        mean, scale = mean.to(x.dtype), scale.to(x.dtype)
        if self.weight is not None and self.bias is not None:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * (scale + self.eps) + mean
