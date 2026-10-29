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

        NaNs and masked values are excluded.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mask: Boolean observed-value mask with shape ``[..., R, D]``.
            correction: Difference between the sample size and sample degrees
                of freedom.

        Returns:
            Mean and standard deviation with shape ``[..., R, 1]``.
        """
        x = x.detach()
        if mask is not None:
            x = x.masked_fill(~mask, torch.nan)

        mean = x.nanmean(dim=-1, keepdim=True)
        var = (x - mean).square().nanmean(dim=-1, keepdim=True)
        if correction != 0:
            count = (~x.isnan()).sum(dim=-1, keepdim=True)
            var = var * (count / (count - correction).clamp(min=1))
        std = var.sqrt()

        mask = mean.isnan()
        return mean.masked_fill(mask, 0.0), std.masked_fill(mask, 1.0)

    def forward(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """The forward pass.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, *]``.
            scale: The scale with shape ``[..., R, *]``.
        """
        out = (x - mean) / (scale + self.eps)
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def inverse(self, x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """Invert the forward pass.

        Args:
            x: Tensor with shape ``[..., R, D]``.
            mean: The mean with shape ``[..., R, *]``.
            scale: The scale with shape ``[..., R, *]``.
        """
        if self.weight is not None and self.bias is not None:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * (scale + self.eps) + mean
