# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import (
    EnsembleData,
    EnsembleTensor,
    Stype,
    TableLike,
)
from sdm.processing import InvertibleMixin, SeriesProcessor
from sdm.processing.numerical._stats import _isfinite


class Detrend(SeriesProcessor, InvertibleMixin):
    """Remove a per-column linear trend over step coordinates.

    Args:
        threshold: If set, detrend a column only when the standard deviation
            after detrending is less than ``threshold`` times the standard
            deviation before detrending.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _slopes: EnsembleTensor[Tensor]
    _intercepts: EnsembleTensor[Tensor]

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.register_buffer("_slopes", None)
        self.register_buffer("_intercepts", None)

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        x = table.numerical
        time = EnsembleData.execute(
            torch.arange, x.size(-2), device=x.device, dtype=x.dtype
        )
        time = time.view(-1, 1)  # [R, 1]

        valid = EnsembleData.execute(_isfinite, x, batched=True)
        count = valid.sum(dim=-2, keepdim=True, dtype=x.dtype).clamp(min=1)

        mean = x.where(valid, 0.0).sum(dim=-2, keepdim=True) / count
        time_mean = time.where(valid, 0.0).sum(-2, keepdim=True) / count

        centered = (x - mean).where(valid, 0.0)
        time_centered = time - time_mean

        time_var = time_centered.square().where(valid, 0.0)
        time_var = time_var.sum(dim=-2, keepdim=True) / count
        cov = (time_centered * centered).sum(dim=-2, keepdim=True) / count

        slope = (cov / time_var).where(time_var > 0, 0.0)
        if self.threshold is not None:
            var = centered.square().sum(dim=-2, keepdim=True) / count
            keep = var - slope * cov < self.threshold**2 * var
            leading = x.isnan().int().cumprod(dim=-2).bool()
            keep &= (valid | leading).all(dim=-2, keepdim=True)
            slope = slope.where(keep, 0.0)
            mean = mean.where(keep, 0.0)

        self.register_buffer("_slopes", slope)
        self.register_buffer("_intercepts", mean - slope * time_mean)

    def _transform(self, table: TableLike) -> TableLike:
        if self._offset is None:
            raise RuntimeError(
                "Use this processor in a Recipe with a resolved time offset"
            )
        time = EnsembleData.execute(
            torch.arange,
            table.numerical.size(-2),
            device=table.device,
            dtype=table.numerical.dtype,
        ).view(-1, 1)
        trend = self._intercepts + self._slopes * (time + self._offset)
        return table.replace_blocks(numerical=table.numerical - trend)

    def _inverse_transform(self, table: TableLike) -> TableLike:
        if self._offset is None:
            raise RuntimeError(
                "Use this processor in a Recipe with a resolved time offset"
            )
        time = EnsembleData.execute(
            torch.arange,
            table.numerical.size(-2),
            device=table.device,
            dtype=table.numerical.dtype,
        ).view(-1, 1)
        trend = self._intercepts + self._slopes * (time + self._offset)
        numerical = table.numerical.unflatten(-1, (self._slopes.size(-1), -1))
        return table.replace_blocks(
            numerical=(numerical + trend.unsqueeze(-1)).flatten(-2)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )
