# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class LinearDetrend(Processor, InvertibleMixin):
    """Remove a linear trend fitted to observed numerical values.

    The ``time_column`` contains numerical step coordinates stored as an ID
    column. Non-finite values are ignored when fitting and preserved during
    transformation. For predictions with multiple outputs per fitted column,
    the outputs for each column must be adjacent.

    Args:
        time_column: Column containing the step coordinate for each row.
        threshold: Detrend only when the residual standard deviation is less
            than this fraction of the original standard deviation. ``None``
            always applies the fitted trend.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        time_column: str,
        *,
        threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.time_column = time_column
        self.threshold = threshold
        self.register_buffer("origin", torch.empty(0, dtype=torch.long))
        self.register_buffer("slope", torch.empty(0))
        self.register_buffer("intercept", torch.empty(0))

    def _time(self, table: TableTensor) -> Tensor:
        return table[self.time_column].id[..., 0]

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        values = table.numerical.to(
            dtype=torch.promote_types(table.numerical.dtype, torch.float32)
        )
        time = self._time(table)
        self.origin = time[..., -1:].clone()
        time = (time - self.origin).to(dtype=values.dtype).unsqueeze(-1)
        valid = values.isfinite()
        count = valid.sum(dim=-2, keepdim=True, dtype=torch.int32)
        count.clamp_min_(1)
        sum_t = torch.where(valid, time, 0.0).sum(dim=-2, keepdim=True)
        sum_t2 = torch.where(valid, time.square(), 0.0).sum(
            dim=-2, keepdim=True
        )
        centered = values.masked_fill(~valid, 0.0)
        mean = centered.sum(dim=-2, keepdim=True) / count
        centered.sub_(mean).masked_fill_(~valid, 0.0)
        time_ss = (sum_t2 - sum_t.square() / count).clamp_min_(0.0)
        cross = time.transpose(-2, -1) @ centered

        constant_time = time_ss == 0
        self.slope = cross / torch.where(constant_time, 1.0, time_ss)
        self.slope.masked_fill_(constant_time, 0.0)
        self.intercept = mean - self.slope * (sum_t / count)

        if self.threshold is not None:
            value_ss = centered.square_().sum(dim=-2, keepdim=True)
            # OLS residual sum of squares: value_ss - slope * cross.
            residual_ss = (value_ss - self.slope * cross).clamp_min_(0.0)
            apply = residual_ss.sqrt() < self.threshold * value_ss.sqrt()
            self.slope.masked_fill_(~apply, 0.0)
            self.intercept.masked_fill_(~apply, 0.0)

    def _trend(self, table: TableTensor) -> Tensor:
        time = (
            (self._time(table) - self.origin)
            .to(dtype=self.slope.dtype)
            .unsqueeze(-1)
        )
        return self.slope * time + self.intercept

    def _transform(self, table: TableTensor) -> TableTensor:
        time = (self._time(table) - self.origin).to(dtype=self.slope.dtype)
        numerical = table.numerical - self.intercept
        numerical.addcmul_(self.slope, time.unsqueeze(-1), value=-1)
        return table.replace_blocks(
            numerical=numerical.to(table.numerical.dtype)
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        trend = self._trend(table)  # [..., F, T]
        values = table.numerical.unflatten(-1, (trend.size(-1), -1))
        numerical = (values + trend.unsqueeze(-1)).flatten(-2)
        return table.replace_blocks(
            numerical=numerical.to(table.numerical.dtype)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"time_column={self.time_column!r}, threshold={self.threshold})"
        )
