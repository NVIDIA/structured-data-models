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


class Interpolate(SeriesProcessor, InvertibleMixin):
    """Interpolate missing numerical values along the row dimension.

    Args:
        fill_value: Value used for columns with no observations.
        fill_leading: Whether to fill missing values before the first
            observation. If ``False``, leading missing values remain NaN.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _last_values: EnsembleTensor[Tensor]
    _last_indices: EnsembleTensor[Tensor]

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
        fill_leading: bool = True,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.fill_leading = fill_leading
        self.register_buffer("_last_values", None)
        self.register_buffer("_last_indices", None)

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        x = table.numerical
        index = EnsembleData.execute(
            torch.arange, x.size(-2), device=x.device
        ).view(-1, 1)
        index = index.where(~x.isnan(), -1)
        last_index = index.amax(dim=-2, keepdim=True)
        self.register_buffer(
            "_last_values", x.gather(-2, last_index.clamp(min=0))
        )
        self.register_buffer("_last_indices", last_index)

    def _transform(self, table: TableLike) -> TableLike:
        if self._offset is None:
            raise RuntimeError(
                "Use this processor in a Recipe with a resolved time offset"
            )
        x = table.numerical
        last_value = self._last_values
        last_index = self._last_indices
        offset = self._offset

        # Prepend the context anchor only when processing future rows.
        mask = (last_index >= 0) & (offset > 0)
        x = EnsembleData.execute(
            torch.cat, [last_value.where(mask, torch.nan), x], dim=-2
        )

        # Find nearest observed row before and after each row.
        index = EnsembleData.execute(
            torch.arange, x.size(-2), device=x.device
        ).view(-1, 1)
        missing = x.isnan()
        left = index.where(~missing, -1).cummax(dim=-2).values
        right = index.where(~missing, x.size(-2)).flip(-2)
        right = right.cummin(dim=-2).values.flip(-2)

        leading_mask = None
        if not self.fill_leading:
            leading_mask = left[..., 1:, :] < 0  # Include leading NaNs.
            leading_mask &= left[..., -1:, :] >= 0  # Exclude NaN columns.

        # At edges, use the closest observation on the available side.
        left, right = (
            left.where(left >= 0, right).clamp(0, x.size(-2) - 1),
            right.where(right < x.size(-2), left).clamp(0, x.size(-2) - 1),
        )

        # Gather endpoint values used for interpolation:
        low = x.gather(-2, left)
        high = x.gather(-2, right)
        low = low.where(~low.isnan(), self.fill_value)
        high = high.where(~high.isnan(), self.fill_value)

        # Map indices to time coordinates:
        anchor_index = last_index - offset + 1
        left_time = left.where(left != 0, anchor_index)
        right_time = right.where(right != 0, anchor_index)

        # Final Interpolation:
        weight = (index - left_time).to(x.dtype)
        weight = weight / (right_time - left_time).clamp(min=1)
        numerical = low * (1 - weight) + high * weight
        numerical = numerical.where(low != high, low)  # Restore infinities
        numerical = numerical.where(missing, x)  # Restore finite values.
        numerical = numerical[..., 1:, :]  # Remove context anchor.
        numerical[numerical.isnan()] = self.fill_value
        if leading_mask is not None:
            numerical[leading_mask] = torch.nan

        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(
        self,
        table: TableLike,
    ) -> TableLike:
        return table

    def __repr__(self, *, indent: int = 0) -> str:
        args = []
        if self.fill_value != 0:
            args.append(f"fill_value={self.fill_value}")
        if not self.fill_leading:
            args.append(f"fill_leading={self.fill_leading}")
        return f"{' ' * indent}{self.__class__.__name__}({', '.join(args)})"
