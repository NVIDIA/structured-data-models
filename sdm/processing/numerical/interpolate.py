# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import Tensor

from sdm import EnsembleTable, Stype
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, SeriesEnsembleProcessor


class Interpolate(SeriesEnsembleProcessor, EnsembleInvertibleMixin):
    """Interpolate missing feature values along the row dimension.

    Args:
        fill_value: Value used for columns with no observations.
        fill_leading: Whether to fill missing values before the first
            observation. If ``False``, leading missing values remain NaN.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
        fill_leading: bool = True,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.fill_leading = fill_leading
        self._last_values: BufferList[Tensor] = BufferList()
        self._last_indices: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        last_values = []
        last_indices = []
        for group in ensemble_table._iter_groups():
            x = group.numerical
            index = torch.arange(x.size(-2), device=x.device).view(-1, 1)
            index = index.where(~x.isnan(), -1)
            last_index = index.amax(dim=-2, keepdim=True)
            last_value = x.gather(-2, last_index.clamp(min=0))
            last_values.append(last_value)
            last_indices.append(last_index)

        self._last_values = BufferList(last_values)
        self._last_indices = BufferList(last_indices)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._offset is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved offset; "
                "use it in a 'Recipe' through model execution"
            )

        groups = []
        for group_id, group in enumerate(ensemble_table._iter_groups()):
            first_member_id = next(
                i
                for i, (g, _) in enumerate(ensemble_table._locations)
                if g == group_id
            )

            x = group.numerical
            last_value = self._last_values[group_id]
            last_index = self._last_indices[group_id]
            offset = self._offset[first_member_id]

            # Prepend the context anchor only when processing future rows.
            mask = (last_index >= 0) & (offset > 0)
            x = torch.cat([last_value.where(mask, torch.nan), x], dim=-2)

            # Find nearest observed row before and after each row.
            index = torch.arange(x.size(-2), device=x.device).view(-1, 1)
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

            groups.append(group.replace_blocks(numerical=numerical))

        return ensemble_table.replace_groups(groups)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table

    def __repr__(self, *, indent: int = 0) -> str:
        args = []
        if self.fill_value != 0:
            args.append(f"fill_value={self.fill_value}")
        if not self.fill_leading:
            args.append(f"fill_leading={self.fill_leading}")
        return f"{' ' * indent}{self.__class__.__name__}({', '.join(args)})"
