# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, SeriesEnsembleProcessor
from sdm.processing.numerical._stats import _isfinite


class Detrend(SeriesEnsembleProcessor, EnsembleInvertibleMixin):
    """Remove a per-column linear trend over step coordinates.

    Args:
        threshold: If set, detrend a column only when the standard deviation
            after detrending is less than ``threshold`` times the standard
            deviation before detrending.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self._slopes: BufferList[Tensor] = BufferList()
        self._intercepts: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        slopes = []
        intercepts = []
        for group in ensemble_table._iter_groups():
            x = group.numerical
            time = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
            time = time.view(-1, 1)  # [R, 1]

            valid = _isfinite(x)
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
                keep &= valid.all(dim=-2, keepdim=True)
                slope = slope.where(keep, 0.0)
                mean = mean.where(keep, 0.0)

            slopes.append(slope)
            intercepts.append(mean - slope * time_mean)

        self._slopes = BufferList(slopes)
        self._intercepts = BufferList(intercepts)

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
            time = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
            time = time.view(-1, 1) + self._offset[first_member_id]
            trend = self._intercepts[group_id] + self._slopes[group_id] * time

            groups.append(group.replace_blocks(numerical=x - trend))

        return ensemble_table.replace_groups(groups)

    def _inverse_transform_ensemble(
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
            time = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
            time = time.view(-1, 1) + self._offset[first_member_id]
            trend = self._intercepts[group_id] + self._slopes[group_id] * time

            x = x.unflatten(-1, (self._slopes[group_id].size(-1), -1))
            x = (x + trend.unsqueeze(-1)).flatten(-2)

            groups.append(group.replace_blocks(numerical=x))

        return ensemble_table.replace_groups(groups)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )
