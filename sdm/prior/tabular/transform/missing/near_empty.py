# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior import HP, Correlated, CorrelatedContext, LogUniform, Uniform
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._table import (
    feature_values,
    mask_cells,
)
from sdm.prior.tabular.transform.missing._count import _sample_count


class MaskNearEmpty(PriorTransform):
    """Mask column groups to sparse, overlapping sets of observed rows.

    Args:
        columns: Range for the fraction of feature columns to mask.
        observed: Sampled fraction of rows retained per column group.
            Groups in one dataset share a correlated rate. Defaults to
            ``LogUniform(0.01, 0.1)``.
        min_observed: Minimum retained rows per column group, capped by
            the row count.
    """

    def __init__(
        self,
        *,
        columns: tuple[float, float] = (0.5, 0.8),
        observed: HP[float] | None = None,
        min_observed: int = 4,
    ) -> None:
        self.columns = columns
        self.observed = Correlated(
            LogUniform(0.01, 0.1) if observed is None else observed
        )
        self._exponent = Uniform(0.9, 1.6)
        self.min_observed = min_observed

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = feature_values(x)
        num_rows, num_columns = values.shape
        if num_columns == 0:
            return x, y
        device = x.device
        column_count = _sample_count(
            total=num_columns,
            low=self.columns[0],
            high=self.columns[1],
            device=device,
            generator=generator,
        )
        columns = torch.randperm(
            num_columns, device=device, generator=generator
        )[:column_count]
        group_count = max(
            1,
            _sample_count(
                total=column_count,
                low=0.06,
                high=0.16,
                device=device,
                generator=generator,
            ),
        )
        exponent = self._exponent.sample(generator=generator)
        weights = torch.arange(
            start=1, end=group_count + 1, device=device, dtype=torch.float32
        ).pow(-exponent)
        weights = weights[
            torch.randperm(group_count, device=device, generator=generator)
        ]
        family = torch.multinomial(
            input=weights,
            num_samples=column_count,
            replacement=True,
            generator=generator,
        )
        # ``columns`` is already a random permutation, so pinning its first
        # entries gives every group at least one column.
        family[:group_count] = torch.arange(group_count, device=device)
        with CorrelatedContext(generator=generator):
            rate = self.observed.sample_batch(size=group_count)
        observed_count = (
            (rate * num_rows)
            .ceil()
            .long()
            .clamp_min(self.min_observed)
            .clamp_max(num_rows)
        )
        sizes = torch.bincount(family, minlength=group_count)
        load = torch.zeros(num_rows, device=device)
        for g in torch.argsort(sizes, descending=True).tolist():
            tie = torch.rand(num_rows, device=device, generator=generator)
            observed_rows = torch.argsort(load + tie)[: int(observed_count[g])]
            load[observed_rows] += float(sizes[g])
            observed = torch.zeros(num_rows, dtype=torch.bool, device=device)
            observed[observed_rows] = True
            family_columns = columns[family == g]
            swapped = torch.zeros_like(family_columns, dtype=torch.bool)
            for i, column in enumerate(family_columns.tolist()):
                column_values = values[:, column]
                seen = column_values[observed]
                distinct = seen[seen.isfinite()].unique()
                if distinct.numel() >= 2:
                    continue
                candidates = column_values.isfinite() & ~observed
                if distinct.numel() == 1:
                    candidates &= column_values != distinct[0]
                candidates = candidates.nonzero().flatten()
                if candidates.numel() == 0:
                    continue
                drop = observed_rows[
                    torch.randint(
                        high=observed_rows.numel(),
                        size=(),
                        device=device,
                        generator=generator,
                    )
                ]
                add = candidates[
                    torch.randint(
                        high=candidates.numel(),
                        size=(),
                        device=device,
                        generator=generator,
                    )
                ]
                column_observed = observed.clone()
                column_observed[drop] = False
                column_observed[add] = True
                swapped[i] = True
                values[~column_observed, column] = torch.nan
            rows = (~observed).nonzero().flatten()
            unswapped = family_columns[~swapped]
            values[rows[:, None], unswapped[None, :]] = torch.nan
        return mask_cells(x, values.isnan()), y
