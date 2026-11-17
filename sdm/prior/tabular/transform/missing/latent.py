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


class MaskLatent(PriorTransform):
    """Mask feature column blocks under an unobserved row factor.

    Args:
        num_proxies: Inclusive range for the number of proxy columns.
        columns: Range for the fraction of non-proxy columns to mask.
        prevalence: Fraction of rows with the latent factor.
            Defaults to ``Uniform(0.1, 0.5)``.
        rate_off: Sampled missing rate per column block when the latent
            factor is inactive. The blocks share a correlated rate.
            Defaults to ``Uniform(0.0, 0.1)``.
        rate_on: Sampled missing rate per column block when the latent
            factor is active. The blocks share a correlated rate.
            Defaults to ``Uniform(0.3, 0.9)``.
    """

    def __init__(
        self,
        *,
        num_proxies: tuple[int, int] = (1, 3),
        columns: tuple[float, float] = (0.2, 0.6),
        prevalence: HP[float] | float | None = None,
        rate_off: HP[float] | None = None,
        rate_on: HP[float] | None = None,
    ) -> None:
        self.num_proxies = num_proxies
        self.columns = columns
        self.prevalence = HP.from_value(
            Uniform(0.1, 0.5) if prevalence is None else prevalence
        )
        self.rate_off = Correlated(
            Uniform(0.0, 0.1) if rate_off is None else rate_off
        )
        self.rate_on = Correlated(
            Uniform(0.3, 0.9) if rate_on is None else rate_on
        )
        self._noise_scale = LogUniform(0.25, 1.0)

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = feature_values(x)
        num_rows, num_columns = values.shape
        device = x.device
        if num_columns >= max(2, self.num_proxies[0] + 1):
            # The latent factor thresholds a noisy random projection of
            # well-observed proxy columns. It drives the missing rate of a
            # fraction of the remaining columns.
            proxy_count = int(
                torch.randint(
                    low=self.num_proxies[0],
                    high=min(self.num_proxies[1], num_columns - 1) + 1,
                    size=(),
                    device=device,
                    generator=generator,
                )
            )
            observed = values.isfinite().float().mean(dim=0) >= 0.5
            candidates = observed.nonzero().flatten()
            if candidates.numel() < proxy_count:
                candidates = torch.arange(num_columns, device=device)
            proxies = candidates[
                torch.randperm(
                    candidates.numel(), device=device, generator=generator
                )[:proxy_count]
            ]
            is_proxy = torch.zeros(
                num_columns, dtype=torch.bool, device=device
            )
            is_proxy[proxies] = True
            remaining = (~is_proxy).nonzero().flatten()
            column_count = _sample_count(
                total=remaining.numel(),
                low=self.columns[0],
                high=self.columns[1],
                device=device,
                generator=generator,
            )
            columns = remaining[
                torch.randperm(
                    remaining.numel(), device=device, generator=generator
                )[:column_count]
            ]

            weight = torch.randn(
                proxy_count,
                device=device,
                dtype=values.dtype,
                generator=generator,
            )
            score = values[:, proxies].nan_to_num(0.0) @ weight
            score = (score - score.mean()) / score.std().clamp_min(1e-6)
            noise_scale = self._noise_scale.sample(generator=generator)
            score = score + noise_scale * torch.randn(
                num_rows,
                device=device,
                dtype=values.dtype,
                generator=generator,
            )
            prevalence = self.prevalence.sample(generator=generator)
            factor = score > score.quantile(1.0 - prevalence)

            block_count = int(
                torch.randint(
                    low=1,
                    high=max(1, columns.numel() // 4) + 1,
                    size=(),
                    device=device,
                    generator=generator,
                )
            )
            block = torch.randint(
                high=block_count,
                size=(columns.numel(),),
                device=device,
                generator=generator,
            )
            with CorrelatedContext(generator=generator):
                low = self.rate_off.sample_batch(size=block_count)
                high = self.rate_on.sample_batch(size=block_count)
            rate = torch.where(
                condition=factor.unsqueeze(0),
                input=high.unsqueeze(1),
                other=low.unsqueeze(1),
            )
            missing = (
                torch.rand(
                    size=(block_count, num_rows),
                    device=device,
                    generator=generator,
                )
                < rate
            )
            values[:, columns] = values[:, columns].masked_fill(
                mask=missing[block].t(), value=torch.nan
            )
        return mask_cells(x, values.isnan()), y
