# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing._random import (
    SeedMixin,
    count,
    log_uniform,
    randint,
    rank,
    regime,
    subset,
    uniform,
)
from sdm.processing.missing._cells import cell_values, mask_cells


class MaskLatent(SeedMixin, Processor):
    """Mask column blocks under a latent-factor MAR mechanism.

    Each table selects proxy columns and groups some remaining columns into
    blocks. A latent row factor derived from the proxy values switches each
    block between two missing-cell rates.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        num_proxies: Range of the number of the proxy columns.
        columns: Range of the fraction of non-proxy columns assigned to
            blocks.
        prevalence: Range of the fraction of the rows where the factor is
            on.
        rate_off: Range of the missing-cell rate when the factor is off.
        rate_on: Range of the missing-cell rate when the factor is on.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True

    _weight: Tensor
    _block: Tensor
    _rate: Tensor
    _noise_scale: Tensor
    _on_fraction: Tensor

    def __init__(
        self,
        *,
        num_proxies: tuple[int, int] = (1, 3),
        columns: tuple[float, float] = (0.2, 0.6),
        prevalence: tuple[float, float] = (0.1, 0.5),
        rate_off: tuple[float, float] = (0.0, 0.1),
        rate_on: tuple[float, float] = (0.3, 0.9),
    ) -> None:
        super().__init__()
        self.num_proxies = num_proxies
        self.columns = columns
        self.prevalence = prevalence
        self.rate_off = rate_off
        self.rate_on = rate_on
        self.register_buffer("_weight", torch.empty(0))
        self.register_buffer("_block", torch.empty(0, dtype=torch.long))
        self.register_buffer("_rate", torch.empty(0))
        self.register_buffer("_noise_scale", torch.empty(0))
        self.register_buffer("_on_fraction", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        values = cell_values(table)
        *batch, _, num_columns = values.shape
        device = values.device
        min_proxies, max_proxies = self.num_proxies
        num_proxies = randint(
            low=min_proxies,
            high=max(min_proxies, min(max_proxies, num_columns - 1)),
            size=batch,
            device=device,
            generator=generator,
        )
        well_observed = (
            values.isfinite().float().mean(dim=-2) >= 0.5
        )  # [..., C]
        has_enough_proxies = (
            well_observed.sum(dim=-1, keepdim=True) >= num_proxies[..., None]
        )
        is_proxy = subset(
            eligible=well_observed | ~has_enough_proxies,
            count=num_proxies,
            generator=generator,
        )
        num_remaining = (~is_proxy).sum(dim=-1)
        num_driven = count(num_remaining, *self.columns, generator=generator)
        is_driven = subset(~is_proxy, num_driven, generator=generator)
        num_blocks = randint(
            low=1,
            high=(is_driven.sum(dim=-1) // 4).clamp(min=1),
            size=batch,
            device=device,
            generator=generator,
        )
        block_id = randint(
            low=0,
            high=num_blocks[..., None] - 1,
            size=(*batch, num_columns),
            device=device,
            generator=generator,
        )
        self._set_buffer("_block", block_id.masked_fill(~is_driven, -1))
        self._set_buffer(
            "_weight",
            torch.randn(
                (*batch, num_columns), device=device, generator=generator
            )
            * is_proxy,
        )
        # [..., 2, K]; factor-off and factor-on rates for each block.
        lower, upper = torch.tensor(
            [self.rate_off, self.rate_on], device=device
        ).unbind(-1)
        max_blocks = max(1, num_columns // 4)
        self._set_buffer(
            "_rate",
            regime(
                size=(*batch, 2, max_blocks),
                device=device,
                generator=generator,
            )
            * (upper - lower)[:, None]
            + lower[:, None],
        )
        self._set_buffer(
            "_noise_scale",
            log_uniform(
                low=0.25,
                high=1.0,
                size=(*batch, 1),
                device=device,
                dtype=torch.float32,
                generator=generator,
            ),
        )
        self._set_buffer(
            "_on_fraction",
            uniform(
                low=self.prevalence[0],
                high=self.prevalence[1],
                size=(*batch, 1),
                device=device,
                dtype=torch.float64,
                generator=generator,
            ),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        values = cell_values(table)  # [..., R, C]
        *batch, num_rows, _ = values.shape
        if num_rows == 0:
            return table
        device = values.device
        generator = self._transform_generator()
        mean = values.nanmean(dim=-2, keepdim=True)
        std = (values - mean).square().nanmean(dim=-2, keepdim=True).sqrt()
        values = ((values - mean) / std.clamp(min=1e-6)).nan_to_num(0.0)
        score = values @ self._weight[..., None].to(values.dtype)
        score = score.squeeze(-1)  # [..., R]
        score = (score - score.mean(dim=-1, keepdim=True)) / score.std(
            dim=-1, keepdim=True, correction=0
        ).clamp(min=1e-6)
        noise = torch.randn((*batch, num_rows), generator=generator)
        score = score + self._noise_scale * noise.to(device)
        row_rank = rank(score, dim=-1)
        if num_rows == 1:
            factor_draw = torch.rand(
                self._on_fraction.shape, device="cpu", generator=generator
            ).to(device)
            factor_on = factor_draw < self._on_fraction
        else:
            factor_on = row_rank > (1 - self._on_fraction) * (num_rows - 1)
            factor_on |= self._on_fraction == 1

        # [..., R, K]; one candidate mask per block.
        block_rate = torch.where(
            factor_on[..., None],
            self._rate[..., 1, None, :],
            self._rate[..., 0, None, :],
        )
        u = torch.rand(block_rate.shape, generator=generator)
        missing = (u.to(device) < block_rate).gather(
            -1,
            self._block.clamp(min=0)[..., None, :].expand_as(values),
        )
        return mask_cells(table, missing & (self._block >= 0)[..., None, :])

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_proxies={self.num_proxies}, columns={self.columns}, "
            f"prevalence={self.prevalence}, rate_off={self.rate_off}, "
            f"rate_on={self.rate_on})"
        )
