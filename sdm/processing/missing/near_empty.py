# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing._random import SeedMixin, count, rank, regime, uniform
from sdm.processing.missing._cells import cell_values, mask_cells


class MaskNearEmpty(SeedMixin, Processor):
    """Mask most cells of column families with overlapping observed rows.

    When the processor is fitted, each table selects a fraction of its
    columns and splits them into families with power-law sizes. Each family
    receives a correlated observed rate in ``observed``, sampled on a
    logarithmic scale. The transform initially keeps the same rows within a
    family, at least ``min_observed``, and may swap individual rows to increase
    observed value diversity.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        columns: Range of the fraction of the columns that are masked.
        observed: Range of the observed rate of a family.
        min_observed: Minimum number of observed rows of a family.
        target: A column that is never masked.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True

    _family: Tensor
    _observed_rate: Tensor

    def __init__(
        self,
        *,
        columns: tuple[float, float] = (0.5, 0.8),
        observed: tuple[float, float] = (0.01, 0.1),
        min_observed: int = 4,
        target: str | None = None,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.observed = observed
        self.min_observed = min_observed
        self.target = target
        # [..., C]; the family of each column (-1 if not selected), and the
        # observed rate of each family id.
        self.register_buffer("_family", torch.empty(0, dtype=torch.long))
        self.register_buffer("_observed_rate", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        batch = table.size()[:-2]
        device = table.device
        names = (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
        )
        num_columns = len(names)
        is_feature = [name != self.target for name in names]
        num_features = sum(is_feature)
        features = torch.tensor(is_feature, device=device)
        total = features.sum().expand(batch)
        num_selected = count(total, *self.columns, generator=generator)
        num_families = count(num_selected, 0.06, 0.16, generator=generator)
        max_families = min(num_features, math.ceil(0.16 * num_features))
        if max_families == 0:
            self._set_buffer(
                "_family",
                torch.full(
                    (*batch, num_columns),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
            )
            self._set_buffer(
                "_observed_rate", torch.zeros((*batch, 0), device=device)
            )
            return

        # Columns in random order; the first ones pin one column per family.
        # The target ranks last, so it is never selected.
        u = torch.rand(
            (2, *batch, num_columns), device=device, generator=generator
        )
        position = rank(u[0].masked_fill(~features, 2.0), dim=-1)  # [..., C]
        active_family = (
            torch.arange(num_columns, device=device) < num_families[..., None]
        )
        family_rank = rank(u[1].masked_fill(~active_family, 2.0), dim=-1)
        exponent = uniform(
            low=0.9,
            high=1.6,
            size=(*batch, 1),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        family_weight = (family_rank + 1).pow(
            -exponent
        ) * active_family  # [..., C]
        sampled_family = family_weight.view(-1, num_columns).multinomial(
            num_columns,
            replacement=True,
            generator=generator,
        )
        family = torch.where(
            position < num_families[..., None],
            position,
            sampled_family.view_as(position),
        )
        self._set_buffer(
            "_family",
            family.masked_fill(position >= num_selected[..., None], -1),
        )
        low, high = self.observed
        self._set_buffer(
            "_observed_rate",
            (
                regime(
                    (*batch, max_families),
                    device=device,
                    generator=generator,
                )
                .mul(math.log(high / low))
                .add(math.log(low))
                .exp()
            ),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        max_families = self._observed_rate.size(-1)
        if max_families == 0 or table.size(-2) == 0:
            return table
        values = cell_values(table)  # [..., R, C]
        *batch, num_rows, _ = values.shape
        device = values.device
        generator = self._transform_generator()
        selected = self._family >= 0
        family_size = torch.zeros_like(self._observed_rate).scatter_add_(
            -1,
            self._family.clamp(min=0),
            selected.to(self._observed_rate.dtype),
        )  # [..., F], family sizes
        num_observed = (
            (self._observed_rate.double() * num_rows)
            .ceil()
            .clamp(self.min_observed, num_rows)
        )
        family_order = family_size.argsort(
            dim=-1, descending=True, stable=True
        )

        # Float64 keeps the random tie order exact, for every input dtype.
        load = torch.zeros(
            (*batch, num_rows), dtype=torch.float64, device=device
        )
        observed = []
        for i in range(max_families):
            family = family_order[..., i : i + 1]
            tie = torch.rand(
                (*batch, num_rows), dtype=torch.float64, generator=generator
            )
            position = rank(load + tie.to(device), dim=-1)
            is_observed = position < num_observed.gather(-1, family)
            load += family_size.gather(-1, family) * is_observed
            observed.append(is_observed)
        # [..., R, C]; each column takes the observed rows of its family.
        family_position = family_order.argsort(dim=-1).gather(
            -1, self._family.clamp(min=0)
        )
        observed = torch.stack(observed, dim=-1).gather(
            -1, family_position[..., None, :].expand_as(values)
        )

        # Swap rows for columns with fewer than two distinct observed values.
        finite = values.isfinite()
        seen = observed & finite
        observed_max = values.masked_fill(~seen, -torch.inf).amax(
            dim=-2, keepdim=True
        )
        observed_min = values.masked_fill(~seen, torch.inf).amin(
            dim=-2, keepdim=True
        )
        candidate = finite & ~observed & (values != observed_max)
        swap = (observed_max <= observed_min) & candidate.any(
            dim=-2, keepdim=True
        )
        u = torch.rand((2, *values.shape), generator=generator).to(device)
        drop = u[0].masked_fill(~observed, -1.0).argmax(dim=-2, keepdim=True)
        add = u[1].masked_fill(~candidate, -1.0).argmax(dim=-2, keepdim=True)
        observed = observed.scatter(
            -2, drop, observed.gather(-2, drop) & ~swap
        ).scatter(-2, add, observed.gather(-2, add) | swap)
        return mask_cells(table, selected[..., None, :] & ~observed)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"columns={self.columns}, observed={self.observed}, "
            f"min_observed={self.min_observed}, target={self.target!r})"
        )
