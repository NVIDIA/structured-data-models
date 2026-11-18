# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.prior import Correlated, CorrelatedContext, LogUniformInt, UniformInt
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import _standardize
from sdm.prior.tabular.transform._table import (
    _unique_values,
    categorical_mask,
    feature_order,
    feature_values,
)
from sdm.prior.util import random_weight


class RecodeHighCardinality(PriorTransform):
    """Recode selected finite feature columns as high-cardinality categories.

    Args:
        min_levels: Minimum requested levels and required number of rows.
        num_columns: Inclusive range for the number of columns to recode.
        categorical: Whether existing categorical columns are eligible.
        standardize: Whether category values use standardized numeric levels.
    """

    def __init__(
        self,
        *,
        min_levels: int = 201,
        num_columns: tuple[int, int] = (1, 3),
        categorical: bool = False,
        standardize: bool = False,
    ) -> None:
        self.min_levels = min_levels
        self.num_columns = num_columns
        self.categorical = categorical
        self.standardize = standardize

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        names = x.columns[Stype.numerical]
        values = x.numerical
        if self.categorical:
            names = (*names, *x.columns[Stype.categorical])
            names = tuple(names[i] for i in feature_order(x))
            values = feature_values(x)
        if (
            values.size(-1) < self.num_columns[0]
            or values.size(-2) < self.min_levels
        ):
            return x, y
        cat_mask = (
            categorical_mask(x)
            if self.categorical
            else torch.zeros(
                values.size(-1), dtype=torch.bool, device=x.device
            )
        )
        num_rows, num_cols = values.shape
        device = values.device
        num_recoded = min(self.num_columns[1], num_cols)

        with CorrelatedContext(generator=generator):
            count = UniformInt(self.num_columns[0], num_recoded).sample(
                generator=generator
            )
            col_index = torch.argsort(
                torch.rand(num_cols, device=device, generator=generator)
            )[:num_recoded]
            selected = torch.arange(num_recoded, device=device) < count
            cardinality = Correlated(
                LogUniformInt(self.min_levels, num_rows),
                key="RecodeHighCardinality.cardinality",
            ).sample_batch(size=num_recoded)

        source = values[:, col_index].mT  # [n, R]
        noise = torch.randn(
            source.shape,
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        order = (source.double() + 1e-6 * noise).argsort(dim=-1)
        sorted_source = source.gather(-1, order)

        # Power-law level widths over the first ``cardinality`` channels,
        # each padded to at least one row:
        keep = torch.arange(num_rows, device=device) < cardinality.unsqueeze(
            -1
        )
        weight = random_weight(
            size=(num_recoded, num_rows),
            dtype=torch.float64,
            generator=generator,
        )
        weight = weight.sort(dim=-1, descending=True).values * keep
        weight = weight / weight.sum(dim=-1, keepdim=True)
        spare = (num_rows - cardinality).unsqueeze(-1)
        edges = ((keep + spare * weight) / num_rows).cumsum(dim=-1)
        position = (torch.arange(num_rows, device=device) + 0.5) / num_rows
        level = torch.searchsorted(
            sorted_sequence=edges,
            input=position.expand_as(edges).contiguous(),
            right=True,
        )
        level = torch.minimum(level, (cardinality - 1).unsqueeze(-1))

        # Also cut at source boundaries when the levels can cover them, so
        # codes refine the source instead of merging distinct values:
        same_source = sorted_source[..., 1:] == sorted_source[..., :-1]
        num_distinct = num_rows - same_source.sum(dim=-1)
        refine = (cardinality >= num_distinct).unsqueeze(-1)
        new_code = (level[..., 1:] != level[..., :-1]) | (
            refine & ~same_source
        )
        sorted_code = torch.nn.functional.pad(
            input=new_code.long().cumsum(dim=-1), pad=(1, 0)
        )
        perm = torch.argsort(
            torch.rand(sorted_code.shape, device=device, generator=generator),
            dim=-1,
        )
        code = torch.empty_like(sorted_code).scatter_(
            dim=-1, index=order, src=perm.gather(-1, sorted_code)
        )
        # Codes above 256 are not representable in half precision:
        code = code.mT.float()
        if self.standardize:
            code = _standardize(code)
        code = code.to(values.dtype)

        apply = selected & source.isfinite().all(dim=-1)
        recoded_values = torch.where(apply, code, source.mT)
        output = values.index_copy(
            dim=1, index=col_index, source=recoded_values
        )
        categorical = cat_mask.index_copy(
            dim=0, index=col_index, source=cat_mask[col_index] | apply
        )
        changed = categorical & (
            ~cat_mask | ((output != values) & values.isfinite()).any(dim=0)
        )
        selected = changed.nonzero().flatten().tolist()
        if not selected:
            return x, y
        columns = [names[index] for index in selected]
        categories, codes = zip(
            *[_unique_values(output[:, index]) for index in selected]
        )
        recoded = TableTensor(
            columns={Stype.categorical: columns},
            categorical=CategoricalTensor(
                code=torch.stack(codes, dim=-1).to(x.categorical.code.dtype),
                categories=categories,
            ),
        )
        return cast(
            TableTensor, torch.cat((x.drop_columns(columns), recoded), dim=-1)
        ), y
