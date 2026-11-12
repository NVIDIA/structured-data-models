# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import (
    CategoricalTensor,
    EnsembleData,
    Stype,
    TableLike,
    TableTensor,
)
from sdm.processing import Processor
from sdm.processing.categorical._categorical import (
    _check_categorical_codes,
    _check_categories,
)
from sdm.tensor.categorical import _iter_categories


class ImputeMode(Processor):
    """Replace missing categorical values with fitted per-column modes.

    Negative codes represent missing values. Ties select the lowest code.

    Transform inputs must use the fitted per-column category vocabularies.
    Use :class:`~sdm.processing.AlignCategories` before this processor when
    training and transform inputs were tensorized independently.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    _fill_values: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        # TODO: Add a separate processor that encodes missing values as their
        # own category instead of imputing an observed one.
        self.register_buffer("_categories", None)
        self.register_buffer(
            "_fill_values",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fill, categories = EnsembleData.execute(
            self._fit_values, table, batched=True
        )
        self.register_buffer("_fill_values", fill)
        self.register_buffer("_categories", categories)

    def _fit_values(
        self, table: TableTensor
    ) -> tuple[torch.Tensor, CategoricalTensor]:
        data = table.categorical
        _check_categorical_codes(table)
        fill_values: list[torch.Tensor] = []
        columns = table.categorical_columns
        observed_mask = data.isfinite()
        for index, category in enumerate(table.categorical.categories):
            codes = data[..., index]  # [*batch, n_samples]
            observed = observed_mask[..., index]
            if not bool(observed.any(dim=-1).all()):
                raise ValueError(
                    "Cannot fit 'ImputeMode' because categorical "
                    f"column {columns[index]!r} has no observed values."
                )

            # Accumulate category counts per batch.
            counts = torch.zeros(
                (*codes.shape[:-1], category.numel()),
                dtype=torch.long,
                device=codes.device,
            )
            counts.scatter_add_(
                -1,
                codes.clamp_min(0).to(torch.long),
                observed.to(torch.long),
            )
            fill_values.append(counts.argmax(dim=-1, keepdim=True))

        fill = (
            torch.stack(fill_values, dim=-1)
            if len(fill_values) > 0
            else data.code.new_empty(
                (*data.shape[:-2], 1, 0), dtype=torch.long
            )
        )
        categories = CategoricalTensor(
            data.code.new_empty((*data.shape[:-2], 0, data.size(-1))),
            categories=data.categories,
        )
        return fill, categories

    def _transform(self, table: TableLike) -> TableLike:
        return EnsembleData.execute(
            self._replace_missing,
            table,
            self._fill_values,
            self._categories,
            batched=True,
        )

    def _replace_missing(
        self,
        table: TableTensor,
        fill: torch.Tensor,
        categories: CategoricalTensor,
    ) -> TableTensor:
        _check_categories(table, tuple(_iter_categories(categories)))
        _check_categorical_codes(table)
        code = table.categorical.where(
            table.categorical.isfinite(),
            fill.to(
                dtype=table.categorical.dtype,
                device=table.categorical.device,
            ),
        )
        categorical = CategoricalTensor(
            code=code,
            categories=tuple(_iter_categories(table.categorical)),
        )
        return table.replace_blocks(categorical=categorical)
