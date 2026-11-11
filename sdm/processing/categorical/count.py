# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

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


class AddCategoryCounts(Processor):
    """Add the log row count of each categorical value.

    Each eligible column receives one numerical column ``<column>__count``
    holding ``log1p`` of the number of fitted rows sharing the row's code.
    Negative codes (missing or unknown values) share one count.

    Transform inputs must use the fitted per-column category vocabularies.
    Use :class:`~sdm.processing.AlignCategories` before this processor when
    training and transform inputs were tensorized independently.

    Args:
        min_cardinality: A count column is added when its fitted vocabulary
            size exceeds this value. Missing values do not contribute to
            vocabulary size. Defaults to 0.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    _log_counts: Tensor
    _num_categories: Tensor

    def __init__(self, min_cardinality: int = 0) -> None:
        super().__init__()
        self.min_cardinality = min_cardinality
        self.register_buffer("_categories", None)
        self.register_buffer("_log_counts", torch.empty(0))
        self.register_buffer(
            "_num_categories",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        counts, sizes, categories = EnsembleData.execute(
            self._fit_counts, table, batched=True
        )
        self.register_buffer("_log_counts", counts)
        self.register_buffer("_num_categories", sizes)
        self.register_buffer("_categories", categories)

    def _fit_counts(
        self, table: TableTensor
    ) -> tuple[Tensor, Tensor, CategoricalTensor]:
        _check_categorical_codes(table)
        categorical = table.categorical
        columns = tuple(
            column
            for column, categories in zip(
                table.categorical_columns, _iter_categories(categorical)
            )
            if categories.numel() > self.min_cardinality
        )
        categorical = table.select_columns(columns).categorical
        codes = categorical.code  # [*batch, num_rows, num_columns]
        sizes = [
            categories.numel() for categories in _iter_categories(categorical)
        ]
        num_categories = codes.new_tensor(sizes, dtype=torch.long)
        # Negative codes accumulate in the slot behind their column's
        # vocabulary: [*batch, num_columns, max(sizes) + 1].
        slots = torch.where(codes >= 0, codes.long(), num_categories)
        slots = slots.transpose(-2, -1)
        counts = torch.zeros(
            (*codes.shape[:-2], len(sizes), max(sizes, default=0) + 1),
            dtype=torch.long,
            device=codes.device,
        )
        counts.scatter_add_(-1, slots, slots.new_ones(()).expand_as(slots))

        dtype = table.numerical.dtype
        count_dtype = (
            torch.float32
            if dtype in {torch.float16, torch.bfloat16}
            else dtype
        )
        log_counts = counts.to(count_dtype).log1p().to(dtype)
        categories = CategoricalTensor(
            table.categorical.code.new_empty(
                (*codes.shape[:-2], 0, table.categorical.size(-1))
            ),
            categories=table.categorical.categories,
        )
        num_categories = num_categories.expand(*codes.shape[:-2], len(sizes))
        return log_counts, num_categories, categories

    def _transform(self, table: TableLike) -> TableLike:
        return EnsembleData.execute(
            self._add_counts,
            table,
            self._log_counts,
            self._num_categories,
            self._categories,
            batched=True,
        )

    def _add_counts(
        self,
        table: TableTensor,
        log_counts: Tensor,
        num_categories: Tensor,
        categories: CategoricalTensor,
    ) -> TableTensor:
        vocabularies = tuple(_iter_categories(categories))
        _check_categories(table, vocabularies)
        _check_categorical_codes(table)
        columns = tuple(
            column
            for column, category in zip(
                table.categorical_columns, vocabularies
            )
            if category.numel() > self.min_cardinality
        )
        if not columns:
            return table
        codes = table.select_columns(columns).categorical.code
        slots = torch.where(
            codes >= 0, codes.long(), num_categories.unsqueeze(-2)
        )
        counts = log_counts.gather(-1, slots.transpose(-2, -1))

        out_table = TableTensor(
            columns={
                Stype.numerical.value: tuple(
                    f"{column}__count" for column in columns
                ),
            },
            numerical=counts.transpose(-2, -1).to(table.numerical.dtype),
        )
        return cast(TableTensor, torch.cat([table, out_table], dim=-1))

    def __repr__(self, *, indent: int = 0) -> str:
        if self.min_cardinality == 0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(min_cardinality={self.min_cardinality!r})"
        )
