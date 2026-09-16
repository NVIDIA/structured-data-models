# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import Processor
from sdm.processing.categorical._categorical import _check_categorical_codes


class AddLevelCounts(Processor):
    """Add log level counts of high-cardinality categorical columns.

    Fitting counts, independently for every leading batch element, the rows
    of each category code in every categorical column. A column whose number
    of observed categories exceeds ``min_cardinality`` receives one numerical
    column ``<column>__count`` holding ``log1p`` of the fitted count of each
    row's code. Rows with a negative code (missing or unknown values) receive
    ``log1p`` of the number of fitted rows with a negative code. Categorical
    columns stay unchanged.

    Transform inputs must use the fitted per-column category vocabularies.
    The processor raises if they do not match. Column names are not validated.
    Use :class:`~sdm.processing.AlignCategories` before this processor when
    training and transform inputs were tensorized independently.

    Args:
        min_cardinality: Number of observed categories a column must exceed
            to receive a count column.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self, min_cardinality: int = 50) -> None:
        super().__init__()
        self.min_cardinality = min_cardinality
        self._columns: tuple[int, ...] = ()
        self._categories: BufferList[Tensor] = BufferList()
        self._log_counts: BufferList[Tensor] = BufferList()
        self.register_buffer("_log_unknown", torch.empty(0))

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._columns

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._columns = cast(tuple[int, ...], state)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        codes = table.categorical.code  # [*batch, num_rows, num_columns]
        _check_categorical_codes(table)
        observed = codes >= 0
        dtype = table.numerical.dtype
        count_dtype = (
            torch.float32
            if dtype in {torch.float16, torch.bfloat16}
            else dtype
        )
        log_counts = []
        cardinalities = []
        for index, categories in enumerate(table.categorical.categories):
            counts = torch.zeros(
                (*codes.shape[:-2], categories.numel()),
                dtype=torch.long,
                device=codes.device,
            )
            if categories.numel() > 0:
                counts.scatter_add_(
                    -1,
                    codes[..., index].clamp_min(0).long(),
                    observed[..., index].long(),
                )
            log_counts.append(counts.to(count_dtype).log1p().to(dtype))
            cardinalities.append((counts > 0).sum(dim=-1))

        selected = torch.stack(cardinalities, dim=-1)
        selected = selected.reshape(-1, len(cardinalities)).amax(dim=0)
        selected = selected > self.min_cardinality
        self._columns = tuple(
            index for index, flag in enumerate(selected.tolist()) if flag
        )
        self._categories = BufferList(table.categorical.categories)
        self._log_counts = BufferList(
            log_counts[index] for index in self._columns
        )
        unknown = (~observed).sum(dim=-2).to(count_dtype).log1p().to(dtype)
        self._log_unknown = unknown[..., list(self._columns)]

    def _transform(self, table: TableTensor) -> TableTensor:
        if not self._columns:
            return table

        self._check_categories(table)
        _check_categorical_codes(table)
        codes = table.categorical.code
        outs = []
        for position, index in enumerate(self._columns):
            column = codes[..., index]
            log_counts = self._log_counts[position]
            num_categories = log_counts.size(-1)
            values = log_counts.gather(
                -1,
                column.clamp(min=0, max=num_categories - 1).long(),
            )
            values = torch.where(column < num_categories, values, 0.0)
            outs.append(
                torch.where(
                    column >= 0,
                    values,
                    self._log_unknown[..., position].unsqueeze(-1),
                )
            )

        columns = table.columns[Stype.categorical]
        out_table = TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{columns[index]}__count" for index in self._columns
                ),
            },
            numerical=torch.stack(outs, dim=-1).to(table.numerical.dtype),
        )
        return cast(TableTensor, torch.cat([table, out_table], dim=-1))

    def _check_categories(self, table: TableTensor) -> None:
        columns = table.columns[Stype.categorical]
        if len(table.categorical.categories) != len(self._categories):
            raise ValueError(
                f"Expected {len(self._categories)} fitted categorical "
                f"columns (got {len(columns)})."
            )
        for index, (actual, expected) in enumerate(
            zip(table.categorical.categories, self._categories)
        ):
            expected = expected.to(device=actual.device)
            if not actual.equal(expected):
                raise ValueError(
                    "Expected the category vocabulary for categorical column "
                    f"{columns[index]!r} to match the fitted values and "
                    "order. "
                    "Use 'AlignCategories' before this processor for "
                    "independently tensorized inputs."
                )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.min_cardinality == 50:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(min_cardinality={self.min_cardinality!r})"
        )
