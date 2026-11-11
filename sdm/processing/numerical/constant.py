# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch

from sdm import EnsembleData, Stype, TableLike, TableTensor
from sdm.processing import Processor


class DropConstantColumns(Processor):
    """Remove non-informative numerical columns learned during fit.

    Columns are retained when they have more than ``threshold`` distinct
    values. When the number of samples is less than or equal to ``threshold``,
    all columns are preserved. NaN is counted once as a distinct value.

    Only numerical columns are supported. Convert other stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.
    Fitting expects data with shape ``[N, C]``, where ``N`` is the number of
    rows and ``C`` is the number of numerical columns. The learned selection
    can transform later tables with shape ``[..., C]``.

    Args:
        threshold: Columns with at most this many unique values are removed.
            Must be positive.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    def __init__(
        self,
        *,
        threshold: int = 1,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.threshold = threshold
        # TODO: Consider recording the fitted column names if transforms should
        # verify that the numerical schema and order match fit.
        self.register_buffer("_kept_indices", None)

    def _keep_mask(self, data: torch.Tensor) -> torch.Tensor:
        # [N, C] or [..., N, C] -> [C] or [..., C].
        # Preserve the schema when too few rows can exceed the threshold.
        if data.size(-2) <= self.threshold:
            return data.new_ones(
                (*data.shape[:-2], data.size(-1)),
                dtype=torch.bool,
            )
        if self.threshold == 1:
            # Any mismatch with the first row proves a second unique value.
            different = (data != data[..., :1, :]).any(dim=-2)
            return different & ~data.isnan().all(dim=-2)

        # A sorted column with k unique values has k - 1 transitions.
        values = data.sort(dim=-2).values
        left, right = values[..., :-1, :], values[..., 1:, :]
        changed = (right != left) & ~(right.isnan() & left.isnan())
        return changed.sum(dim=-2) >= self.threshold

    @staticmethod
    def _select_columns(
        table: TableTensor,
        kept_indices: torch.Tensor,
    ) -> TableTensor:
        numerical_columns = table.numerical_columns
        if kept_indices.numel() == len(numerical_columns):
            return table
        kept_numerical_columns = tuple(
            numerical_columns[index] for index in kept_indices.tolist()
        )
        columns = [
            column
            for stype, stype_columns in table.column_groups
            for column in (
                kept_numerical_columns
                if stype == Stype.numerical
                else stype_columns
            )
        ]
        return table.select_columns(columns)

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        keep = EnsembleData.execute(
            self._keep_mask, table.numerical, batched=True
        )
        self.register_buffer("_kept_indices", keep.nonzero(as_tuple=True)[0])

    def _transform(self, table: TableLike) -> TableLike:
        return EnsembleData.execute(
            self._select_columns, table, self._kept_indices
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.threshold == 1:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold!r})"
        )
