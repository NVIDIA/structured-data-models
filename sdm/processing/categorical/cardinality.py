# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor


class DropHighCardinalityColumns(EnsembleProcessor):
    """Drop categorical columns with at least ``threshold`` distinct values.

    Count only nonmissing values observed during fitting, excluding unused
    vocabulary entries. Reuse the fitted column selection for later inputs.
    Fitting expects tables with shape ``[R, C]``, where ``R`` is the number
    of rows and ``C`` is the number of columns. Later inputs may have shape
    ``[..., C]``.

    Args:
        threshold: Minimum distinct-value count for dropping a column.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    def __init__(self, threshold: int) -> None:
        super().__init__()
        self.threshold = threshold
        self._columns: tuple[tuple[str, ...], ...] = ()

    def get_extra_state(self) -> tuple[tuple[str, ...], ...]:
        r""":meta private:"""  # noqa: D415
        return self._columns

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._columns = cast(tuple[tuple[str, ...], ...], state)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(
            EnsembleTable.from_table(table, num_members=1),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.drop_columns(self._columns[0])

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        columns = []
        for group in ensemble_table._iter_groups():
            codes = group.categorical.code
            num_tables, num_rows, _ = codes.shape
            if num_rows < self.threshold:
                columns.append(((),) * num_tables)
                continue

            # Count transitions in sorted codes: [B, R, C] -> [B, C].
            values = codes.sort(dim=-2).values
            first = (values[..., :1, :] >= 0).sum(dim=-2)
            changed = (values[..., 1:, :] != values[..., :-1, :]) & (
                values[..., 1:, :] >= 0
            )
            drop = (first + changed.sum(dim=-2)) >= self.threshold
            columns.append(
                tuple(
                    tuple(
                        name
                        for name, remove in zip(
                            group.categorical_columns, mask, strict=True
                        )
                        if remove
                    )
                    for mask in drop.tolist()
                )
            )
        self._columns = tuple(
            columns[group_id][position]
            for group_id, position in ensemble_table._locations
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(ensemble_table) != len(self._columns):
            raise ValueError("Expected the fitted number of ensemble members")

        outputs, member_ids = [], []
        output_ids: dict[tuple[tuple[int, int], tuple[str, ...]], int] = {}
        for member_id, (location, columns) in enumerate(
            zip(ensemble_table._locations, self._columns, strict=True)
        ):
            key = (location, columns)
            if key not in output_ids:
                output_ids[key] = len(outputs)
                outputs.append(ensemble_table[member_id].drop_columns(columns))
            member_ids.append(output_ids[key])
        return ensemble_table.replace_tables(
            tables=outputs, member_table_ids=member_ids
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold!r})"
        )
