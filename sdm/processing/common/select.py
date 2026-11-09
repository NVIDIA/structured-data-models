# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from sdm import EnsembleTable, Stype, StypeLike, TableTensor
from sdm.processing import EnsembleProcessor


class SelectColumns(EnsembleProcessor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep per semantic type.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block. ``"round_robin"`` assigns each ensemble
            member the next consecutive chunk, wrapping to the first column
            after the last. For five columns and ``max_columns=2``, members
            0, 1, and 2 receive columns (0, 1), (2, 3), and (4, 0).

    .. warning::
        Each ensemble member keeps at most ``max_columns`` columns per
        semantic type. If a table has more columns than
        ``num_estimators * max_columns``, some columns are never seen by any
        member, and this processor emits a warning. Below that bound, coverage
        can still have gaps when columns are chosen by shuffling: with
        ``"first"`` after :class:`~sdm.processing.ShuffleColumns`, a column is
        seen by about ``1 - (1 - max_columns / n) ** num_estimators`` of the
        runs for ``n`` columns. Increase ``num_estimators`` or reduce the
        number of columns before the model.
    """

    handles_stypes = tuple(Stype)
    requires_fit = False

    def __init__(
        self,
        max_columns: int,
        method: Literal["first", "round_robin"] = "first",
    ) -> None:
        super().__init__()
        if max_columns <= 0:
            raise ValueError("max_columns must be positive")
        self.max_columns = max_columns
        self.method = method

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        self._warn_if_columns_unseen(ensemble_table)
        if all(
            group.size(-1) <= self.max_columns
            for group in ensemble_table._iter_groups()
        ):
            return ensemble_table

        if self.method == "first":
            groups = []
            for group in ensemble_table._iter_groups():
                columns: Mapping[StypeLike, Sequence[str]] = {
                    stype: column_names[: self.max_columns]
                    for stype, column_names in group.column_groups
                }
                blocks = {
                    stype: block[..., : self.max_columns]
                    for stype, block in TableTensor.items(group)
                }
                groups.append(group.__class__(columns=columns, **blocks))
            return ensemble_table.replace_groups(groups)

        assert self.method == "round_robin"
        tables: list[TableTensor] = []
        for member_id in range(len(ensemble_table)):
            table = ensemble_table[member_id]
            columns_by_type = dict(table.column_groups)
            columns: dict[StypeLike, tuple[str, ...]] = {}
            blocks = {}
            for stype, block in TableTensor.items(table):
                column_names = columns_by_type[stype]
                num_columns = len(column_names)
                # Empty blocks are passed through unchanged.
                if num_columns == 0:
                    columns[stype] = column_names
                    blocks[stype] = block
                    continue
                chunk_size = min(self.max_columns, num_columns)
                start = member_id * chunk_size % num_columns
                first_count = min(chunk_size, num_columns - start)
                wrap_count = chunk_size - first_count
                # Consecutive chunk from ``start``, wrapping past the last
                # column back to the first.
                columns[stype] = (
                    column_names[start : start + first_count]
                    + column_names[:wrap_count]
                )
                if wrap_count == 0:
                    blocks[stype] = block[..., start : start + first_count]
                    continue
                blocks[stype] = torch.cat(
                    [
                        block[..., start : start + first_count],
                        block[..., :wrap_count],
                    ],
                    dim=-1,
                )
            tables.append(
                table.__class__(
                    columns=columns,
                    **blocks,
                )
            )
        return ensemble_table.replace_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _warn_if_columns_unseen(self, ensemble_table: EnsembleTable) -> None:
        widths: dict[str, int] = {}
        for group in ensemble_table._iter_groups():
            for stype, column_names in group.column_groups:
                key = str(getattr(stype, "value", stype))
                widths[key] = max(widths.get(key, 0), len(column_names))
        capacity = len(ensemble_table) * self.max_columns
        for name, width in widths.items():
            if width > capacity:
                warnings.warn(
                    f"{self.__class__.__name__} keeps at most "
                    f"{self.max_columns} {name} columns per estimator, but "
                    f"the table has {width} and there are "
                    f"{len(ensemble_table)} estimators. At least "
                    f"{width - capacity} columns are never seen by any "
                    f"estimator. Increase the number of estimators to at "
                    f"least {-(-width // self.max_columns)}, or reduce the "
                    f"number of columns before the model.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = [f"max_columns={self.max_columns!r}"]
        if self.method != "first":
            arguments.append(f"method={self.method!r}")
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
