# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch

from sdm import EnsembleData, EnsembleTable, Stype, TableLike, TableTensor
from sdm._warnings import warn_once
from sdm.processing import Processor


class SelectColumns(Processor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep per semantic type.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block. ``"round_robin"`` assigns each ensemble
            member the next consecutive chunk, wrapping to the first column
            after the last. For five columns and ``max_columns=2``, members
            0, 1, and 2 receive columns (0, 1), (2, 3), and (4, 0).
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

    def _transform(self, table: TableLike) -> TableLike:
        num_members = len(table) if isinstance(table, EnsembleTable) else 1
        members = (
            EnsembleData(
                groups=tuple(range(num_members)),
                locations=tuple((i, 0) for i in range(num_members)),
            )
            if isinstance(table, EnsembleTable)
            and self.method == "round_robin"
            else 0
        )
        return EnsembleData.execute(
            self._select, table, members, num_members, batched=True
        )

    def _select(
        self, table: TableTensor, member: int, num_members: int
    ) -> TableTensor:
        if all(
            block.size(-1) <= self.max_columns
            for _, block in TableTensor.items(table)
        ):
            return table
        if any(
            block.size(-1) > num_members * self.max_columns
            for _, block in TableTensor.items(table)
        ):
            warn_once(
                key="select-columns-coverage",
                message=(
                    "Some columns will not be seen by any estimator. "
                    "Increase the number of estimators for better coverage."
                ),
            )
        columns = {}
        blocks = {}
        for stype, block in TableTensor.items(table):
            names = table.columns[stype]
            width = len(names)
            count = min(self.max_columns, width)
            start = (
                member * count % width
                if width and self.method == "round_robin"
                else 0
            )
            first = min(count, width - start)
            wrap = count - first
            columns[stype.value] = names[start : start + first] + names[:wrap]
            blocks[stype] = (
                torch.cat(
                    (block[..., start : start + first], block[..., :wrap]),
                    dim=-1,
                )
                if wrap
                else block[..., start : start + first]
            )
        return TableTensor(columns=columns, **blocks)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = [f"max_columns={self.max_columns!r}"]
        if self.method != "first":
            arguments.append(f"method={self.method!r}")
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
