# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from typing import Literal

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor


class ReduceEstimators(EnsembleProcessor):
    """Reduce the leading ensemble dimension of model outputs.

    Input must be a numerical output table with shape ``[E, ..., R, O]``.
    ``E`` is the non-empty leading ensemble dimension, ``R`` is the row
    dimension, and ``O`` is the output-column dimension. The result has shape
    ``[..., R, O]`` and retains the input column schema, device, and floating
    dtype.

    Place processors that support stacked outputs before this processor.
    Processors after it receive an already-reduced output table.
    Ensemble transformation returns one member containing the reduction.

    Args:
        method: Reduction applied across ensemble members. ``"mean"``
            averages all members. ``"trimmed_mean"`` sorts the members at
            every output coordinate, removes ``proportion`` from both ends,
            and averages the remainder.
        proportion: Proportion removed from each end.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        method: Literal["mean", "trimmed_mean"] = "mean",
        proportion: float | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.proportion = proportion

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if ensemble_table.num_members == 0:
            raise ValueError("Expected at least one ensemble member.")

        reference = ensemble_table.table(0)
        extra = reference.active_stypes - self.handles_stypes
        if extra:
            found = ", ".join(sorted(extra))
            raise ValueError(
                f"Expected a numerical-only output table (also found {found})."
            )
        reference_columns = reference.columns[Stype.numerical]

        groups: list[torch.Tensor] = []
        for group in ensemble_table:
            if group.stypes != reference.stypes:
                raise ValueError(
                    "Expected ensemble members to have the same column names "
                    "and stypes."
                )

            numerical = group.numerical
            columns = group.columns[Stype.numerical]
            if columns != reference_columns:
                column_to_index = {
                    column: index for index, column in enumerate(columns)
                }
                numerical = torch.cat(
                    tensors=[
                        numerical.narrow(
                            dim=-1,
                            start=column_to_index[column],
                            length=1,
                        )
                        for column in reference_columns
                    ],
                    dim=-1,
                )

            if numerical.shape[1:] != reference.numerical.shape:
                raise ValueError(
                    "Expected ensemble members to have the same shape."
                )
            groups.append(numerical)

        if self.method == "mean":
            counts_by_location = Counter(ensemble_table._locations)
            total = None
            for group_id, numerical in enumerate(groups):
                counts = numerical.new_tensor(
                    [
                        counts_by_location[(group_id, position)]
                        for position in range(numerical.size(0))
                    ]
                )
                partial = torch.tensordot(
                    a=counts,
                    b=numerical,
                    dims=([0], [0]),
                )
                total = partial if total is None else total + partial
            assert total is not None
            reduced = total / ensemble_table.num_members
        else:
            assert self.method == "trimmed_mean"
            members = torch.stack(
                [
                    groups[group_id][position]
                    for group_id, position in ensemble_table._locations
                ]
            )  # [E, ..., R, O]
            output = self._transform(
                reference.__class__(
                    columns={Stype.numerical: reference_columns},
                    numerical=members,
                )
            )
            return EnsembleTable.from_table(output, num_members=1)

        output = reference.replace_blocks(numerical=reduced)
        return EnsembleTable.from_table(output, num_members=1)

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.dim() < 3:
            raise ValueError(
                "Expected a leading ensemble dimension in an output table "
                f"with at least 3 dimensions (got {table.dim()}D)."
            )
        if table.size(0) == 0:
            raise ValueError("Expected at least one ensemble member.")
        extra = table.active_stypes - self.handles_stypes
        if extra:
            found = ", ".join(sorted(extra))
            raise ValueError(
                f"Expected a numerical-only output table (also found {found})."
            )

        numerical = table.numerical  # [E, ..., R, O]
        if self.method == "mean":
            reduced = numerical.mean(dim=0)
        else:
            assert self.method == "trimmed_mean"
            assert self.proportion is not None
            assert 0 < self.proportion < 0.5
            cut = int(self.proportion * numerical.size(0))
            if cut == 0:
                reduced = numerical.mean(dim=0)
            else:
                ordered = numerical.sort(dim=0).values
                reduced = ordered.narrow(
                    dim=0,
                    start=cut,
                    length=numerical.size(0) - 2 * cut,
                ).mean(dim=0)

        return table.__class__(
            columns={Stype.numerical: table.columns[Stype.numerical]},
            numerical=reduced,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        prefix = f"{' ' * indent}{self.__class__.__name__}"
        if self.method == "mean":
            return f"{prefix}(method={self.method!r})"
        return (
            f"{prefix}(method={self.method!r}, proportion={self.proportion})"
        )
