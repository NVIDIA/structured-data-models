# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import EnsembleProcessor, Processor


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
            every output coordinate, drops ``proportion`` of them from each
            end, and averages the remainder.
        proportion: Proportion dropped from each end in ``[0, 0.5)``. When it
            does not resolve to a whole number of members, the number of
            dropped members is rounded down, so small ensembles keep all
            members.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        method: Literal["mean", "trimmed_mean"] = "mean",
        proportion: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= proportion < 0.5:
            raise ValueError("proportion must satisfy 0 <= proportion < 0.5.")
        self.method = method
        self.proportion = proportion

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if ensemble_table.num_members == 0:
            raise ValueError("Expected at least one ensemble member.")

        reference = ensemble_table.member(0)
        extra = reference.active_stypes - self.handles_stypes
        if extra:
            found = ", ".join(sorted(extra))
            raise ValueError(
                f"Expected a numerical-only output table (also found {found})."
            )
        reference_columns = reference.columns[Stype.numerical]

        if self.method == "trimmed_mean":
            # Sorting depends on how often each table appears as a member, so
            # shared tables are materialized once per member.
            groups = [
                ensemble_table.expanded_group(group_id)
                for group_id in range(ensemble_table.num_groups)
            ]
        else:
            groups = list(ensemble_table)

        numerical_groups: list[Tensor] = []
        for group in groups:
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
            numerical_groups.append(numerical)

        if self.method == "mean":
            counts_by_location = Counter(ensemble_table._locations)
            total = None
            for group_id, numerical in enumerate(numerical_groups):
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
            members = torch.cat(numerical_groups, dim=0)  # [E, ..., R, O]
            cut = int(self.proportion * members.size(0))
            ordered = members.sort(dim=0).values
            reduced = ordered.narrow(
                dim=0,
                start=cut,
                length=members.size(0) - 2 * cut,
            ).mean(dim=0)

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
            cut = int(self.proportion * numerical.size(0))
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
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"method={self.method!r}, proportion={self.proportion})"
        )


class ReduceQuantiles(Processor):
    r"""Reduce quantile predictions to their mean point prediction.

    The ``O`` numerical columns of an output table with shape
    ``[..., R, O]`` hold the predicted quantiles of every row. They are
    replaced by a single column ``"mean"`` with shape ``[..., R, 1]``, which
    approximates the expected value for evenly spaced quantiles.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        mean = TableTensor(
            columns={Stype.numerical: ("mean",)},
            numerical=table.numerical.mean(dim=-1, keepdim=True),
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.numerical), mean], dim=-1),
        )
