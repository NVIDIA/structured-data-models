from collections import Counter
from typing import Literal

import torch

from sdm.processing.ensemble import EnsembleProcessor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


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
        method: Reduction applied across ensemble members. Currently only
            ``"mean"`` is supported.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        method: Literal["mean"] = "mean",
    ) -> None:
        super().__init__()
        # TODO: Support `method="median"` when required by a model recipe.
        if method != "mean":
            raise ValueError("method must be 'mean'")
        self.method = method

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if ensemble_table.num_members == 0:
            raise ValueError("Expected at least one ensemble member.")

        reference = ensemble_table.table(0)
        reference_columns = reference.columns[Stype.numerical]
        counts_by_location = Counter(
            ensemble_table._locations[member_id]
            for member_id in range(ensemble_table.num_members)
        )

        total = None
        for group_id, group in enumerate(ensemble_table):
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

            counts = numerical.new_tensor(
                [
                    counts_by_location[(group_id, position)]
                    for position in range(group.size(0))
                ]
            )
            partial = torch.tensordot(
                a=counts,
                b=numerical,
                dims=([0], [0]),
            )
            total = partial if total is None else total + partial

        assert total is not None
        output = reference.replace_blocks(
            numerical=total / ensemble_table.num_members
        )
        return EnsembleTable(output, num_members=1)

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.dim() < 3:
            raise ValueError(
                "Expected a leading ensemble dimension in an output table "
                f"with at least 3 dimensions (got {table.dim()}D)."
            )
        if table.size(0) == 0:
            raise ValueError("Expected at least one ensemble member.")

        if self.method == "mean":
            numerical = table.numerical.mean(dim=0)
        else:
            raise ValueError("method must be 'mean'")
        return table.__class__(
            columns={Stype.numerical.value: table.columns[Stype.numerical]},
            numerical=numerical,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
