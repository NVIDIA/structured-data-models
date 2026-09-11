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
        method: Reduction applied across ensemble members. Currently only
            ``"mean"`` is supported.
    """

    handles_stypes = frozenset({Stype.numerical})
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
        extra = reference.active_stypes - self.handles_stypes
        if extra:
            found = ", ".join(sorted(extra))
            raise ValueError(
                f"Expected a numerical-only output table (also found {found})."
            )
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
            if numerical.shape[1:] != reference.numerical.shape:
                raise ValueError(
                    "Expected ensemble members to have the same shape."
                )

            multiplicities = [
                counts_by_location[(group_id, position)]
                for position in range(group.size(0))
            ]
            contiguous_members = numerical[:1].is_contiguous() and (
                numerical.size(0) <= 1 or numerical.stride(0) != 0
            )
            autocast = torch.is_autocast_enabled(numerical.device.type)
            if (
                columns == reference_columns
                and contiguous_members
                and not autocast
                and (
                    total is None
                    or (
                        total.dtype == numerical.dtype
                        and numerical.dtype
                        not in (torch.float16, torch.bfloat16)
                    )
                )
            ):
                # Accumulate the weighted group directly into the only output.
                counts = numerical.new_tensor(multiplicities)
                matrix = numerical.flatten(1)
                if total is None:
                    total = torch.mm(counts.unsqueeze(0), matrix).view(
                        numerical.shape[1:]
                    )
                else:
                    target = total.view(1, -1)
                    torch.addmm(
                        target, counts.unsqueeze(0), matrix, out=target
                    )
                continue

            if (
                contiguous_members
                or autocast
                or numerical.dtype not in (torch.float32, torch.float64)
            ):
                # Keep GEMM's accumulation precision for low-precision inputs.
                counts = numerical.new_tensor(multiplicities)
                partial = torch.tensordot(
                    a=counts,
                    b=numerical,
                    dims=([0], [0]),
                )
            else:
                # Flattening a strided member would copy the entire group.
                accumulate = (
                    columns == reference_columns
                    and total is not None
                    and total.dtype == numerical.dtype
                )
                if accumulate:
                    assert total is not None
                    partial = total
                    partial.add_(numerical[0], alpha=multiplicities[0])
                else:
                    partial = numerical.new_empty(numerical.shape[1:])
                    torch.mul(numerical[0], multiplicities[0], out=partial)
                for position, count in enumerate(multiplicities[1:], start=1):
                    partial.add_(numerical[position], alpha=count)
                if accumulate:
                    del partial
                    continue

            if columns == reference_columns:
                total = partial if total is None else total.add_(partial)
            else:
                # Reorder the reduced columns, not every input estimator.
                column_to_index = {
                    column: index
                    for index, column in enumerate(reference_columns)
                }
                indices = torch.tensor(
                    [column_to_index[column] for column in columns],
                    device=numerical.device,
                    dtype=torch.int64,
                )
                if total is None:
                    total = torch.zeros_like(partial)
                if total.dtype == partial.dtype:
                    total.index_add_(-1, indices, partial)
                else:
                    total.add_(partial.index_select(-1, indices.argsort()))
            del partial

        assert total is not None
        output = reference.replace_blocks(
            numerical=total.div_(ensemble_table.num_members)
        )
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

        if self.method == "mean":
            numerical = table.numerical.mean(dim=0)
        else:
            raise ValueError("method must be 'mean'")
        return table.__class__(
            columns={Stype.numerical: table.columns[Stype.numerical]},
            numerical=numerical,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
