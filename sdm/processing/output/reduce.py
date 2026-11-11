# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from sdm import EnsembleTable, Stype, TableLike, TableTensor
from sdm.processing import Processor


class AverageEstimators(Processor):
    """Reduce the leading ensemble dimension of model outputs.

    Inputs must be a numerical output table with shape ``[E, ...]``, where
    ``E`` is the non-empty leading ensemble dimension.

    Args:
        trim_fraction: Fraction of estimators discarded from each tail after
            sorting each output coordinate, in ``[0, 0.5)``. This can make the
            aggregation less sensitive to outlying estimator predictions.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def __init__(
        self,
        *,
        trim_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("'trim_fraction' must be in [0, 0.5)")
        self.trim_fraction = trim_fraction

    def _transform(self, table: TableLike) -> TableLike:
        ensemble = isinstance(table, EnsembleTable)
        if ensemble:
            table = table.as_tensor()
        elif table.dim() < 3:
            raise ValueError("Expected a leading ensemble dimension")
        numerical = table.numerical
        cut = int(self.trim_fraction * numerical.size(0))
        if cut:
            numerical = numerical.sort(dim=0).values
            numerical = numerical[cut : numerical.size(0) - cut]
        out = TableTensor(
            columns={Stype.numerical: table.numerical_columns},
            numerical=numerical.mean(dim=0),
        )
        return (
            EnsembleTable.from_table(out, num_members=1) if ensemble else out
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.trim_fraction == 0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"trim_fraction={self.trim_fraction})"
        )
