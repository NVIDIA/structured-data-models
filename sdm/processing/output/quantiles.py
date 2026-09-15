# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ReduceQuantiles(Processor):
    r"""Reduce quantile predictions to their mean point prediction.

    Input must be a numerical output table with shape ``[..., R, O]`` whose
    ``O`` output columns hold the predicted quantiles of every row. The result
    has shape ``[..., R, 1]`` with the single output column ``"mean"`` holding
    the mean over the quantiles, and retains the input device and floating
    dtype.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        extra = table.active_stypes - self.handles_stypes
        if extra:
            found = ", ".join(sorted(extra))
            raise ValueError(
                f"Expected a numerical-only output table (also found {found})."
            )
        return table.__class__(
            columns={Stype.numerical: ("mean",)},
            numerical=table.numerical.mean(dim=-1, keepdim=True),
        )
