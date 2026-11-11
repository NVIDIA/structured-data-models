# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleData, EnsembleTable, Stype, TableTensor
from sdm.processing import Processor


class ClipQuantiles(Processor):
    """Clamp each numerical column to its fitted quantile bounds.

    Values outside the fitted bounds are discarded, so this processor is not
    invertible. Bounds are fitted independently for each column.

    Args:
        q_low: Lower quantile in ``[0, 1]`` used as the per-column lower bound.
        q_high: Upper quantile in ``[0, 1]`` used as the per-column upper
            bound. Must satisfy ``0 <= q_low <= q_high <= 1``.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    lower_bound: Tensor
    upper_bound: Tensor

    def __init__(
        self,
        *,
        q_low: float = 0.0,
        q_high: float = 1.0,
    ) -> None:
        super().__init__()
        if q_low > q_high:
            raise ValueError("q_low and q_high must satisfy q_low <= q_high.")
        self.q_low = q_low
        self.q_high = q_high
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        q_low, q_high = EnsembleData.execute(
            self._bounds, table.numerical, batched=True
        )
        self._set_buffer("lower_bound", q_low)
        self._set_buffer("upper_bound", q_high)

    def _bounds(self, numerical: Tensor) -> tuple[Tensor, Tensor]:
        if numerical.size(-1) == 0:
            empty = numerical.new_empty((*numerical.shape[:-2], 1, 0))
            return empty, empty
        quantiles = numerical.new_tensor([self.q_low, self.q_high])
        low, high = numerical.quantile(quantiles, dim=-2, keepdim=True).unbind(
            0
        )
        return low, high

    def _transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        """Clamp ``table`` to the fitted lower and upper bounds."""
        numerical = table.numerical.clamp(
            min=self.lower_bound,
            max=self.upper_bound,
        )
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self.q_low != 0.0:
            arguments.append(f"q_low={self.q_low!r}")
        if self.q_high != 1.0:
            arguments.append(f"q_high={self.q_high!r}")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
