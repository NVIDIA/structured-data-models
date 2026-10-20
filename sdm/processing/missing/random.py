# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing._random import SeedMixin, uniform
from sdm.processing.missing._cells import mask_cells


class MaskRandom(SeedMixin, Processor):
    """Mask numerical and categorical cells completely at random.

    Each table draws a missing-cell probability when the processor is fitted.
    Transform masks each cell independently with this probability. Apply the
    processor to all rows in one call.

    Args:
        rate: Range from which to draw each table's missing-cell probability.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True

    _missing_rate: Tensor

    def __init__(self, *, rate: tuple[float, float] = (0.01, 0.08)) -> None:
        super().__init__()
        low, high = rate
        if not 0 <= low <= high <= 1:
            raise ValueError("rate must satisfy 0 <= low <= high <= 1.")
        self.rate = rate
        self.register_buffer("_missing_rate", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        self._set_buffer(
            "_missing_rate",
            uniform(
                low=self.rate[0],
                high=self.rate[1],
                size=table.size()[:-2],
                device=table.device,
                dtype=table.numerical.dtype,
                generator=generator,
            ),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        num_columns = table.numerical.size(-1) + table.categorical.size(-1)
        u = torch.rand(
            (*table.size()[:-1], num_columns),
            device="cpu",
            generator=self._transform_generator(),
        )
        missing = u.to(table.device) < self._missing_rate[..., None, None]
        return mask_cells(table, missing)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(rate={self.rate})"
