# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class FlipSign(Processor, InvertibleMixin):
    """Randomly negate numerical feature columns.

    Args:
        probability: Probability of negating a numerical column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, probability: float = 0.5) -> None:
        super().__init__()
        self.probability = probability
        self.register_buffer("sign", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        shape = (*table.numerical.size()[:-2], 1, table.numerical.size(-1))
        self.sign = table.numerical.new_empty(shape)
        self.sign.bernoulli_(self.probability, generator=generator)
        self.sign.mul_(-2).add_(1)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical * self.sign)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return self._transform(table)
