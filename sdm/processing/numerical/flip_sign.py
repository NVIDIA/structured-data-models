# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleData, EnsembleTable, EnsembleTensor, Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class FlipSign(Processor, InvertibleMixin):
    """Randomly negate numerical columns.

    Args:
        probability: Probability of negating each numerical column.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _signs: EnsembleTensor[Tensor]

    def __init__(
        self,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.probability = probability
        self.register_buffer("_signs", None)

    def _fit(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        signs = EnsembleData.execute(
            self._draw_signs,
            table.numerical,
            generator=generator,
            randomness="different",
        )
        self.register_buffer("_signs", signs)

    def _draw_signs(
        self, numerical: Tensor, *, generator: torch.Generator | None
    ) -> Tensor:
        signs = numerical.new_empty(
            (*numerical.shape[:-2], 1, numerical.size(-1), 1)
        )
        signs.bernoulli_(self.probability, generator=generator)
        return signs.mul_(-2).add_(1)

    def _transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        numerical = EnsembleData.execute(
            self._multiply, table.numerical, self._signs, batched=True
        )
        return table.replace_blocks(numerical=numerical)

    @staticmethod
    def _multiply(numerical: Tensor, signs: Tensor) -> Tensor:
        if signs.size(-2) == 0:
            return numerical
        numerical = numerical.unflatten(-1, (signs.size(-2), -1))
        return (numerical * signs).flatten(-2)

    def _inverse_transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        return self._transform(table)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(probability={self.probability})"
        )
