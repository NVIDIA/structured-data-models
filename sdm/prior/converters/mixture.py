# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Generic, TypeVar, cast

import torch

from sdm.prior import Mixture
from sdm.prior.converters import (
    CategoricalConverter,
    NumericalConverter,
)

T = TypeVar("T", NumericalConverter, CategoricalConverter)


class MixtureConverter(Mixture["T | MixtureConverter[T]"], Generic[T]):
    """Pick one converter per column from a weighted pool.

    Args:
        converters: Converters of one kind, nested mixtures of that kind,
            or weighted pairs.
        correlated: Share selection probabilities within an active
            :class:`~sdm.prior.CorrelatedContext`.
        key: Key for shared probabilities. Defaults to this mixture's
            identity.
    """

    def __init__(
        self,
        converters: Sequence[
            T | MixtureConverter[T] | tuple[T | MixtureConverter[T], float]
        ],
        correlated: bool = True,
        key: Hashable | None = None,
    ) -> None:
        super().__init__(converters, correlated, key)

    def sample(
        self,
        num_columns: int,
        generator: torch.Generator | None = None,
    ) -> list[T]:
        """Select a concrete converter for each column, in column order.

        Args:
            num_columns: Number of columns.
            generator: Pseudorandom generator.
        """
        if num_columns == 0:
            return []
        choice = self._draw(num_samples=num_columns, generator=generator)
        converters = [self.items[c] for c in choice.tolist()]
        for c in choice.unique().tolist():
            item = self.items[c]
            if isinstance(item, MixtureConverter):
                indices = (choice == c).nonzero().flatten().tolist()
                selected = item.sample(len(indices), generator=generator)
                for index, converter in zip(indices, selected):
                    converters[index] = converter
        return cast(list[T], converters)
