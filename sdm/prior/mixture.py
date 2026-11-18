# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence
from typing import Generic, TypeVar, cast

import torch
from torch import Tensor

from sdm.prior.correlated import _CONTEXT

T = TypeVar("T")


class Mixture(Generic[T]):
    """Select items using fixed or shared random weights.

    Args:
        items: Objects or ``(object, weight)`` pairs. Bare objects have
            weight one. Weights must be nonnegative, with at least one
            positive value.
        correlated: Share random selection probabilities within an active
            :class:`~sdm.prior.CorrelatedContext`.
        key: Key for shared probabilities. Defaults to this mixture's
            identity.
    """

    def __init__(
        self,
        items: Sequence[T | tuple[T, float]],
        correlated: bool = False,
        key: Hashable | None = None,
    ) -> None:
        self.items: list[T] = []
        weights: list[float] = []
        for item in items:
            if isinstance(item, tuple):
                obj, w = cast(tuple[T, float], item)
            else:
                obj, w = item, 1.0
            self.items.append(obj)
            weights.append(w)
        if any(weight < 0 for weight in weights) or not any(
            weight > 0 for weight in weights
        ):
            raise ValueError(
                "Weights must be nonnegative with at least one positive value."
            )
        self.weight = torch.tensor(weights, dtype=torch.float, device="cpu")
        self._is_uniform_weight = len(set(weights)) == 1
        self.correlated = correlated
        self.key = key

    def _draw(
        self,
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        weight = self.weight
        if self.correlated and (ctx := _CONTEXT.get()) is not None:
            generator = ctx._resolve_generator(generator)
            weight = ctx.mixture_weight(
                self.key if self.key is not None else id(self),
                weight,
            )

        if replacement and self._is_uniform_weight and weight is self.weight:
            return torch.randint(
                low=0,
                high=len(self.items),
                size=(num_samples,),
                device="cpu",
                generator=generator,
            )

        return torch.multinomial(
            weight,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )

    def pick(self, generator: torch.Generator | None = None) -> T:
        """Select one item.

        Args:
            generator: Pseudorandom generator. In a correlated context, omit it
                or pass the context's generator.
        """
        return self.items[int(self._draw(1, generator=generator))]
