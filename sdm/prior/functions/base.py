# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor


class RandomFunction(ABC):
    """Base class for random functions on node values."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,  # [num_rows, in_channels]
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Apply a sampled function to every row of ``x``.

        Args:
            x: Input values with shape ``[R, H]`` for rows and channels.
            out_channels: Number of output channels.
            generator: Pseudorandom generator.

        Returns:
            Node values with shape ``[R, out_channels]``.
        """
        ...


class ParentMerger(ABC):
    """Base class for combining parent values into node values."""

    @abstractmethod
    def forward(
        self,
        parent_values: Sequence[Tensor],  # each: [num_rows, in_channels]
        random_fn: RandomFunction,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Compute node values from parent values with ``random_fn``.

        Args:
            parent_values: Parent values with shape ``[R, H]`` for rows and
                channels. Parents may have different numbers of channels.
            random_fn: Random function applied to the parent values.
            out_channels: Number of output channels.
            generator: Pseudorandom generator.

        Returns:
            Node values with shape ``[R, out_channels]``.
        """
        ...
