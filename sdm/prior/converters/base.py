# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NumericalConverter(ABC):
    """Base class for converting latent channels to numerical columns."""

    def sample_num_channels(
        self,
        num_columns: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample how many latent channels each numerical column reads.

        Args:
            num_columns: Number of columns.
            generator: Pseudorandom generator.

        Returns:
            Integer channel counts with shape ``[C]`` for ``C`` columns.
            The default is one channel per column.
        """
        return torch.ones(num_columns, dtype=torch.long, device="cpu")

    @abstractmethod
    def forward(
        self,
        slices: Tensor,  # [num_cols, channels, num_rows]
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return numerical column values and slices passed to child nodes.

        Args:
            slices: Node-channel slices per column with shape
                ``[C, H, R]`` for columns, channels, and rows. The channel
                count ``H`` may be ragged across columns.
            generator: Pseudorandom generator.

        Returns:
            Column values with shape ``[C, R]`` and updated node-channel
            slices with the input shape for use by child nodes.
        """
        ...


class CategoricalConverter(ABC):
    """Base class for converting latent channels to categorical columns."""

    @abstractmethod
    def forward(
        self,
        slices: Tensor,  # [num_cols, num_categories, num_rows]
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return category indices and slices passed to child nodes.

        Args:
            slices: Node-channel slices per column with shape
                ``[C, K, R]`` for columns, categories, and rows. The category
                count ``K`` may be ragged across columns.
            generator: Pseudorandom generator.

        Returns:
            Category indices with shape ``[C, R]`` and updated node-channel
            slices with the input shape for use by child nodes.
        """
        ...
