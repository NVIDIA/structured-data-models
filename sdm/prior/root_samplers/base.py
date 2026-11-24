# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class RootSampler(ABC):
    """Base class for sampling values at nodes without parents."""

    @abstractmethod
    def sample(
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample root values with shape ``[R, H]`` for rows and channels.

        Args:
            num_rows: Number of rows.
            channels: Number of output channels.
            generator: Pseudorandom generator.
        """
        ...
