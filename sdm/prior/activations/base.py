# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Activation(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Apply the activation to input values.

        Args:
            x: Input values.
            generator: Pseudorandom generator.
        """
        ...
