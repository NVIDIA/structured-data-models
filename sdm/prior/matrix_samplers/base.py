# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class MatrixSampler(ABC):
    """Base class for sampling random matrices."""

    @abstractmethod
    def sample(
        self,
        size: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample random matrices.

        Args:
            size: Output shape ``[..., M, N]`` with ``M`` rows and ``N``
                columns per matrix.
            dtype: Output dtype.
            generator: Pseudorandom generator.

        Returns:
            Random matrices with shape ``size``.
        """
        ...
