# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch

from sdm import TableTensor


class PriorTransform(ABC):
    """Base class for transforming synthetic feature and target tables."""

    @abstractmethod
    def __call__(
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor] | None:
        """Transform a synthetic dataset.

        Args:
            x: Feature table of shape ``[R, C]`` for ``R`` rows and ``C``
                features.
            y: Target table of shape ``[R, 1]``.
            generator: Pseudorandom generator.

        Returns:
            Transformed feature and target tables, or ``None`` if the dataset
            is rejected.
        """
        ...
