# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import PriorTransform


class Choice(PriorTransform):
    """Apply one randomly selected transform per dataset.

    Args:
        transforms: Candidate transforms.
        weights: Nonnegative relative selection weights with at least one
            positive value. If omitted, select uniformly.
    """

    def __init__(
        self,
        *transforms: PriorTransform,
        weights: Sequence[float] | None = None,
    ) -> None:
        if weights is not None and len(weights) != len(transforms):
            raise ValueError("Expected one weight per transform.")
        self.transforms = transforms
        self.weights = (
            torch.tensor(weights, dtype=torch.float32)
            if weights is not None
            else None
        )

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor] | None:
        if self.weights is None:
            index = torch.randint(
                len(self.transforms), (), device=x.device, generator=generator
            )
        else:
            index = torch.multinomial(
                self.weights.to(device=x.device), 1, generator=generator
            )
        return self.transforms[int(index)](x=x, y=y, generator=generator)
