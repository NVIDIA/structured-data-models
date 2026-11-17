# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torch import Tensor

from sdm.prior.functions import ParentMerger, RandomFunction


class ConcatParentMerger(ParentMerger):
    """Concatenate parent values and apply ``random_fn``."""

    def forward(  # noqa: D102
        self,
        parent_values: Sequence[Tensor],
        random_fn: RandomFunction,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return random_fn.forward(
            x=torch.cat(tuple(parent_values), dim=-1),
            out_channels=out_channels,
            generator=generator,
        )
