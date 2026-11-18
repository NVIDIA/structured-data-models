# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.prior.root_samplers import RootSampler


class BallRootSampler(RootSampler):
    r"""Sample root values uniformly within the unit ball."""

    def sample(  # noqa: D102
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        direction = torch.randn(
            size=(num_rows, channels),
            device="cpu",
            generator=generator,
        )
        direction = F.normalize(direction, dim=-1)
        radius = torch.rand(
            size=(num_rows, 1),
            device="cpu",
            generator=generator,
        ).pow(1.0 / channels)
        return radius * direction
