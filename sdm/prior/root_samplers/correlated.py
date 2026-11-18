# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.root_samplers import RootSampler
from sdm.prior.util import random_weight


class CorrelatedRootSampler(RootSampler):
    r"""Mix sampled root values to create correlated channels.

    Args:
        base: Sampler for the initial root values.
    """

    def __init__(self, base: RootSampler) -> None:
        self.base = base

    def sample(  # noqa: D102
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        points = self.base.sample(
            num_rows=num_rows,
            channels=channels,
            generator=generator,
        )
        mix = torch.randn(
            size=(1, channels, channels),
            dtype=points.dtype,
            device=points.device,
            generator=generator,
        )
        importance = random_weight(
            size=(1, channels),
            dtype=points.dtype,
            generator=generator,
        )
        mix = mix * importance.unsqueeze(1)
        return torch.bmm(points.unsqueeze(0), mix.transpose(1, 2)).squeeze(0)
