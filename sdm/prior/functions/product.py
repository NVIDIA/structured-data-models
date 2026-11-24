# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.functions import RandomFunction
from sdm.prior.functions.mixture import MixtureFunction


class RandomProduct(RandomFunction):
    """Multiply the outputs of two sampled random functions.

    Args:
        functions: Mixture from which both functions are selected.
    """

    def __init__(self, functions: MixtureFunction) -> None:
        self.functions = functions

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        f = self.functions.pick(generator=generator)
        g = self.functions.pick(generator=generator)

        f_out = f.forward(x, out_channels, generator=generator)
        g_out = g.forward(x, out_channels, generator=generator)
        return f_out * g_out
