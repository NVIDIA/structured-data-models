# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Correlated, LogUniform
from sdm.prior.activations import Activation
from sdm.prior.util import standardize


class StandardizedActivation(Activation):
    """Apply an activation at a random center and scale.

    Args:
        activation: Activation applied to the centered and scaled values.
        scale: Scale after random centering. Defaults to
            ``Correlated(LogUniform(1.0, 10.0))``.
    """

    def __init__(
        self,
        activation: Activation,
        *,
        scale: HP[float] | float | None = None,
    ) -> None:
        self.activation = activation
        self.scale = HP.from_value(
            Correlated(
                hp=LogUniform(1.0, 10.0),
            )
            if scale is None
            else scale
        )

    def forward(  # noqa: D102
        self,
        x: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        num_rows, channels = x.size()

        x, _ = standardize(x, dim=0)

        scale = self.scale.sample_batch(size=(), generator=generator).to(
            x.dtype
        )
        rows = torch.randint(
            high=num_rows,
            size=(channels,),
            device=x.device,
            generator=generator,
        )
        center = x[rows, torch.arange(channels, device=x.device)]
        x = scale * (x - center)

        x = self.activation.forward(x, generator=generator)

        x, _ = standardize(x, dim=0)
        return x
