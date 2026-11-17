# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Mixture
from sdm.prior.activations import Activation
from sdm.prior.functions import RandomFunction, RandomLinear


class RandomNN(RandomFunction):
    """Apply a random neural network.

    Args:
        num_layers: Number of linear layers.
        hidden_channels: Hidden width.
        activation: Activation applied between layers and, by random
            choice, before the first and after the last layer.
        linear: Random linear function of each layer. Defaults to
            :class:`RandomLinear`.
    """

    def __init__(
        self,
        num_layers: HP[int] | int,
        hidden_channels: HP[int] | int,
        activation: Activation,
        linear: RandomLinear | None = None,
    ) -> None:
        self.num_layers = HP.from_value(num_layers)
        self.hidden_channels = HP.from_value(hidden_channels)
        self.activation = activation
        self.linear = linear or RandomLinear()
        self.start_activation = Mixture(
            items=[False, True],
            correlated=True,
        )
        self.end_activation = Mixture(
            items=[False, True],
            correlated=True,
        )

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        num_layers = self.num_layers.sample(generator=generator)
        hidden_channels = self.hidden_channels.sample(generator=generator)

        if self.start_activation.pick(generator=generator):
            x = self.activation.forward(
                x=x,
                generator=generator,
            )

        for i in range(num_layers):
            channels = out_channels if i == num_layers - 1 else hidden_channels
            x = self.linear.forward(
                x=x, out_channels=channels, generator=generator
            )
            if i < num_layers - 1:
                x = self.activation.forward(
                    x=x,
                    generator=generator,
                )

        if self.end_activation.pick(generator=generator):
            x = self.activation.forward(
                x=x,
                generator=generator,
            )

        return x
