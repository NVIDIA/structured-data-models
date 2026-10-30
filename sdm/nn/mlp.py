# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, Sequential


class ResidualMLP(torch.nn.Module):
    """A two-layer MLP with a learned linear residual projection.

    Args:
        in_channels: The number of input channels.
        hidden_channels: The number of hidden channels.
        out_channels: The number of output channels.
        activation: The activation within the MLP.
        bias: Whether to learn additive biases.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: torch.nn.Module,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {
            "bias": bias,
            "device": device,
            "dtype": dtype,
        }
        self.mlp = Sequential(
            Linear(in_channels, hidden_channels, **factory_kwargs),
            activation,
            Linear(hidden_channels, out_channels, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            x: The tensor.
        """
        return self.mlp(x) + self.res(x)
