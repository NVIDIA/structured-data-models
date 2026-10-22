# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import Linear


class ResidualMLP(torch.nn.Module):
    """A two-layer SiLU MLP with a learned linear residual projection.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels.
        bias: Whether the projections learn additive biases.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        kwargs = {"bias": bias, "device": device, "dtype": dtype}
        self.hidden_lin = Linear(in_channels, hidden_channels, **kwargs)
        self.out_lin = Linear(hidden_channels, out_channels, **kwargs)
        self.residual_lin = Linear(in_channels, out_channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Transform the last dimension of ``x``.

        Args:
            x: Input with shape ``[..., C]``, where ``C`` is input channels.
        """
        return self.out_lin(
            torch.nn.functional.silu(self.hidden_lin(x))
        ) + self.residual_lin(x)
