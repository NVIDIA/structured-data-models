# Copyright (c) 2024 Auton Lab, Carnegie Mellon University
# Licensed under the MIT License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forecasting output heads."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear


class ForecastingHead(torch.nn.Module):
    """Project encoded patches into a fixed forecast horizon.

    Args:
        input_channels: Number of flattened patch channels.
        prediction_length: Number of future time steps to predict.
        dropout: Dropout probability applied to predictions.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        input_channels: int,
        prediction_length: int,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.linear = Linear(
            input_channels,
            prediction_length,
            **factory_kwargs,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Predict a future horizon from encoded patches.

        Args:
            x: Encoded patches with shape ``[..., P, C]``.

        Returns:
            Forecast tensor with shape ``[..., H]``.
        """
        x = x.flatten(start_dim=-2)
        return self.dropout(self.linear(x))
