# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ReLU, RMSNorm


class ResidualBlock(torch.nn.Module):
    """Apply the TimesFM-3 two-layer residual block.

    Args:
        in_channels: Size of the last input dimension.
        out_channels: Output width and hidden-layer width.
        bias: Whether linear layers have bias parameters.
        identity_residual: Add the input directly to the output. Requires
            ``in_channels == out_channels``. Otherwise, project the input.
        prenorm: Apply RMS normalization before the hidden layer.
        device: Device on which to create parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        identity_residual: bool = False,
        prenorm: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if identity_residual and out_channels != in_channels:
            raise ValueError(
                "identity_residual requires matching channels, got "
                f"{out_channels} and {in_channels}"
            )
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.hidden_layer = Linear(
            in_channels, out_channels, bias=bias, **factory_kwargs
        )
        self.output_layer = Linear(
            out_channels, out_channels, bias=bias, **factory_kwargs
        )
        self.residual_layer = (
            Linear(in_channels, out_channels, bias=bias, **factory_kwargs)
            if not identity_residual
            else Identity()
        )
        self.activation = ReLU()
        self.pre_norm = (
            RMSNorm(in_channels, **factory_kwargs) if prenorm else Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """The forward pass.

        Args:
            x: Input tensor with shape ``[..., Ci]``, where ``Ci`` is the
                number of input channels.

        Returns:
            Tensor with shape ``[..., Co]``, where ``Co`` is the
                number of output channels.
        """
        hidden = self.activation(self.hidden_layer(self.pre_norm(x)))
        return self.output_layer(hidden) + self.residual_layer(x)
