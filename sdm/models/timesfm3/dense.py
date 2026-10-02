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

from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, RMSNorm


class ResidualBlock(torch.nn.Module):
    """Apply the TimesFM-3 two-layer residual block.

    Args:
        input_dims: Size of the last input dimension.
        hidden_dims: Hidden-layer width.
        output_dims: Output width.
        use_bias: Whether linear layers use bias parameters.
        identity_skip: Whether to use an identity residual connection.
        prenorm: Normalization applied before the hidden layer.
        device: Device on which to create parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        use_bias: bool,
        identity_skip: bool = False,
        prenorm: Literal["rms", "none"] = "none",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if identity_skip and output_dims != input_dims:
            raise ValueError(
                "identity_skip requires output_dims to match input_dims, got "
                f"{output_dims} and {input_dims}"
            )
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.hidden_layer = Linear(
            in_features=input_dims,
            out_features=hidden_dims,
            bias=use_bias,
            **factory_kwargs,
        )
        self.output_layer = Linear(
            in_features=hidden_dims,
            out_features=output_dims,
            bias=use_bias,
            **factory_kwargs,
        )
        if identity_skip:
            self.residual_layer: Linear | None = None
        else:
            self.residual_layer = Linear(
                in_features=input_dims,
                out_features=output_dims,
                bias=use_bias,
                **factory_kwargs,
            )

        self.activation = ReLU()
        if prenorm == "rms":
            self.pre_norm: RMSNorm | None = RMSNorm(
                input_dims,
                **factory_kwargs,
            )
        elif prenorm == "none":
            self.pre_norm = None
        else:
            raise AssertionError(f"Unhandled pre-normalization: {prenorm}")

    def forward(self, x: Tensor) -> Tensor:
        """Transform input values and add the residual connection.

        Args:
            x: Input values with shape ``[..., D]``, where ``D`` is the input
                dimension.

        Returns:
            Transformed values with shape ``[..., O]``, where ``O`` is the
            configured output dimension.
        """
        hidden_input = self.pre_norm(x) if self.pre_norm is not None else x
        hidden_output = self.activation(self.hidden_layer(hidden_input))
        output = self.output_layer(hidden_output)
        if self.residual_layer is not None:
            return output + self.residual_layer(x)
        return output + x
