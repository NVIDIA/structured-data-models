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

from sdm.models.timesfm3.configs import ResidualBlockConfig
from sdm.models.timesfm3.util import get_activation_fn


class ResidualBlock(torch.nn.Module):
    """Apply the TimesFM-3 two-layer residual block.

    The input dimension is initialized by :meth:`set_input_dims` or inferred
    from the first input. TimesFM-3 initializes it explicitly during model
    construction, before loading checkpoint weights.

    Args:
        config: Residual block configuration.
        device: Device on which to create parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        config: ResidualBlockConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        # Register placeholders so device and dtype conversions applied before
        # the input dimension is known propagate to the final layers.
        self.hidden_layer = torch.nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.hidden_dims,
            bias=config.use_bias,
            **factory_kwargs,
        )
        self.output_layer = torch.nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
            **factory_kwargs,
        )
        if config.identity_skip:
            self.residual_layer: torch.nn.Linear | None = None
        else:
            self.residual_layer = torch.nn.Linear(
                in_features=config.hidden_dims,
                out_features=config.output_dims,
                bias=config.use_bias,
                **factory_kwargs,
            )

        self.activation = get_activation_fn(config.activation)
        if config.prenorm == "rms":
            self.pre_norm: torch.nn.RMSNorm | None = torch.nn.RMSNorm(
                config.hidden_dims,
                **factory_kwargs,
            )
        else:
            self.pre_norm = None
        self._input_dim_set = False

    def set_input_dims(self, input_dim: int) -> None:
        """Initialize layers for the input dimension.

        The first call replaces the placeholder layers while preserving their
        device and dtype. Subsequent calls have no effect.

        Args:
            input_dim: Size of the last input dimension.
        """
        if self._input_dim_set:
            return

        factory_kwargs: dict[str, Any] = {
            "device": self.hidden_layer.weight.device,
            "dtype": self.hidden_layer.weight.dtype,
        }
        self.hidden_layer = torch.nn.Linear(
            in_features=input_dim,
            out_features=self.config.hidden_dims,
            bias=self.config.use_bias,
            **factory_kwargs,
        )
        self.output_layer = torch.nn.Linear(
            in_features=self.config.hidden_dims,
            out_features=self.config.output_dims,
            bias=self.config.use_bias,
            **factory_kwargs,
        )
        if self.residual_layer is not None:
            self.residual_layer = torch.nn.Linear(
                in_features=input_dim,
                out_features=self.config.output_dims,
                bias=self.config.use_bias,
                **factory_kwargs,
            )
        if self.pre_norm is not None:
            self.pre_norm = torch.nn.RMSNorm(input_dim, **factory_kwargs)
        self._input_dim_set = True

    def forward(self, x: Tensor) -> Tensor:
        """Transform input values and add the residual connection.

        Args:
            x: Input values with shape ``[..., D]``, where ``D`` is the input
                dimension.

        Returns:
            Transformed values with shape ``[..., O]``, where ``O`` is the
            configured output dimension.
        """
        if not self._input_dim_set:
            self.set_input_dims(x.shape[-1])

        hidden_input = self.pre_norm(x) if self.pre_norm is not None else x
        hidden_output = self.activation(self.hidden_layer(hidden_input))
        output = self.output_layer(hidden_output)
        if self.residual_layer is not None:
            return output + self.residual_layer(x)
        return output + x
