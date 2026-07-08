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
#
# Modified for the structured-data-models package.

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList

from sdm.models.tabfm._utils import _get_activation


class MLP(torch.nn.Module):
    """Apply a sequence of linear layers with activations between them.

    The output layer has no activation. Parameter names follow the released
    TabFM PyTorch model so its target encoders and decoders load without
    remapping.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of every hidden layer.
        out_channels: Number of output channels.
        activation: Activation applied after each hidden layer.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        activation: str = "gelu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if any(channels <= 0 for channels in hidden_channels):
            raise ValueError("hidden_channels must contain positive values")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.act = _get_activation(activation)
        dimensions = [in_channels, *hidden_channels, out_channels]
        self.layers = ModuleList(
            Linear(
                dimensions[index],
                dimensions[index + 1],
                **factory_kwargs,
            )
            for index in range(len(dimensions) - 1)
        )

    def forward(self, input: Tensor) -> Tensor:
        """Transform ``input`` along its final dimension.

        Args:
            input: Input tensor with shape ``[..., C_in]``.

        Returns:
            Tensor with shape ``[..., C_out]``.
        """
        for index, layer in enumerate(self.layers):
            input = layer(input)
            if index < len(self.layers) - 1:
                input = self.act(input)
        return input
