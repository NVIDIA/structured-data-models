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
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList


def _gelu_tanh(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


class _MLP(torch.nn.Module):
    """Apply the TabFM v1.0.0 tanh-GELU multilayer perceptron.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output channels.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        channels = (in_channels, *hidden_channels, out_channels)
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.layers = ModuleList(
            [
                Linear(channels[index], channels[index + 1], **factory_kwargs)
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``[..., in_channels]`` into ``[..., out_channels]``."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = _gelu_tanh(x)
        return x
