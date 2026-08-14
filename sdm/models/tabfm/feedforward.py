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

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class _SwiGLUFeedForward(torch.nn.Module):
    """Apply TabFM's SwiGLU feed-forward projection."""

    def __init__(
        self,
        channels: int,
        feedforward_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.value_lin = Linear(
            in_features=channels,
            out_features=feedforward_channels,
            **factory_kwargs,
        )
        self.gate_lin = Linear(
            in_features=channels,
            out_features=feedforward_channels,
            **factory_kwargs,
        )
        self.out_lin = Linear(
            in_features=feedforward_channels,
            out_features=channels,
            **factory_kwargs,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Apply the projection."""
        value = self.value_lin(tensor)
        gate = F.silu(self.gate_lin(tensor))
        return self.out_lin(gate * value)


class _ChunkedFeedForward(torch.nn.Module):
    """Apply a tokenwise feed-forward module in bounded-size chunks."""

    def __init__(
        self,
        module: torch.nn.Module,
        chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_size = chunk_size

    def forward(self, tensor: Tensor) -> Tensor:
        """Apply the wrapped feed-forward module."""
        if self.chunk_size is None:
            return self.module(tensor)

        batch_shape = tensor.shape[:-1]
        tensor = tensor.reshape(-1, tensor.size(-1))
        if tensor.size(0) == 0:
            output = self.module(tensor)
        else:
            output = torch.cat(
                [
                    self.module(chunk)
                    for chunk in tensor.split(self.chunk_size)
                ],
                dim=0,
            )
        return output.reshape(*batch_shape, output.size(-1))
