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

"""Learned attention transforms with float32 arithmetic."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class Float32RMSNorm(torch.nn.Module):
    r"""Root mean square normalization evaluated in float32.

    The input and learned weight are converted to float32 for the complete
    normalization computation. The result is converted back to the input
    dtype.

    Args:
        channels: The number of channels in the final input dimension.
        eps: The value added to the mean square before reciprocal square root.
        device: The device.
        dtype: The parameter dtype.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(
            data=torch.ones(channels, device=device, dtype=dtype)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Normalize the final dimension of ``tensor``."""
        dtype = tensor.dtype
        tensor = tensor.float()
        variance = tensor.square().mean(dim=-1, keepdim=True)
        tensor = tensor * (variance + self.eps).rsqrt()
        return (tensor * self.weight.float()).to(dtype)


class SoftplusScale(torch.nn.Module):
    r"""Scale the final dimension by learned positive factors.

    The factors are ``multiplier * softplus(weight)`` and are evaluated in
    float32 before being converted to the input dtype.

    Args:
        channels: The number of channels in the final input dimension.
        multiplier: The fixed multiplier applied to every learned factor.
        parameter_init: The initial value of every learned parameter.
        device: The device.
        dtype: The parameter dtype.
    """

    def __init__(
        self,
        channels: int,
        multiplier: float = 1.0,
        parameter_init: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(
                f"'multiplier' must be positive and finite (got {multiplier})"
            )
        self.multiplier = multiplier
        self.weight = Parameter(
            data=torch.full(
                size=(channels,),
                fill_value=parameter_init,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Scale the final dimension of ``tensor``."""
        scale = self.multiplier * F.softplus(self.weight.float())
        return tensor * scale.to(tensor.dtype)
