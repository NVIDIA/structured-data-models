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

"""Learned softplus scaling."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


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
