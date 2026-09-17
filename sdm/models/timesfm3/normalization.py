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

"""Normalization layers for TimesFM-3."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041


class PerDimScale(torch.nn.Module):
    r"""Apply learnable per-dimension scaling.

    Args:
        num_dims: The number of dimensions.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        num_dims: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_dims = num_dims
        self.per_dim_scale = torch.nn.Parameter(
            torch.zeros(num_dims, **factory_kwargs)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Apply per-dimension scaling to the input."""
        return (
            tensor
            * _RECIPROCAL_OF_SOFTPLUS_0
            / math.sqrt(self.num_dims)
            * F.softplus(self.per_dim_scale)
        )
