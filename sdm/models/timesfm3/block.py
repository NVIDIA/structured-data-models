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

# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.mlp = Sequential(
            Linear(in_channels, out_channels, bias=bias, **factory_kwargs),
            ReLU(),
            Linear(out_channels, out_channels, bias=bias, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, bias, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.res(x)
