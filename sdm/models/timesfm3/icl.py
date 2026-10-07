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

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList

from sdm.models.timesfm3.block import TimesFM3TransformerBlock


class ICLBlock(torch.nn.Module):
    """Apply the TimesFM-3 transformer stack from Jain and Sen (2026).

    Args:
        channels: Width of the patch embeddings.
        out_channels: Number of predicted values per patch.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layers = ModuleList(
            TimesFM3TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )
        self.head = Linear(
            channels,
            out_channels,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,  # [..., V, N, C]
        patch_mask: Tensor | None = None,  # [..., V, N]
    ) -> Tensor:  # [..., V, N, O]
        """Predict values from patch embeddings.

        Args:
            x: Patch embeddings with shape ``[..., V, N, C]``.
            patch_mask: Boolean mask with shape ``[..., V, N]``. ``True``
                excludes a patch from attention keys. ``None`` excludes none.

        Returns:
            Predictions with shape ``[..., V, N, O]``, where ``O`` is
            ``out_channels``.
        """
        for layer in self.layers:
            x = layer(x, patch_mask)
        return self.head(x)
