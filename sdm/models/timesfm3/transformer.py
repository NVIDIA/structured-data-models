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


def make_attn_mask(patch_mask: Tensor, causal: bool = True) -> Tensor:
    """Create an attention mask in which ``True`` permits attention.

    Args:
        patch_mask: Masked patches with shape ``[B, N]``.
        causal: Whether queries may attend only to preceding positions.

    Returns:
        Boolean mask with shape ``[B, 1, N, N]`` when causal and broadcastable
        shape ``[B, 1, 1, N]`` otherwise.
    """
    mask = ~patch_mask[:, None, None, :]
    if not causal:
        return mask
    causal_mask = torch.ones(
        patch_mask.size(1),
        patch_mask.size(1),
        dtype=torch.bool,
        device=patch_mask.device,
    ).tril()
    return causal_mask[None, None] & mask
