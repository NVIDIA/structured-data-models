# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

def run(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    head_dim: int,
    eps: float,
    cosine: Tensor | None,
    sine: Tensor | None,
    packed_key_value: bool,
) -> Tensor: ...
