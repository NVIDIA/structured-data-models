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

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ResidualBlockConfig:
    """Configure a TimesFM-3 residual block.

    Args:
        hidden_dims: Hidden-layer width.
        output_dims: Output width.
        use_bias: Whether linear layers use bias parameters.
        activation: Hidden-layer activation.
        identity_skip: Whether to use an identity residual connection.
        prenorm: Normalization applied before the hidden layer.
    """

    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: Literal["relu", "swish", "none"]
    identity_skip: bool = False
    prenorm: Literal["rms", "none"] = "none"


@dataclass(frozen=True)
class TransformerConfig:
    """Configure a TimesFM-3 mixing transformer.

    Args:
        model_dims: Input and output width.
        hidden_dims: Feed-forward hidden width.
        num_heads: Number of attention heads.
        qk_norm: Query and key normalization.
        use_bias: Whether linear layers use bias parameters.
        use_rope_seq: Whether temporal attention uses rotary embeddings.
        use_rope_var: Whether variate attention uses rotary embeddings.
        ff_activation: Feed-forward activation.
        v_norm: Value normalization.
        causal_attention: Whether temporal attention is causal.
        use_memory_efficient_attention: Whether to retain TimesFM's
            square-root head-dimension logit scaling.
        use_sdpa: Whether to use PyTorch scaled dot-product attention.
    """

    model_dims: int
    hidden_dims: int
    num_heads: int
    qk_norm: Literal["rms", "none"]
    use_bias: bool
    use_rope_seq: bool
    use_rope_var: bool
    ff_activation: Literal["relu", "swish", "none"]
    v_norm: Literal["rms", "none"] = "none"
    causal_attention: bool = True
    use_memory_efficient_attention: bool = True
    use_sdpa: bool = True
