# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm.models.timesfm3.configs import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.normalization import PerDimScale
from sdm.testing import withCUDA


def test_checkpoint_config() -> None:
    residual = ResidualBlockConfig(
        hidden_dims=1280,
        output_dims=1280,
        use_bias=False,
        activation="relu",
    )
    transformer = TransformerConfig(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
    )
    stacked = StackedTransformersConfig(
        num_layers=20,
        transformer=transformer,
    )

    assert residual.dropout == 0.0
    assert residual.identity_skip is False
    assert residual.prenorm == "none"
    assert transformer.max_variates == 32
    assert transformer.use_memory_efficient_attention is True
    assert transformer.use_sdpa is True
    assert stacked.use_remat is True


@withCUDA
def test_per_dim_scale(device: torch.device) -> None:
    num_dims = 8
    module = PerDimScale(num_dims=num_dims, device=device)
    tensor = torch.ones(2, 3, num_dims, device=device)

    out = module(tensor)

    torch.testing.assert_close(
        out,
        torch.full_like(tensor, 1.0 / math.sqrt(num_dims)),
    )


@withCUDA
def test_per_dim_scale_dtype_device(device: torch.device) -> None:
    dtype = torch.float64
    module = PerDimScale(num_dims=4, device=device, dtype=dtype)
    tensor = torch.ones(2, 4, device=device, dtype=dtype)

    out = module(tensor)

    assert out.dtype == dtype
    assert out.device == device
    assert module.per_dim_scale.dtype == dtype
    assert module.per_dim_scale.device == device
    assert tuple(module.state_dict()) == ("per_dim_scale",)
