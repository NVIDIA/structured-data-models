# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
def test_transformer_block(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)


@onlyCUDA
def test_transformer_block_float64_autocast() -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        query_scaling=None,
        device="cuda",
        dtype=torch.float64,
    )
    query = torch.randn(2, 5, 32, device="cuda", dtype=torch.float64)
    key_value = torch.randn(2, 3, 32, device="cuda", dtype=torch.float64)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)
