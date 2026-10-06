# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from sdm.models.tabiclv2.block import TabICLv2TransformerBlock
from sdm.testing import withCUDA


@pytest.mark.skipif(os.getenv("FULL_TEST", "0") != "1", reason="Fast test run")
@withCUDA
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_transformer_compile_output_buffer(
    device: torch.device,
    fullgraph: bool,
    strided: bool,
) -> None:
    module = TabICLv2TransformerBlock(
        channels=8,
        num_heads=2,
        norm_bias=True,
        qassmax=False,
        device=device,
    ).eval()
    query = torch.linspace(-1, 1, 48, device=device).reshape(2, 3, 8)
    # Exercise the MLP residual rather than its zero-initialized projection.
    with torch.no_grad():
        module.mlp[-1].weight.fill_(0.05)
        module.mlp[-1].bias.fill_(0.1)

    storage = torch.full((2, 3, 16 if strided else 8), -123.0, device=device)
    out = storage[..., ::2] if strided else storage
    expected = torch.empty_like(out)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(
            module, backend="inductor", fullgraph=fullgraph
        )
        with torch.no_grad():
            module(query, out=expected)
            result = compiled(query, out=out)
        torch.testing.assert_close(result, expected)
        assert result.data_ptr() == out.data_ptr()
        assert result.stride() == out.stride()
        torch.testing.assert_close(out, expected)
        if strided:
            assert (storage[..., 1::2] == -123.0).all()
    finally:
        torch._dynamo.reset()
