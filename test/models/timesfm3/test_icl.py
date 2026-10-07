# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch.nn import Linear

from sdm.models.timesfm3.block import TimesFM3TransformerBlock
from sdm.models.timesfm3.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
def test_icl_block_applies_all_layers_and_head(device: torch.device) -> None:
    block = ICLBlock(4, 1, 2, 1, device=device)
    with torch.no_grad():
        for module in block.layers:
            layer = cast(TimesFM3TransformerBlock, module)
            layer.time.attn.out_lin.weight.zero_()
            layer.var.attn.out_lin.weight.zero_()
            ff1 = cast(Linear, layer.var.get_submodule("mlp.3"))
            ff1.weight.zero_()
        block.head.weight.zero_()
        block.head.weight[0, 1] = 1
        block.head.bias.zero_()

    x = torch.tensor([[[[-3.0, 4.0, -1.0, 2.0]]]], device=device)
    patch_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    torch.testing.assert_close(block(x, patch_mask), x[..., 1:2])

    with torch.no_grad():
        second = cast(TimesFM3TransformerBlock, block.layers[1])
        for index in (1, 3):
            linear = cast(Linear, second.var.get_submodule(f"mlp.{index}"))
            linear.weight.copy_(torch.eye(4, device=device))

    assert (block(x, patch_mask) > x[..., 1:2]).all()


@withCUDA
def test_icl_block_without_patch_mask(device: torch.device) -> None:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    block = ICLBlock(8, 4, 2, 2, device=device, dtype=dtype)
    x = torch.arange(48, device=device, dtype=dtype).reshape(1, 2, 3, 8) / 10
    patch_mask = torch.zeros(1, 2, 3, device=device, dtype=torch.bool)

    torch.testing.assert_close(
        block(x), block(x, patch_mask), rtol=2e-3, atol=2e-3
    )
