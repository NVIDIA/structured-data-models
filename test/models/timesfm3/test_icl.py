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
            layer.time_attn.out_lin.weight.zero_()
            layer.var_attn.out_lin.weight.zero_()
            ff1 = cast(Linear, layer.ff_block.get_submodule("ff1"))
            ff1.weight.zero_()
        block.head.weight.zero_()
        block.head.weight[0, 1] = 1
        block.head.bias.zero_()

    x = torch.tensor([[[[-3.0, 4.0, -1.0, 2.0]]]], device=device)
    patch_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    torch.testing.assert_close(block(x, patch_mask), x[..., 1:2])

    with torch.no_grad():
        second = cast(TimesFM3TransformerBlock, block.layers[1])
        for name in ("ff0", "ff1"):
            linear = cast(Linear, second.ff_block.get_submodule(name))
            linear.weight.copy_(torch.eye(4, device=device))

    assert (block(x, patch_mask) > x[..., 1:2]).all()
