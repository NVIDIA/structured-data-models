# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models import TabICLv2
from sdm.nn import Attention


@pytest.mark.parametrize("model_cls", [TabICLv2])
def test_model_fp8_opt_in(model_cls: type[torch.nn.Module]) -> None:
    model = model_cls(
        task="regression",
        pretrained=False,
        device="meta",
        attention_quantization="fp8",
    )
    ordinary = model_cls(task="regression", pretrained=False, device="meta")
    assert ordinary.state_dict().keys() == model.state_dict().keys()
    quantized = [
        name
        for name, module in model.named_modules()
        if isinstance(module, Attention)
        and module.attention_quantization == "fp8"
    ]
    assert quantized
    assert all("icl_block.layers" in name for name in quantized)
    assert all(
        module.attention_quantization is None
        for module in ordinary.modules()
        if isinstance(module, Attention)
    )
