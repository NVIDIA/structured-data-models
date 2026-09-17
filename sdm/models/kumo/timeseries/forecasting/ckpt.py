# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from torch import Tensor

_LAYER_KEYS = {
    "0.SelfAttention.q.weight": "q.weight",
    "0.SelfAttention.k.weight": "k.weight",
    "0.SelfAttention.v.weight": "v.weight",
    "0.SelfAttention.o.weight": "o.weight",
    "0.layer_norm.weight": "attention_norm.weight",
    "1.DenseReluDense.wi_0.weight": "wi_0.weight",
    "1.DenseReluDense.wi_1.weight": "wi_1.weight",
    "1.DenseReluDense.wo.weight": "wo.weight",
    "1.layer_norm.weight": "ffn_norm.weight",
}


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map an NV-Tesseract forecasting state dict to SDM parameter names.

    Only the unused T5 vocabulary embedding is discarded. Unknown parameters
    remain in the result so strict loading reports incompatible checkpoints.
    """
    out: dict[str, Tensor] = {}
    for key, value in ckpt.items():
        key = key.removeprefix("module.")
        if key == "encoder.embed_tokens.weight":
            continue
        if (
            key == "encoder.block.0.layer.0.SelfAttention."
            "relative_attention_bias.weight"
        ):
            key = "encoder.relative_attention_bias.weight"
        elif key.startswith("encoder.block."):
            match = re.fullmatch(r"encoder\.block\.(\d+)\.layer\.(.*)", key)
            if match is not None and match[2] in _LAYER_KEYS:
                key = f"encoder.layers.{match[1]}.{_LAYER_KEYS[match[2]]}"
        elif key == "encoder.final_layer_norm.weight":
            key = "encoder.norm.weight"
        out[key] = value
    return out
