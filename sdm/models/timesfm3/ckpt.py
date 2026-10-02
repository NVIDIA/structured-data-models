# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import RMSNorm

from sdm.models.timesfm3.transformer import TimesFM3Attention
from sdm.nn import RotaryEmbedding


def remap_ckpt(
    ckpt: dict[str, Tensor],
    model: torch.nn.Module,
) -> dict[str, Tensor]:
    """Remap Google TimesFM-3 attention weights for strict SDM loading.

    Args:
        ckpt: Google's state dictionary with separate query, key, and value
            projections.
        model: Target TimesFM-3 model.

    Returns:
        A new state dictionary with fused attention projections and fixed
        rotary frequencies.
    """
    out = dict(ckpt)
    for name, module in model.named_modules():
        if not isinstance(module, TimesFM3Attention):
            continue
        prefix = f"{name}." if name else ""
        qkv_weights = [
            out.pop(prefix + f"{part}_proj.weight")
            for part in ("query", "key", "value")
        ]
        out[prefix + "qkv_lin.weight"] = torch.cat(qkv_weights)
        out[prefix + "out_lin.weight"] = out.pop(prefix + "out_proj.weight")

        if module.qkv_lin.bias is not None:
            out[prefix + "qkv_lin.bias"] = torch.cat(
                [
                    out.pop(prefix + f"{part}_proj.bias")
                    for part in ("query", "key", "value")
                ]
            )
            out[prefix + "out_lin.bias"] = out.pop(prefix + "out_proj.bias")

        for branch in ("query", "key"):
            transform = getattr(module, f"{branch}_transform")
            if isinstance(transform.norm, RMSNorm):
                out[prefix + f"{branch}_transform.norm.weight"] = out.pop(
                    prefix + f"{branch}_ln.weight"
                )
            if isinstance(transform.rope, RotaryEmbedding):
                dim = transform.rope.channels
                exponent = (
                    torch.arange(
                        0,
                        dim,
                        2,
                        device=qkv_weights[0].device,
                        dtype=torch.float32,
                    )
                    / dim
                )
                out[prefix + f"{branch}_transform.rope.inv_freq"] = 1.0 / (
                    10_000**exponent
                )

        out[prefix + "query_transform.scale.weight"] = out.pop(
            prefix + "per_dim_scale.per_dim_scale"
        )

    return out
