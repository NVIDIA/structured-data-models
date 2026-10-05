# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.models.timesfm3.transformer import TimesFM3TransformerBlock


def remap_ckpt(
    ckpt: dict[str, Tensor],
    model: torch.nn.Module,
) -> dict[str, Tensor]:
    """Remap Google TimesFM-3 attention weights for strict SDM loading.

    Args:
        ckpt: Google's state dictionary with separate query, key, and value
            projections.
        model: Target model containing TimesFM-3 transformer blocks.

    Returns:
        A new state dictionary with fused attention projections and fixed
        rotary frequencies.
    """
    out = dict(ckpt)
    for name, module in model.named_modules():
        if not isinstance(module, TimesFM3TransformerBlock):
            continue
        prefix = f"{name}." if name else ""
        for axis, attention in (
            ("seq", module.seq_attn),
            ("var", module.var_attn),
        ):
            attn_prefix = f"{prefix}{axis}_attn."
            qkv_weights = [
                out.pop(attn_prefix + f"{part}_proj.weight")
                for part in ("query", "key", "value")
            ]
            out[attn_prefix + "qkv_lin.weight"] = torch.cat(qkv_weights)
            out[attn_prefix + "out_lin.weight"] = out.pop(
                attn_prefix + "out_proj.weight"
            )
            for branch in ("query", "key"):
                out[attn_prefix + f"{branch}_transform.norm.weight"] = out.pop(
                    attn_prefix + f"{branch}_ln.weight"
                )
            out[attn_prefix + "query_transform.scale.weight"] = out.pop(
                attn_prefix + "per_dim_scale.per_dim_scale"
            )

            if axis == "seq":
                exponent = (
                    torch.arange(
                        0,
                        attention.head_dim,
                        2,
                        device=qkv_weights[0].device,
                        dtype=torch.float32,
                    )
                    / attention.head_dim
                )
                inv_freq = 1.0 / (10_000**exponent)
                for branch in ("query", "key"):
                    out[attn_prefix + f"{branch}_transform.rope.inv_freq"] = (
                        inv_freq
                    )

    return out
