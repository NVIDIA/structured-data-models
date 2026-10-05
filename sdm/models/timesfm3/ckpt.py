# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.models.timesfm3.block import TimesFM3TransformerBlock


def remap_ckpt(
    ckpt: dict[str, Tensor],
    model: torch.nn.Module,
) -> dict[str, Tensor]:
    """Remap Google TimesFM-3 block weights for strict SDM loading.

    Args:
        ckpt: Google's state dictionary with separate query, key, and value
            projections.
        model: Target model containing TimesFM-3 transformer blocks.

    Returns:
        A new state dictionary with renamed block weights, fused attention
        projections, and fixed rotary frequencies.
    """
    out = dict(ckpt)
    for name, module in model.named_modules():
        if not isinstance(module, TimesFM3TransformerBlock):
            continue
        prefix = f"{name}." if name else ""
        for source_name, target_name in (
            ("pre_seq_attn_ln", "pre_time_attn"),
            ("post_seq_attn_ln", "post_time_attn"),
            ("pre_var_attn_ln", "pre_var_attn"),
            ("post_var_attn_ln", "post_var_attn"),
            ("pre_ff_ln", "ff_block.pre_ff"),
            ("post_ff_ln", "ff_block.post_ff"),
        ):
            out[prefix + target_name + ".weight"] = out.pop(
                prefix + source_name + ".weight"
            )
        for layer in ("ff0", "ff1"):
            out[prefix + f"ff_block.{layer}.weight"] = out.pop(
                prefix + f"{layer}.weight"
            )

        for source_axis, target_axis, attention in (
            ("seq", "time", module.time_attn),
            ("var", "var", module.var_attn),
        ):
            source_prefix = f"{prefix}{source_axis}_attn."
            target_prefix = f"{prefix}{target_axis}_attn."
            qkv_weights = [
                out.pop(source_prefix + f"{part}_proj.weight")
                for part in ("query", "key", "value")
            ]
            out[target_prefix + "qkv_lin.weight"] = torch.cat(qkv_weights)
            out[target_prefix + "out_lin.weight"] = out.pop(
                source_prefix + "out_proj.weight"
            )
            for branch in ("query", "key"):
                out[target_prefix + f"{branch}_transform.norm.weight"] = (
                    out.pop(source_prefix + f"{branch}_ln.weight")
                )
            out[target_prefix + "query_transform.scale.weight"] = out.pop(
                source_prefix + "per_dim_scale.per_dim_scale"
            )

            if source_axis == "seq":
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
                    out[
                        target_prefix + f"{branch}_transform.rope.inv_freq"
                    ] = inv_freq

    return out
