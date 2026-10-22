# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D103
    out = {
        "lin.weight": ckpt["patch_embedding.value_embedding.weight"],
        "icl.relative_attention_bias.weight": ckpt[
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        ],
        "icl.norm.weight": ckpt["encoder.final_layer_norm.weight"],
        "head.weight": ckpt["head.linear.weight"],
        "head.bias": ckpt["head.linear.bias"],
    }
    layers = sorted(
        {
            key.split(".")[2]
            for key in ckpt
            if key.startswith("encoder.block.")
        },
        key=int,
    )
    for layer in layers:
        source = f"encoder.block.{layer}.layer."
        target = f"icl.layers.{layer}."
        norm = ckpt[source + "0.layer_norm.weight"]
        out[target + "query_norm.weight"] = norm
        out[target + "key_value_norm.weight"] = norm
        out[target + "attn.qkv_lin.weight"] = torch.cat(
            [
                ckpt[source + f"0.SelfAttention.{name}.weight"]
                for name in "qkv"
            ],
            dim=0,
        )
        out[target + "attn.out_lin.weight"] = ckpt[
            source + "0.SelfAttention.o.weight"
        ]
        out[target + "mlp.norm.weight"] = ckpt[source + "1.layer_norm.weight"]
        for name in ("wi_0", "wi_1", "wo"):
            out[target + f"mlp.{name}.weight"] = ckpt[
                source + f"1.DenseReluDense.{name}.weight"
            ]

    if "cross_channel_attn.attn.in_proj_weight" in ckpt:
        for old, new in (
            ("attn.in_proj_weight", "block.attn.qkv_lin.weight"),
            ("attn.in_proj_bias", "block.attn.qkv_lin.bias"),
            ("attn.out_proj.weight", "block.attn.out_lin.weight"),
            ("attn.out_proj.bias", "block.attn.out_lin.bias"),
            ("norm.weight", "norm.weight"),
            ("norm.bias", "norm.bias"),
        ):
            out["cross_channel." + new] = ckpt["cross_channel_attn." + old]

    return out
