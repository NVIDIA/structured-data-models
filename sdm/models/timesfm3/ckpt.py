# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D103
    out = {
        "patch_embedding.mlp.0.weight": ckpt[
            "pre_transformer_resblock.hidden_layer.weight"
        ],
        "patch_embedding.mlp.2.weight": ckpt[
            "pre_transformer_resblock.output_layer.weight"
        ],
        "patch_embedding.res.weight": ckpt[
            "pre_transformer_resblock.residual_layer.weight"
        ],
        "icl_block.head.weight": ckpt["output_head.weight"],
        "icl_block.head.bias": ckpt["output_head.bias"],
    }
    layers = sorted(
        {
            key.split(".")[2]
            for key in ckpt
            if key.startswith("transformer_stack.layers.")
        },
        key=int,
    )
    weight = ckpt["transformer_stack.layers.0.seq_attn.query_ln.weight"]
    head_dim = weight.numel()
    arange = torch.arange(
        0, head_dim, 2, device=weight.device, dtype=weight.dtype
    )
    inv_freq = 1.0 / (10_000 ** (arange / head_dim))
    for layer in layers:
        source = f"transformer_stack.layers.{layer}."
        var = f"icl_block.var_blocks.{layer}."
        for old, new in (
            ("pre_ff_ln.weight", "mlp.0.weight"),
            ("ff0.weight", "mlp.1.weight"),
            ("ff1.weight", "mlp.3.weight"),
            ("post_ff_ln.weight", "mlp.4.weight"),
        ):
            out[var + new] = ckpt[source + old]

        for side, blocks in (("seq", "time"), ("var", "var")):
            target = f"icl_block.{blocks}_blocks.{layer}."
            attn = source + side + "_attn."
            norm = ckpt[source + f"pre_{side}_attn_ln.weight"]
            out[target + "query_norm.weight"] = norm
            out[target + "key_value_norm.weight"] = norm
            out[target + "post_attn_norm.weight"] = ckpt[
                source + f"post_{side}_attn_ln.weight"
            ]
            out[target + "attn.qkv_lin.weight"] = torch.cat(
                [
                    ckpt[attn + f"{name}_proj.weight"]
                    for name in ("query", "key", "value")
                ],
                dim=0,
            )
            out[target + "attn.out_lin.weight"] = ckpt[
                attn + "out_proj.weight"
            ]
            index = int(side == "seq")
            out[target + f"attn.query_transform.{index}.weight"] = ckpt[
                attn + "query_ln.weight"
            ]
            out[target + f"attn.key_transform.{index}.weight"] = ckpt[
                attn + "key_ln.weight"
            ]
            out[target + f"attn.query_transform.{index + 1}.weight"] = ckpt[
                attn + "per_dim_scale.per_dim_scale"
            ]
            if side == "seq":
                for name in ("query", "key"):
                    out[target + f"attn.{name}_transform.0.inv_freq"] = (
                        inv_freq
                    )

    return out
