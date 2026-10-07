# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.models.timesfm3.block import TimesFM3TransformerBlock


def remap_ckpt(
    ckpt: dict[str, Tensor],
    model: torch.nn.Module,
) -> dict[str, Tensor]:
    """Remap Google TimesFM-3 weights for strict SDM loading.

    Args:
        ckpt: Google's state dictionary with separate query, key, and value
            projections.
        model: Target model containing TimesFM-3 transformer blocks.

    Returns:
        A new state dictionary with renamed layers, fused attention
        projections, and fixed rotary frequencies.
    """
    out = {}
    for key, value in ckpt.items():
        if key.startswith("transformer_stack.layers."):
            key = "icl_block.layers." + key.removeprefix(
                "transformer_stack.layers."
            )
        elif key.startswith("output_head."):
            key = "icl_block.head." + key.removeprefix("output_head.")
        elif key.startswith("pre_transformer_resblock.hidden_layer."):
            key = "patch_embedding.mlp.0." + key.removeprefix(
                "pre_transformer_resblock.hidden_layer."
            )
        elif key.startswith("pre_transformer_resblock.output_layer."):
            key = "patch_embedding.mlp.2." + key.removeprefix(
                "pre_transformer_resblock.output_layer."
            )
        elif key.startswith("pre_transformer_resblock.residual_layer."):
            key = "patch_embedding.res." + key.removeprefix(
                "pre_transformer_resblock.residual_layer."
            )
        out[key] = value

    for name, module in model.named_modules():
        if not isinstance(module, TimesFM3TransformerBlock):
            continue
        prefix = f"{name}." if name else ""
        for source_name, target_name in (
            ("pre_seq_attn_ln", "time.query_norm"),
            ("post_seq_attn_ln", "time.post_attn_norm"),
            ("pre_var_attn_ln", "var.query_norm"),
            ("post_var_attn_ln", "var.post_attn_norm"),
            ("pre_ff_ln", "var.mlp.0"),
            ("ff0", "var.mlp.1"),
            ("ff1", "var.mlp.3"),
            ("post_ff_ln", "var.mlp.4"),
        ):
            out[prefix + target_name + ".weight"] = out.pop(
                prefix + source_name + ".weight"
            )
        for source_axis, target_axis, attention in (
            ("seq", "time", module.time.attn),
            ("var", "var", module.var.attn),
        ):
            source_prefix = f"{prefix}{source_axis}_attn."
            target_prefix = f"{prefix}{target_axis}.attn."
            qkv_weights = [
                out.pop(source_prefix + f"{part}_proj.weight")
                for part in ("query", "key", "value")
            ]
            out[target_prefix + "qkv_lin.weight"] = torch.cat(qkv_weights)
            out[target_prefix + "out_lin.weight"] = out.pop(
                source_prefix + "out_proj.weight"
            )
            norm_index = 1 if source_axis == "seq" else 0
            scale_index = 2 if source_axis == "seq" else 1
            out[target_prefix + f"query_transform.{norm_index}.weight"] = (
                out.pop(source_prefix + "query_ln.weight")
            )
            out[target_prefix + f"key_transform.{norm_index}.weight"] = (
                out.pop(source_prefix + "key_ln.weight")
            )
            out[target_prefix + f"query_transform.{scale_index}.weight"] = (
                out.pop(source_prefix + "per_dim_scale.per_dim_scale")
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
                    out[target_prefix + f"{branch}_transform.0.inv_freq"] = (
                        inv_freq
                    )

    return out
