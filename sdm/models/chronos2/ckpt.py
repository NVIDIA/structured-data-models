# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def remap_ckpt(
    ckpt: dict[str, Tensor], dtype: torch.dtype | None = None
) -> dict[str, Tensor]:
    """Map the released Chronos-2 checkpoint to SDM components.

    Args:
        ckpt: Upstream model state dictionary.
        dtype: Parameter dtype. RoPE frequencies remain in float32.
    """
    out = {"reg_token": ckpt["shared.weight"][1:2]}
    for source, target in (
        ("input_patch_embedding", "patch_embedding"),
        ("output_patch_embedding", "head"),
    ):
        for upstream, local in (
            ("hidden_layer", "mlp.0"),
            ("output_layer", "mlp.2"),
            ("residual_layer", "res"),
        ):
            for suffix in ("weight", "bias"):
                out[f"{target}.{local}.{suffix}"] = ckpt[
                    f"{source}.{upstream}.{suffix}"
                ]
    for layer in range(12):
        prefix = f"encoder.block.{layer}.layer"
        for index, stack in ((0, "time_blocks"), (1, "group_blocks")):
            source = f"{prefix}.{index}"
            target = f"icl_block.{stack}.{layer}"
            norm = ckpt[f"{source}.layer_norm.weight"]
            out[f"{target}.query_norm.weight"] = norm
            out[f"{target}.key_value_norm.weight"] = norm
            out[f"{target}.attn.qkv_lin.weight"] = torch.cat(
                [
                    ckpt[f"{source}.self_attention.{key}.weight"]
                    for key in "qkv"
                ]
            )
            out[f"{target}.attn.out_lin.weight"] = ckpt[
                f"{source}.self_attention.o.weight"
            ]
        target = f"icl_block.mlps.{layer}"
        out[f"{target}.0.weight"] = ckpt[f"{prefix}.2.layer_norm.weight"]
        out[f"{target}.1.weight"] = ckpt[f"{prefix}.2.mlp.wi.weight"]
        out[f"{target}.3.weight"] = ckpt[f"{prefix}.2.mlp.wo.weight"]
    out["icl_block.norm.weight"] = ckpt["encoder.final_layer_norm.weight"]
    if dtype is not None:
        out = {key: value.to(dtype) for key, value in out.items()}

    inv_freq = 1.0 / (
        10_000
        ** (
            torch.arange(0, 64, 2, device=out["reg_token"].device).float() / 64
        )
    )
    for layer in range(12):
        for transform in ("query_transform", "key_transform"):
            key = f"icl_block.time_blocks.{layer}.attn.{transform}.inv_freq"
            out[key] = inv_freq
    return out
