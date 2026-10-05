# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm.models.timesfm3.ckpt import remap_ckpt
from sdm.models.timesfm3.model import _TimesFM3
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_remap_ckpt_strict_meta_load(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model = _TimesFM3(
        input_patch_len=2,
        output_patch_len=4,
        quantiles=(0.1, 0.5, 0.9),
        channels=8,
        num_layers=1,
        num_heads=2,
        device="meta",
        dtype=dtype,
    )
    prefix = "transformer_stack.layers.0."
    mapped_prefix = "icl_block.layers.0."
    source: dict[str, Tensor] = {
        prefix + name + ".weight": torch.full(
            (8,), index + 1, device=device, dtype=dtype
        )
        for index, name in enumerate(
            (
                "pre_seq_attn_ln",
                "post_seq_attn_ln",
                "pre_var_attn_ln",
                "post_var_attn_ln",
                "pre_ff_ln",
                "post_ff_ln",
            )
        )
    }
    for index, name in enumerate(("ff0", "ff1")):
        source[prefix + name + ".weight"] = torch.full(
            (8, 8), index + 1, device=device, dtype=dtype
        )
    for axis, offset in (("seq", 0), ("var", 4)):
        attn_prefix = f"{prefix}{axis}_attn."
        for index, part in enumerate(("query", "key", "value", "out")):
            source[attn_prefix + f"{part}_proj.weight"] = torch.full(
                (8, 8),
                (index + offset + 1) / 10,
                device=device,
                dtype=dtype,
            )
        source[attn_prefix + "query_ln.weight"] = torch.full(
            (4,), 2.0, device=device, dtype=dtype
        )
        source[attn_prefix + "key_ln.weight"] = torch.full(
            (4,), 3.0, device=device, dtype=dtype
        )
        source[attn_prefix + "per_dim_scale.per_dim_scale"] = torch.arange(
            4, device=device, dtype=dtype
        )

    for name, shape in (
        ("pre_transformer_resblock.hidden_layer.weight", (8, 12)),
        ("pre_transformer_resblock.output_layer.weight", (8, 8)),
        ("pre_transformer_resblock.residual_layer.weight", (8, 12)),
        ("output_head.weight", (12, 8)),
        ("output_head.bias", (12,)),
    ):
        source[name] = torch.full(shape, 0.1, device=device, dtype=dtype)

    source_keys = set(source)
    mapped = remap_ckpt(source, model)

    assert set(source) == source_keys
    assert set(mapped) == set(model.state_dict())
    for source_name, target_name in (
        ("pre_seq_attn_ln", "pre_time_attn"),
        ("post_seq_attn_ln", "post_time_attn"),
        ("pre_var_attn_ln", "pre_var_attn"),
        ("post_var_attn_ln", "post_var_attn"),
        ("pre_ff_ln", "ff_block.pre_ff"),
        ("post_ff_ln", "ff_block.post_ff"),
    ):
        torch.testing.assert_close(
            mapped[mapped_prefix + target_name + ".weight"],
            source[prefix + source_name + ".weight"],
        )
    for layer in ("ff0", "ff1"):
        torch.testing.assert_close(
            mapped[mapped_prefix + f"ff_block.{layer}.weight"],
            source[prefix + f"{layer}.weight"],
        )
    for source_axis, target_axis in (("seq", "time"), ("var", "var")):
        source_prefix = f"{prefix}{source_axis}_attn."
        target_prefix = f"{mapped_prefix}{target_axis}_attn."
        torch.testing.assert_close(
            mapped[target_prefix + "qkv_lin.weight"],
            torch.cat(
                [
                    source[source_prefix + f"{part}_proj.weight"]
                    for part in ("query", "key", "value")
                ]
            ),
        )
        torch.testing.assert_close(
            mapped[target_prefix + "query_transform.scale.weight"],
            source[source_prefix + "per_dim_scale.per_dim_scale"],
        )
    for branch in ("query", "key"):
        rope_key = f"time_attn.{branch}_transform.rope.inv_freq"
        torch.testing.assert_close(
            mapped[mapped_prefix + rope_key],
            torch.tensor([1.0, 0.01], device=device),
        )

    torch.testing.assert_close(
        mapped["icl_block.head.weight"], source["output_head.weight"]
    )
    model.load_state_dict(mapped, strict=True, assign=True)
    assert all(parameter.device == device for parameter in model.parameters())

    x = torch.ones(1, 2, 3, 12, device=device, dtype=dtype)
    patch_mask = torch.zeros(1, 2, 3, device=device, dtype=torch.bool)
    output = model.icl_block(model.pre_transformer_resblock(x), patch_mask)
    assert output.isfinite().all()
    assert output.shape == (1, 2, 3, 12)
