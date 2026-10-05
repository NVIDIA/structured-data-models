# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor
from torch.nn import ModuleDict, ModuleList

from sdm.models.timesfm3.block import TimesFM3TransformerBlock
from sdm.models.timesfm3.ckpt import remap_ckpt
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_remap_ckpt_strict_meta_load(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    block = TimesFM3TransformerBlock(8, 2, device="meta", dtype=dtype)
    model = ModuleDict({"layers": ModuleList([block])})
    prefix = "layers.0."
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
            mapped[prefix + target_name + ".weight"],
            source[prefix + source_name + ".weight"],
        )
    for layer in ("ff0", "ff1"):
        torch.testing.assert_close(
            mapped[prefix + f"ff_block.{layer}.weight"],
            source[prefix + f"{layer}.weight"],
        )
    for source_axis, target_axis in (("seq", "time"), ("var", "var")):
        source_prefix = f"{prefix}{source_axis}_attn."
        target_prefix = f"{prefix}{target_axis}_attn."
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
        torch.testing.assert_close(
            mapped[prefix + f"time_attn.{branch}_transform.rope.inv_freq"],
            torch.tensor([1.0, 0.01], device=device),
        )

    model.load_state_dict(mapped, strict=True, assign=True)
    assert all(parameter.device == device for parameter in model.parameters())
    x = torch.arange(48, device=device, dtype=dtype).reshape(1, 2, 3, 8) / 10
    patch_mask = torch.zeros(1, 2, 3, device=device, dtype=torch.bool)
    output = block(x, patch_mask)
    assert output.isfinite().all()
    assert output.shape == x.shape
