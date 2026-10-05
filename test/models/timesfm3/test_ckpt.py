# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor
from torch.nn import ModuleDict, ModuleList

from sdm.models.timesfm3.ckpt import remap_ckpt
from sdm.models.timesfm3.transformer import TimesFM3TransformerBlock
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
        name: torch.ones(value.shape, device=device, dtype=value.dtype)
        for name, value in model.state_dict().items()
        if not name.startswith((prefix + "seq_attn.", prefix + "var_attn."))
    }
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
    for axis in ("seq", "var"):
        attn_prefix = f"{prefix}{axis}_attn."
        torch.testing.assert_close(
            mapped[attn_prefix + "qkv_lin.weight"],
            torch.cat(
                [
                    source[attn_prefix + f"{part}_proj.weight"]
                    for part in ("query", "key", "value")
                ]
            ),
        )
        torch.testing.assert_close(
            mapped[attn_prefix + "query_transform.scale.weight"],
            source[attn_prefix + "per_dim_scale.per_dim_scale"],
        )
    for branch in ("query", "key"):
        torch.testing.assert_close(
            mapped[prefix + f"seq_attn.{branch}_transform.rope.inv_freq"],
            torch.tensor([1.0, 0.01], device=device),
        )

    model.load_state_dict(mapped, strict=True, assign=True)
    assert all(parameter.device == device for parameter in model.parameters())
    x = torch.arange(48, device=device, dtype=dtype).reshape(1, 2, 3, 8) / 10
    patch_mask = torch.zeros(1, 2, 3, device=device, dtype=torch.bool)
    output, mask = block(x, patch_mask)
    assert output.isfinite().all()
    assert mask.shape == (2, 1, 3, 3)
