# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch
from torch import Tensor
from torch.nn import ModuleDict

from sdm.models.timesfm3.ckpt import remap_ckpt
from sdm.models.timesfm3.transformer import TimesFM3Attention
from sdm.testing import withCUDA


@pytest.mark.parametrize(
    ("use_bias", "qk_norm", "use_rope"),
    [(False, "rms", True), (True, "none", False)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@withCUDA
def test_remap_ckpt_strict_meta_load(
    device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
    qk_norm: Literal["rms", "none"],
    use_rope: bool,
) -> None:
    model = ModuleDict(
        {
            name: TimesFM3Attention(
                model_dims=8,
                num_heads=2,
                use_bias=use_bias,
                qk_norm=qk_norm,
                use_rope=use_rope,
                device="meta",
                dtype=dtype,
            )
            for name in ("seq_attn", "var_attn")
        }
    )
    source: dict[str, Tensor] = {}
    for name, offset in (("seq_attn", 0), ("var_attn", 4)):
        for index, part in enumerate(("query", "key", "value", "out")):
            value = (index + offset + 1) / 10
            source[f"{name}.{part}_proj.weight"] = torch.full(
                (8, 8), value, device=device, dtype=dtype
            )
            if use_bias:
                source[f"{name}.{part}_proj.bias"] = torch.full(
                    (8,), value, device=device, dtype=dtype
                )
        source[f"{name}.per_dim_scale.per_dim_scale"] = torch.arange(
            4, device=device, dtype=dtype
        )
        if qk_norm == "rms":
            source[f"{name}.query_ln.weight"] = torch.full(
                (4,), 2.0, device=device, dtype=dtype
            )
            source[f"{name}.key_ln.weight"] = torch.full(
                (4,), 3.0, device=device, dtype=dtype
            )

    source_keys = set(source)
    mapped = remap_ckpt(source, model)

    assert set(source) == source_keys
    assert set(mapped) == set(model.state_dict())
    for name in ("seq_attn", "var_attn"):
        torch.testing.assert_close(
            mapped[f"{name}.qkv_lin.weight"],
            torch.cat(
                [
                    source[f"{name}.{part}_proj.weight"]
                    for part in ("query", "key", "value")
                ]
            ),
        )
        torch.testing.assert_close(
            mapped[f"{name}.out_lin.weight"],
            source[f"{name}.out_proj.weight"],
        )
        torch.testing.assert_close(
            mapped[f"{name}.query_transform.scale.weight"],
            source[f"{name}.per_dim_scale.per_dim_scale"],
        )
        if use_bias:
            torch.testing.assert_close(
                mapped[f"{name}.qkv_lin.bias"],
                torch.cat(
                    [
                        source[f"{name}.{part}_proj.bias"]
                        for part in ("query", "key", "value")
                    ]
                ),
            )
            torch.testing.assert_close(
                mapped[f"{name}.out_lin.bias"],
                source[f"{name}.out_proj.bias"],
            )
        if qk_norm == "rms":
            for part in ("query", "key"):
                torch.testing.assert_close(
                    mapped[f"{name}.{part}_transform.norm.weight"],
                    source[f"{name}.{part}_ln.weight"],
                )
        if use_rope:
            for part in ("query", "key"):
                torch.testing.assert_close(
                    mapped[f"{name}.{part}_transform.rope.inv_freq"],
                    torch.tensor([1.0, 0.01], device=device),
                )

    model.load_state_dict(mapped, strict=True, assign=True)
    assert all(parameter.device == device for parameter in model.parameters())
    inputs = torch.arange(24, device=device, dtype=dtype).reshape(1, 3, 8)
    assert model["seq_attn"](inputs).isfinite().all()

    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(
            {**mapped, "unexpected": inputs}, strict=True, assign=True
        )
