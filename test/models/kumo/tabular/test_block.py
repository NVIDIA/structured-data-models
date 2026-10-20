# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import cast

import pytest
import torch
from torch.nn import Linear, Sequential

from sdm.cache import KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale, RotaryEmbedding
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
def test_transformer_block(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels", [128, 256])
def test_transformer_block_rope(
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> None:
    block = KumoTabularTransformerBlock(
        channels=channels,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        rope=RotaryEmbedding(
            channels=channels // 4,
            layout="split_half",
            requires_grad=False,
            device=device,
        ),
        device=device,
    )
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.02)
        cast(Linear, cast(Sequential, block.mlp)[-1]).weight.normal_(std=0.02)

    reference = copy.deepcopy(block)
    reference.attn.query_transform = Sequential(
        *cast(Sequential, reference.attn.query_transform)
    )
    reference.attn.key_transform = Sequential(
        *cast(Sequential, reference.attn.key_transform)
    )
    block.load_state_dict(reference.state_dict(), strict=True)
    query = torch.randn(2, 17, channels, device=device, dtype=dtype)
    key_value = torch.randn(2, 13, channels, device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        actual = block(query=query, key_value=key_value)
        expected = reference(query=query, key_value=key_value)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize(
    ("channels", "head_dim"), [(128, 32), (256, 64), (384, 128), (1024, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rotary", ["full", "partial", "interleaved"])
def test_projected_rotary_query(
    channels: int,
    head_dim: int,
    dtype: torch.dtype,
    cached: bool,
    strided: bool,
    rotary: str,
) -> None:
    heads = channels // head_dim
    block = KumoTabularTransformerBlock(
        channels=channels,
        num_heads=heads,
        query_scaling=LogScale(num_heads=heads, device="cuda"),
        rope=RotaryEmbedding(
            channels=head_dim,
            layout="interleaved" if rotary == "interleaved" else "split_half",
            partial_rotary_factor=0.5 if rotary == "partial" else 1.0,
            requires_grad=False,
            device="cuda",
        ),
        device="cuda",
        dtype=dtype,
    ).eval()
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.03)
        cast(Linear, cast(Sequential, block.mlp)[-1]).weight.normal_(std=0.03)
    reference = copy.deepcopy(block)
    assert reference.attn.query_transform is not None
    reference.attn.query_transform = Sequential(reference.attn.query_transform)
    key_value = torch.randn(3, 13, channels, device="cuda", dtype=dtype)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if cached:
            _, key_value = reference(
                query=key_value, key_value=key_value, return_key_value=True
            )
        for rows in (17, 31):
            query = torch.randn(
                3, rows * 2, channels, device="cuda", dtype=dtype
            )
            query = query[:, ::2] if strided else query[:, :rows].contiguous()
            actual = block(query=query, key_value=key_value)
            expected = reference(query=query, key_value=key_value)
            assert torch.equal(actual, expected)


@onlyCUDA
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("target", ["norm", "sequence"])
@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_projected_rotary_query_preserves_hooks(
    requires_grad: bool, target: str, hook_kind: str
) -> None:
    block = KumoTabularTransformerBlock(
        channels=128,
        num_heads=4,
        query_scaling=None,
        rope=RotaryEmbedding(channels=32, layout="split_half", device="cuda"),
        device="cuda",
    ).eval()
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.03)
    reference = copy.deepcopy(block)
    original = reference.attn.query_transform
    assert isinstance(original, Sequential)
    reference.attn.query_transform = Sequential(original)
    handles = []
    for transform in (block.attn.query_transform, original):
        assert isinstance(transform, Sequential)
        hooked = transform[1] if target == "norm" else transform
        if hook_kind == "pre":
            handles.append(
                hooked.register_forward_pre_hook(
                    lambda mod, args: (args[0] + 0.25,)
                )
            )
        else:
            handles.append(
                hooked.register_forward_hook(lambda mod, args, out: out + 0.25)
            )
    try:
        query = torch.randn(
            3, 33, 128, device="cuda", requires_grad=requires_grad
        )
        key_value = KVCacheEntry(
            torch.randn(3, 17, 4, 32, device="cuda", dtype=torch.float16),
            torch.randn(3, 17, 4, 32, device="cuda", dtype=torch.float16),
        )
        with (
            torch.set_grad_enabled(requires_grad),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            actual = block(query=query, key_value=key_value)
            expected = reference(query=query, key_value=key_value)
            assert torch.equal(actual, expected)
            if requires_grad:
                actual_grad = torch.autograd.grad(actual.sum(), query)[0]
                expected_grad = torch.autograd.grad(expected.sum(), query)[0]
                torch.testing.assert_close(actual_grad, expected_grad)
    finally:
        for handle in handles:
            handle.remove()


@onlyCUDA
@pytest.mark.parametrize(
    ("channels", "head_dim"), [(128, 32), (256, 64), (384, 128), (1024, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rotary", ["full", "partial", "interleaved"])
def test_projected_rotary_key_value(
    channels: int,
    head_dim: int,
    dtype: torch.dtype,
    cached: bool,
    strided: bool,
    rotary: str,
) -> None:
    heads = channels // head_dim
    block = KumoTabularTransformerBlock(
        channels=channels,
        num_heads=heads,
        query_scaling=LogScale(num_heads=heads, device="cuda"),
        rope=RotaryEmbedding(
            channels=head_dim,
            layout="interleaved" if rotary == "interleaved" else "split_half",
            partial_rotary_factor=0.5 if rotary == "partial" else 1.0,
            requires_grad=False,
            device="cuda",
        ),
        device="cuda",
        dtype=dtype,
    ).eval()
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.03)
        cast(Linear, cast(Sequential, block.mlp)[-1]).weight.normal_(std=0.03)
    reference = copy.deepcopy(block)
    assert reference.attn.key_transform is not None
    reference.attn.key_transform = Sequential(reference.attn.key_transform)
    key_value = torch.randn(3, 34, channels, device="cuda", dtype=dtype)
    key_value = (
        key_value[:, ::2] if strided else key_value[:, :17].contiguous()
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        reference_key_value = key_value
        if cached:
            _, reference_key_value = reference(
                query=key_value, key_value=key_value, return_key_value=True
            )
            _, key_value = block(
                query=key_value, key_value=key_value, return_key_value=True
            )
            assert torch.equal(key_value.key, reference_key_value.key)
            assert torch.equal(key_value.value, reference_key_value.value)
        for rows in (13, 31):
            query = torch.randn(3, rows, channels, device="cuda", dtype=dtype)
            actual = block(query=query, key_value=key_value)
            expected = reference(query=query, key_value=reference_key_value)
            assert torch.equal(actual, expected)


@onlyCUDA
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize(
    ("target", "hook_kind"),
    [
        ("norm", "none"),
        ("norm", "pre"),
        ("norm", "post"),
        ("sequence", "pre"),
        ("sequence", "post"),
    ],
)
def test_projected_rotary_key_value_preserves_hooks(
    requires_grad: bool, target: str, hook_kind: str
) -> None:
    block = KumoTabularTransformerBlock(
        channels=128,
        num_heads=4,
        query_scaling=None,
        rope=RotaryEmbedding(channels=32, layout="split_half", device="cuda"),
        device="cuda",
    ).eval()
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.03)
    reference = copy.deepcopy(block)
    original = reference.attn.key_transform
    assert isinstance(original, Sequential)
    reference.attn.key_transform = Sequential(original)
    handles = []
    for transform in (block.attn.key_transform, original):
        if hook_kind == "none":
            continue
        assert isinstance(transform, Sequential)
        hooked = transform[1] if target == "norm" else transform
        if hook_kind == "pre":
            handles.append(
                hooked.register_forward_pre_hook(
                    lambda mod, args: (args[0] + 0.25,)
                )
            )
        else:
            handles.append(
                hooked.register_forward_hook(lambda mod, args, out: out + 0.25)
            )
    try:
        query = torch.randn(
            3, 33, 128, device="cuda", requires_grad=requires_grad
        )
        key_value = torch.randn(
            3, 17, 128, device="cuda", requires_grad=requires_grad
        )
        with (
            torch.set_grad_enabled(requires_grad),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            actual = block(query=query, key_value=key_value)
            expected = reference(query=query, key_value=key_value)
            assert torch.equal(actual, expected)
            if requires_grad:
                actual_grad = torch.autograd.grad(
                    actual.sum(), (query, key_value)
                )
                expected_grad = torch.autograd.grad(
                    expected.sum(), (query, key_value)
                )
                torch.testing.assert_close(actual_grad, expected_grad)
    finally:
        for handle in handles:
            handle.remove()
