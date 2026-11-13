# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm.nn import GatedLogScale, LogScale, QASSMax
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_qassmax(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    channels = 2
    num_heads = 3
    module = QASSMax(
        channels=channels,
        num_heads=num_heads,
        hidden_channels=4,
        device=device,
    )

    query = torch.randn(2, 3, num_heads, channels, device=device)

    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    out = module(query, key_len=key_len)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
    ],
)
def test_log_scale(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = LogScale(num_heads=3, device=device)
    with torch.no_grad():
        module.head_scale.copy_(torch.tensor([0.5, 1.0, 1.5], device=device))

    query = torch.arange(1, 37, dtype=torch.float32, device=device).reshape(
        2, 3, 3, 2
    )
    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)
        expected_key_len = key_len.float().broadcast_to((2, 3))
    else:
        expected_key_len = torch.full((2, 3), key_len, device=device)

    out = module(query, key_len=key_len)

    expected_scale = expected_key_len.clamp(min=1.0).log()[..., None, None]
    expected_head_scale = module.head_scale[None, None, :, None]
    torch.testing.assert_close(
        out,
        query * expected_scale * expected_head_scale,
    )


@withCUDA
def test_gated_log_scale_forward(device: torch.device) -> None:
    module = GatedLogScale(
        channels=2,
        num_heads=3,
        hidden_channels=4,
        device=device,
    )
    query = torch.ones(2, 3, 3, 2, device=device)
    out = module(query, key_len=7)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


def _log_scale_reference(
    module: LogScale,
    query: Tensor,
    key_len: Tensor | int,
) -> Tensor:
    length = torch.as_tensor(key_len, device=query.device).float()
    length = length.clamp(min=1).log().to(query.dtype)
    heads = module.head_scale.to(query.dtype)
    heads = heads.reshape(*((1,) * (query.dim() - 2)), -1, 1)
    out = query * length[..., None, None] * heads
    if isinstance(module, GatedLogScale):
        out = out * (1 + module.gate(query).tanh())
    return out


@withCUDA
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty"])
@pytest.mark.parametrize("length_kind", ["scalar", "batch", "query"])
def test_log_scale_exact_inference(
    device: torch.device,
    gated: bool,
    dtype: torch.dtype,
    layout: str,
    length_kind: str,
) -> None:
    module = (
        GatedLogScale(
            channels=7, num_heads=3, hidden_channels=11, device=device
        )
        if gated
        else LogScale(num_heads=3, device=device)
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.5, 0.5)
    rows = 0 if layout == "empty" else 17
    query = torch.randn(2, rows, 3, 14, dtype=dtype, device=device)
    query = query[..., ::2]
    if layout != "strided":
        query = query.contiguous()
    key_len: Tensor | int = 311
    if length_kind == "batch":
        key_len = torch.tensor([[0], [311]], device=device)
    elif length_kind == "query":
        key_len = torch.arange(2 * rows, device=device).reshape(2, rows)
    with (
        torch.inference_mode(),
        torch.autocast(
            device.type, dtype=dtype, enabled=dtype != torch.float32
        ),
    ):
        expected = _log_scale_reference(module, query, key_len)
        actual = module(query, key_len=key_len)
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )


@withCUDA
@pytest.mark.parametrize("gated", [False, True])
def test_log_scale_gradients(device: torch.device, gated: bool) -> None:
    module = (
        GatedLogScale(
            channels=7, num_heads=3, hidden_channels=11, device=device
        )
        if gated
        else LogScale(num_heads=3, device=device)
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.5, 0.5)
    query = torch.randn(2, 17, 3, 7, device=device, requires_grad=True)
    key_len = torch.tensor(11.0, device=device, requires_grad=True)
    expected = _log_scale_reference(module, query, key_len)
    actual = module(query, key_len=key_len)
    inputs = (query, key_len, *module.parameters())
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    actual_grad = torch.autograd.grad(actual.sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize("gated", [False, True])
def test_log_scale_without_triton(
    monkeypatch: pytest.MonkeyPatch, gated: bool
) -> None:
    kernel = importlib.import_module("sdm._kernels.log_scale")
    monkeypatch.setattr(kernel, "_triton_log_scale", None)
    module = (
        GatedLogScale(7, 3, 11, device="cuda")
        if gated
        else LogScale(3, device="cuda")
    )
    query = torch.randn(17, 3, 7, device="cuda", dtype=torch.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=query.dtype):
        expected = _log_scale_reference(module, query, 31)
        actual = module(query, key_len=31)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gated_log_scale_dtype_fallback(dtype: torch.dtype) -> None:
    module = GatedLogScale(7, 3, 11, device="cuda", dtype=dtype)
    query = torch.randn(17, 3, 7, device="cuda", dtype=dtype)
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda", dtype=torch.float16, enabled=dtype != torch.float64
        ),
    ):
        expected = _log_scale_reference(module, query, 31)
        actual = module(query, key_len=31)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize(
    "key_len", [-17, 0, 1, 311, 2**24 + 1, 2**31 + 1, 2**63 - 1]
)
def test_log_scale_integer_lengths(dtype: torch.dtype, key_len: int) -> None:
    module = GatedLogScale(7, 3, 11, device="cuda")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.5, 0.5)
    query = torch.randn(17, 3, 7, device="cuda", dtype=dtype)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        expected = _log_scale_reference(module, query, key_len)
        actual = module(query, key_len=key_len)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize("tensor_length", [False, True])
def test_log_scale_compilation(compilation: None, tensor_length: bool) -> None:
    module = GatedLogScale(7, 3, 11, device="cuda")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.5, 0.5)
    query = torch.randn(17, 3, 7, device="cuda", dtype=torch.float16)
    key_len = torch.tensor(31, device="cuda") if tensor_length else 31
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=query.dtype):
        expected = _log_scale_reference(module, query, key_len)
        actual = compiled(query, key_len=key_len)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
