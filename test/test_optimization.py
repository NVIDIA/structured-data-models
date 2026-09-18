# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from sdm import optimize
from sdm.optimization import _attention_quantization


def test_nested_optimization_restores_on_exception() -> None:
    assert _attention_quantization.get() is None
    with optimize(attention="fp8"):
        with optimize():
            assert _attention_quantization.get() == "fp8"
        with optimize(attention=None):
            assert _attention_quantization.get() is None
        with (
            pytest.raises(ValueError, match="test exception"),
            optimize(attention=None),
        ):
            raise ValueError("test exception")
        assert _attention_quantization.get() == "fp8"
    assert _attention_quantization.get() is None


def test_optimization_is_thread_local() -> None:
    with optimize(attention="fp8"), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(_attention_quantization.get).result() is None
        assert _attention_quantization.get() == "fp8"


def test_optimization_is_task_local() -> None:
    async def enabled() -> None:
        with optimize(attention="fp8"):
            await asyncio.sleep(0)
            assert _attention_quantization.get() == "fp8"

    async def disabled() -> None:
        await asyncio.sleep(0)
        assert _attention_quantization.get() is None

    async def run() -> None:
        await asyncio.gather(enabled(), disabled())

    asyncio.run(run())
