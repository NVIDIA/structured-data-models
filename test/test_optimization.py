# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
