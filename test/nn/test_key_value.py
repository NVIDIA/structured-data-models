# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm.nn import Float8PerHead, KeyValue


def test_key_value_rejects_mismatched_kv_cache_dtype() -> None:
    x = torch.randn(2, 3, 1, 4)
    with pytest.raises(ValueError, match="do not match"):
        KeyValue(key=x, value=x, kv_cache_dtype=Float8PerHead())
    with pytest.raises(ValueError, match="do not match"):
        KeyValue(key=x, value=x, key_scale=x, value_scale=x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_float8_per_head(dtype: torch.dtype) -> None:
    key = torch.randn(2, 5, 3, 4, dtype=dtype)  # [B, KV, H, C]
    value = torch.randn(2, 5, 3, 4, dtype=dtype)
    entry = Float8PerHead().quantize(KeyValue(key=key, value=value))
    assert entry.key.dtype == torch.float8_e4m3fn
    assert cast(torch.Tensor, entry.key_scale).size() == (2, 1, 3, 1)

    out = entry.dequantize(dtype)
    torch.testing.assert_close(out.key, key, rtol=2**-4, atol=1e-2)
    torch.testing.assert_close(out.value, value, rtol=2**-4, atol=1e-2)
