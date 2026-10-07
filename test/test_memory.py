# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

import pytest
import torch
from torch.torch_version import TorchVersion

from sdm._memory import configure_pinned_memory
from sdm.testing import onlyCUDA


@pytest.mark.parametrize("version", ["2.7.0", "2.12.0", "2.12.0.dev20260701"])
def test_pinned_memory_older_torch(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    def unsupported(*args: object) -> None:
        raise AssertionError("Older PyTorch must keep its allocation behavior")

    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(
        torch._C,
        "_accelerator_getAllocatorSettings",
        unsupported,
        raising=False,
    )
    monkeypatch.setattr(
        torch._C,
        "_accelerator_setAllocatorSettings",
        unsupported,
        raising=False,
    )
    configure_pinned_memory()


@onlyCUDA
@pytest.mark.parametrize(
    ("allocator_env", "runtime_settings", "expected_mib"),
    [
        ({}, "", 6),
        (
            {"PYTORCH_ALLOC_CONF": "pinned_max_cached_size_mb:64"},
            "",
            6,
        ),
        (
            {"PYTORCH_CUDA_ALLOC_CONF": "pinned_max_round_threshold_mb:128"},
            "",
            8,
        ),
        ({}, "max_split_size_mb:256,pinned_max_cached_size_mb:64", 6),
        ({}, "pinned_max_round_threshold_mb:128", 8),
    ],
)
def test_fit_pinned_memory(
    allocator_env: dict[str, str],
    runtime_settings: str,
    expected_mib: int,
) -> None:
    if TorchVersion(torch.__version__) < "2.13":
        pytest.skip("Pinned allocation rounding requires PyTorch 2.13")

    # A fresh process isolates allocator state and previously retained blocks.
    script = """
import json
import sys
import torch
from sdm import Recipe, Stype
from sdm.models import ICLModel

class CacheModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
    supports_related_tables = False

    @classmethod
    def default_recipe(cls):
        return Recipe()

    def _forward(self, x_context, cache, **kwargs):
        cache['tensor'] = torch.full(
            (3 * 1024**2 // 4,), 7.0, device=x_context.device
        )

x = torch.zeros(2, 1, device='cuda')
if sys.argv[1]:
    torch._C._accelerator_setAllocatorSettings(sys.argv[1])
before_settings = torch._C._accelerator_getAllocatorSettings()
before_bytes = torch.cuda.host_memory_stats()['allocated_bytes.current']
model = CacheModel(task='regression').eval()
model.fit(x, x, num_estimators=2)
after_bytes = torch.cuda.host_memory_stats()['allocated_bytes.current']
states = [model._cache[i]['tensor'] for i in range(2)]
print(json.dumps({
    'before_settings': before_settings,
    'settings': torch._C._accelerator_getAllocatorSettings(),
    'bytes': after_bytes - before_bytes,
    'values_ok': all(t.is_pinned() and t[0] == 7 and t[-1] == 7
                     for t in states),
}))
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
        }
    }
    env.update(allocator_env)
    output = subprocess.check_output(
        [sys.executable, "-c", script, runtime_settings],
        env=env,
        text=True,
    )
    result = json.loads(output)
    assert result["values_ok"]
    assert (
        expected_mib * 1024**2
        <= result["bytes"]
        < (expected_mib * 1024**2 + 1024)
    )
    assert result["before_settings"] in result["settings"]
    if "pinned_max_round_threshold_mb" not in result["before_settings"]:
        assert "pinned_max_round_threshold_mb:1" in result["settings"]
