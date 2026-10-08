# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from research.multigpu._test_models import _RandomCacheModel
from research.multigpu.process_ensemble import ProcessEnsembleParallel

from sdm.models import EnsembleParallel


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_process_ensemble_preserves_member_plan(
    workers: int, device: str
) -> None:
    if device == "cuda" and torch.cuda.device_count() < workers:
        pytest.skip("needs CUDA devices for each worker")
    primary = "cuda:0" if device == "cuda" else "cpu"
    x = torch.randn(16, 4, device=primary)
    y = torch.randn(16, 1, device=primary) * 10
    with EnsembleParallel([_RandomCacheModel(primary)]) as serial:
        serial.fit(
            x,
            y,
            num_estimators=5,
            generator=torch.Generator(device=primary).manual_seed(12),
            member_seed=31,
        )
        expected = serial.predict(x[:3])
    parallel = ProcessEnsembleParallel(
        [
            _RandomCacheModel(f"cuda:{i}" if device == "cuda" else "cpu")
            for i in range(workers)
        ]
    )
    try:
        for _ in range(2):
            parallel.fit(
                x,
                y,
                num_estimators=5,
                generator=torch.Generator(device=primary).manual_seed(12),
                member_seed=31,
            )
            actual = parallel.predict(x[:3])
            torch.testing.assert_close(
                actual.numerical, expected.numerical, rtol=0, atol=0
            )
        states = parallel.memory()
        assert len({state["pid"] for state in states}) == workers
        assert all(state["cache_storage_bytes"] > 0 for state in states)
        assert all(state["max_cpu_rss_bytes"] > 0 for state in states)
        parallel.clear()
        with pytest.raises(RuntimeError, match="fit"):
            parallel.predict(x[:3])
    finally:
        parallel.close()
