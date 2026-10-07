# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
import torch

import sdm.processing as sp
from sdm import Recipe, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import EnsembleParallel, ICLModel


class _RandomCacheModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
    supports_related_tables = False

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, device=device))
        self.fail = False
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe(features=sp.ShuffleColumns(), target=sp.Standardize())

    def _forward(
        self,
        *,
        cache: Cache,
        x_context: TableTensor | None,
        x_query: TableTensor | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if self.fail:
            raise ValueError("injected model failure")
        if cache.is_recording:
            assert x_context is not None
            cache["bias"] = torch.rand(
                1, device=x_context.device, generator=generator
            )
            return x_context
        assert x_query is not None
        return TableTensor(
            numerical=x_query.numerical[..., :1]
            + cast(torch.Tensor, cache["bias"])
        )


@pytest.mark.parametrize("replicas", [1, 2, 4])
def test_member_order_rng_and_target_inversion(replicas: int) -> None:
    x = torch.arange(48.0).view(12, 4)
    y = torch.arange(12.0).view(-1, 1) * 10
    with EnsembleParallel([_RandomCacheModel()]) as serial:
        serial.fit(
            x, y, num_estimators=7, generator=torch.Generator().manual_seed(14)
        )
        expected = serial.predict(x[:3])
    with EnsembleParallel(
        [_RandomCacheModel() for _ in range(replicas)]
    ) as parallel:
        parallel.fit(
            x, y, num_estimators=7, generator=torch.Generator().manual_seed(14)
        )
        actual = parallel.predict(x[:3])
        torch.testing.assert_close(
            actual.numerical, expected.numerical, rtol=0, atol=0
        )
        assert actual.size() == (7, 3, 1)
        assert parallel.member_seeds == tuple(range(7))
        assert sum(parallel.cache_bytes) > 0
        torch.testing.assert_close(
            parallel.predict(x[:3]).numerical, actual.numerical
        )


def test_failed_refit_invalidates_previous_cache_and_recovers() -> None:
    models = [_RandomCacheModel(), _RandomCacheModel()]
    x, y = torch.randn(8, 3), torch.randn(8, 1)
    with EnsembleParallel(models) as executor:
        executor.fit(x, y, num_estimators=4)
        models[1].fail = True
        with pytest.raises(ValueError, match="injected"):
            executor.fit(x, y, num_estimators=4)
        with pytest.raises(RuntimeError, match="fit"):
            executor.predict(x)
        assert executor.cache_bytes == (0, 0)
        models[1].fail = False
        executor.fit(x, y, num_estimators=4)
        assert executor.predict(x).size() == (4, 8, 1)


def test_duplicate_replica_rejected() -> None:
    model = _RandomCacheModel()
    with pytest.raises(ValueError, match="distinct"):
        EnsembleParallel([model, model])


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA GPUs"
)
def test_cuda_caller_stream_and_autocast() -> None:
    with torch.cuda.stream(torch.cuda.Stream(device="cuda:0")):
        x = torch.arange(48.0, device="cuda:0").view(12, 4)
        y = torch.arange(12.0, device="cuda:0").view(-1, 1)
        with (
            EnsembleParallel([_RandomCacheModel("cuda:0")]) as serial,
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            serial.fit(
                x,
                y,
                num_estimators=5,
                generator=torch.Generator(device="cuda:0").manual_seed(14),
            )
            expected = serial.predict(x[:3])
        with EnsembleParallel(
            [_RandomCacheModel("cuda:0"), _RandomCacheModel("cuda:1")]
        ) as parallel:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                parallel.fit(
                    x,
                    y,
                    num_estimators=5,
                    generator=torch.Generator(device="cuda:0").manual_seed(14),
                )
                actual = parallel.predict(x[:3])
            torch.testing.assert_close(
                actual.numerical, expected.numerical, rtol=0, atol=0
            )
