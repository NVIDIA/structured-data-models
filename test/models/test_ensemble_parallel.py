# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, cast

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import EnsembleParallel, ICLModel, KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


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
            numerical=x_query.numerical[..., :1] * self.weight
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


@pytest.mark.parametrize("replicas", [1, 2, 4])
@pytest.mark.parametrize("classes", [7, 12])
def test_kumo_ecoc_and_class_permutations(
    replicas: int, classes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 8,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 4,
            "group_size": 2,
            "num_frequencies": 2,
            "num_readout_tokens": 2,
            "icl_channels": 16,
            "num_icl_layers": 1,
            "num_icl_heads": 2,
            "num_icl_key_value_heads_for_query": None,
        },
    )
    model = KumoTabular(task="classification", size="small", pretrained=False)
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(24, 4)
    y = TableTensor(
        categorical=CategoricalTensor.from_tensor(
            torch.arange(24).remainder(classes).view(-1, 1) * 10
        )
    )
    recipe = Recipe(
        features=sp.ShuffleColumns(),
        target=sp.ShuffleCategories(),
        output=[sp.AverageEstimators(), sp.Softmax()],
    )
    with EnsembleParallel([copy.deepcopy(model)]) as serial:
        serial.fit(
            x,
            y,
            num_estimators=5,
            recipe=recipe,
            generator=torch.Generator().manual_seed(4),
            member_seed=42,
        )
        expected = serial.predict(x[:4])
        codebooks = [c.get("ecoc_codebook") for c in serial._caches]
    if classes <= 10:
        model.fit(
            x,
            y,
            num_estimators=5,
            recipe=recipe,
            generator=torch.Generator().manual_seed(4),
        )
        torch.testing.assert_close(
            model.predict(x[:4]).numerical, expected.numerical, rtol=0, atol=0
        )
    with EnsembleParallel(
        [copy.deepcopy(model) for _ in range(replicas)]
    ) as parallel:
        for _ in range(2):
            parallel.fit(
                x,
                y,
                num_estimators=5,
                recipe=recipe,
                generator=torch.Generator().manual_seed(4),
                member_seed=42,
            )
            actual = parallel.predict(x[:4])
            assert actual.columns == expected.columns
            torch.testing.assert_close(
                actual.numerical, expected.numerical, rtol=0, atol=0
            )
            for cache, codebook in zip(
                parallel._caches, codebooks, strict=True
            ):
                if codebook is not None:
                    torch.testing.assert_close(
                        cache["ecoc_codebook"], codebook, rtol=0, atol=0
                    )


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


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA GPUs"
)
def test_cuda_replica_initialization_and_changed_caller_stream() -> None:
    stream0 = torch.cuda.Stream(device="cuda:0")
    stream1 = torch.cuda.Stream(device="cuda:1")
    with torch.cuda.stream(stream1), torch.no_grad():
        remote = _RandomCacheModel("cuda:1")
        torch.cuda._sleep(10_000_000)
        remote.weight.fill_(2)
        local = _RandomCacheModel("cuda:0")
        local.weight.fill_(2)
        executor = EnsembleParallel([local, remote])
    with executor:
        with torch.cuda.stream(stream0):
            x = torch.arange(24.0, device="cuda:0").view(8, 3)
            y = torch.arange(8.0, device="cuda:0").view(-1, 1)
            executor.fit(x, y, recipe=Recipe(), num_estimators=4)
            ready = torch.cuda.Event()
            ready.record(stream0)
        with torch.cuda.stream(torch.cuda.Stream(device="cuda:0")):
            torch.cuda.current_stream().wait_event(ready)
            actual = executor.predict(x).numerical.cpu()
        serial_model = _RandomCacheModel("cuda:0")
        with torch.no_grad():
            serial_model.weight.fill_(2)
        with EnsembleParallel([serial_model]) as serial:
            serial.fit(x.cpu(), y.cpu(), recipe=Recipe(), num_estimators=4)
            expected = serial.predict(x.cpu()).numerical
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
