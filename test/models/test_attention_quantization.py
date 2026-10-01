# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
from typing import Any, Literal, cast

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    Recipe,
    RelatedTables,
    TableTensor,
    optimize,
)
from sdm.cache import Cache, QuantizedKVCacheEntry
from sdm.models import KumoRelational, KumoTabular, TabFM
from sdm.models.kumo.tabular.icl import ICLBlock as KumoICLBlock
from sdm.models.tabfm import model as tabfm_module
from sdm.models.tabfm.icl import ICLBlock as TabFMICLBlock
from sdm.testing import onlyCUDA

pytestmark = pytest.mark.usefixtures("fp8_rng")


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize(
    ("kind", "heads"),
    [("kumo", None), ("kumo", 1), ("kumo", 2), ("tabfm", None)],
)
def test_fp8_icl_cache(
    kind: str, heads: int | None, dtype: torch.dtype
) -> None:
    if torch.cuda.get_device_capability() not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 integration supports Ada, Hopper, and RTX Blackwell")
    if kind == "kumo":
        block = KumoICLBlock(
            num_classes=0,
            out_channels=5,
            channels=256,
            num_layers=2,
            num_heads=4,
            num_key_value_heads_for_query=heads,
            device="cuda",
        )
    else:
        # Stock TabFM has 256 channels per ICL attention head.
        block = TabFMICLBlock(
            num_classes=0,
            out_channels=5,
            channels=512,
            num_layers=2,
            num_heads=2,
            device="cuda",
        )
    for parameter in block.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    reference = copy.deepcopy(block)
    channels = 256 if kind == "kumo" else 512
    x = torch.randn(2, 8226, channels, device="cuda")
    y = torch.randn(2, 8193, device="cuda")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        expected = reference(x.clone(), y)
        with optimize(attention="fp8"):
            actual = block(x.clone(), y)
            cache = Cache()
            block(x[:, :8193].clone(), y, cache=cache)
            replayed = block(
                x[:, 8193:].clone(), y[:, :0], cache=cache.freeze()
            )
            split = torch.cat(
                [
                    block(x[:, 8193:8209].clone(), y[:, :0], cache=cache),
                    block(x[:, 8209:].clone(), y[:, :0], cache=cache),
                ],
                dim=-2,
            )
    entry = cache["icl_block.layer0"]
    assert isinstance(entry, QuantizedKVCacheEntry)
    assert not isinstance(cache["icl_block.layer1"], QuantizedKVCacheEntry)
    assert entry.key.dtype == torch.float8_e4m3fn
    assert entry.value.dtype == torch.float8_e4m3fn
    assert entry.query_scale.size(-2) == (4 if kind == "kumo" else 2)
    if heads is not None:
        assert entry.key.size(-2) == heads
        assert entry.key.untyped_storage().nbytes() == entry.key.numel()
        assert entry.value.untyped_storage().nbytes() == entry.value.numel()
    assert actual.isfinite().all()
    torch.testing.assert_close(replayed, actual, atol=0.01, rtol=0.03)
    torch.testing.assert_close(split, replayed, atol=0.01, rtol=0.03)
    relative_rms = (
        (actual.float() - expected.float()).square().mean()
        / expected.float().square().mean()
    ).sqrt()
    assert relative_rms < 0.08


@onlyCUDA
@pytest.mark.parametrize(
    "kind", ["kumo-small", "kumo-medium", "kumo-large", "tabfm"]
)
@pytest.mark.parametrize("estimator_batch_size", [1, 2])
def test_fp8_model_fit_predict(
    kind: str, estimator_batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.get_device_capability() not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 integration supports Ada, Hopper, and RTX Blackwell")
    kwargs: dict[str, Any] = {
        "task": "regression",
        "pretrained": False,
        "device": "cuda",
    }
    if kind.startswith("kumo-"):
        model = KumoTabular(
            size=cast(
                Literal["small", "medium", "large"], kind.removeprefix("kumo-")
            ),
            **kwargs,
        )
    else:
        # Keep the stock 256-channel ICL heads while reducing model depth.
        monkeypatch.setattr(
            tabfm_module,
            "_TabFM",
            functools.partial(
                tabfm_module._TabFM,
                num_icl_layers=2,
                num_embedding_layers=1,
                num_embedding_repeats=1,
            ),
        )
        model = TabFM(**kwargs)
    x = torch.randn(8193, 3, device="cuda")
    y = torch.randn(8193, 1, device="cuda")
    query = torch.randn(17, 3, device="cuda")
    num_estimators = 3 if estimator_batch_size == 2 else 1
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16),
        optimize(attention="fp8"),
    ):
        expected = model(
            x,
            y,
            query,
            recipe=Recipe(),
            num_estimators=num_estimators,
            estimator_batch_size=1,
        )
        model.fit(
            x,
            y,
            recipe=Recipe(),
            num_estimators=num_estimators,
            estimator_batch_size=estimator_batch_size,
        )
        actual = model.predict(query)
    assert actual.numerical.isfinite().all()
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=0.01, rtol=0.03
    )
    assert model._cache is not None
    if num_estimators > 1:
        assert model._cache.is_cpu
        assert model._cache["num_batches"] == 2
    entries = cast(Cache, model._cache[0])
    quantized = [
        entry
        for entry in entries.values()
        if isinstance(entry, QuantizedKVCacheEntry)
    ]
    assert quantized
    if kind in {"kumo-medium", "kumo-large"}:
        assert all(entry.key.size(-2) == 2 for entry in quantized)
    with torch.autocast("cuda", dtype=torch.float16):
        with pytest.raises(RuntimeError, match=r"sdm\.optimize"):
            model.predict(query)
        with optimize(attention="fp8"):
            torch.testing.assert_close(
                model.predict(query).numerical, actual.numerical
            )
    model.clear()
    assert model._cache is None


@onlyCUDA
def test_fp8_relational_fit_predict() -> None:
    if torch.cuda.get_device_capability() not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 integration supports Ada, Hopper, and RTX Blackwell")
    model = KumoRelational(
        task="regression",
        pretrained=False,
        device="cuda",
    )
    table = TableTensor(
        id=ColumnarTensor((torch.arange(8210, device="cuda"),)),
        numerical=torch.randn(8210, 1, device="cuda"),
        columns={"id": ["id"], "numerical": ["value"]},
    )
    related = RelatedTables(
        tables={"entities": table},
        relationships=[],
        task_links=[
            {"task_column": "id", "table": "entities", "table_column": "id"}
        ],
    )
    x, query = table[:8193], table[8193:]
    y = torch.randn(8193, 1, device="cuda")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16),
        optimize(attention="fp8"),
    ):
        expected = model(
            x,
            y,
            query,
            related_context_tables=related,
            related_query_tables=related,
            num_hops=0,
            recipe=Recipe(),
            num_estimators=1,
        )
        model.fit(x, y, related, num_hops=0, recipe=Recipe(), num_estimators=1)
        actual = model.predict(query, related_tables=related)
    assert actual.numerical.isfinite().all()
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=0.01, rtol=0.03
    )
    assert model._cache is not None
    entries = cast(Cache, model._cache[0])
    assert any(
        isinstance(entry, QuantizedKVCacheEntry) for entry in entries.values()
    )


@onlyCUDA
def test_fp8_ecoc_cache() -> None:
    if torch.cuda.get_device_capability() not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 integration supports Ada, Hopper, and RTX Blackwell")
    model = KumoTabular(
        task="classification", size="small", pretrained=False, device="cuda"
    )
    x = torch.randn(8193, 3, device="cuda")
    query = torch.randn(17, 3, device="cuda")
    y = TableTensor(
        columns={"categorical": ["target"]},
        categorical=CategoricalTensor.from_tensor(
            (torch.arange(8193, device="cuda") % 12).unsqueeze(-1)
        ),
    )
    with (
        torch.autocast("cuda", dtype=torch.float16),
        optimize(attention="fp8"),
    ):
        expected = model(
            x,
            y,
            query,
            recipe=Recipe(),
            num_estimators=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        model.fit(
            x,
            y,
            recipe=Recipe(),
            num_estimators=1,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        actual = model.predict(query)
    torch.testing.assert_close(
        actual.numerical, expected.numerical, atol=0.01, rtol=0.03
    )
    assert actual.size() == (1, 17, 12)
    assert model._cache is not None
    nested = cast(Cache, cast(Cache, model._cache[0])["ecoc_model"])
    assert any(
        isinstance(entry, QuantizedKVCacheEntry) for entry in nested.values()
    )
