# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from research.multigpu.compact_ensemble import (
    CompactEnsembleParallel,
    compact_cache,
    storage_bytes,
)

from sdm import CategoricalTensor, Recipe, TableTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.models import EnsembleParallel, KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


def test_compact_unused_storage_and_preserve_aliases() -> None:
    projection = torch.arange(120.0).view(10, 12)
    value = projection[:, 6:]
    key = value.square()
    cache = Cache(kv=KVCacheEntry(key, value), same_value=value).freeze()
    compact = compact_cache(cache)
    assert compact.is_replaying
    assert storage_bytes(cache) == 720
    assert storage_bytes(compact) == 480
    original = list(cache._tensors())
    actual = list(compact._tensors())
    for before, after in zip(original, actual, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert actual[1].data_ptr() == actual[2].data_ptr()


def test_fully_referenced_shared_storage_is_not_copied() -> None:
    projection = torch.arange(120.0).view(10, 12)
    key, value = projection.chunk(2, dim=-1)
    cache = Cache(kv=KVCacheEntry(key, value)).freeze()
    compact = compact_cache(cache)
    assert storage_bytes(cache) == storage_bytes(compact) == 480
    for before, after in zip(
        cache._tensors(), compact._tensors(), strict=True
    ):
        assert before.data_ptr() == after.data_ptr()


def test_kumo_cache_compaction_preserves_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 16,
            "num_embedding_layers": 2,
            "num_embedding_heads": 4,
            "num_inducing_points": 8,
            "group_size": 2,
            "num_frequencies": 4,
            "num_readout_tokens": 2,
            "icl_channels": 32,
            "num_icl_layers": 2,
            "num_icl_heads": 4,
            "num_icl_key_value_heads_for_query": 2,
        },
    )
    model = KumoTabular(task="classification", size="small", pretrained=False)
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(32, 6)
    y = TableTensor(
        categorical=CategoricalTensor.from_tensor(
            torch.arange(32).remainder(3).view(-1, 1)
        )
    )
    with EnsembleParallel([copy.deepcopy(model)]) as reference:
        reference.fit(x, y, num_estimators=2, recipe=Recipe())
        expected = reference.predict(x[:4])
    with CompactEnsembleParallel([model]) as compact:
        compact.fit(x, y, num_estimators=2, recipe=Recipe())
        actual = compact.predict(x[:4])
        torch.testing.assert_close(
            actual.numerical, expected.numerical, rtol=0, atol=0
        )
        assert all(
            after < before
            for before, after in compact.compaction_storage_bytes
        )
