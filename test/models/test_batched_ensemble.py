# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from research.multigpu.batched_ensemble import BatchedEnsembleParallel

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, TableTensor
from sdm.models import EnsembleParallel, KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


@pytest.mark.parametrize("classes", [0, 7, 12])
@pytest.mark.parametrize("replicas", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [2, None])
def test_local_member_batching(
    classes: int,
    replicas: int,
    batch_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
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
    model = KumoTabular(
        task="classification" if classes else "regression",
        size="small",
        pretrained=False,
    )
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(24, 4)
    if classes:
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
    else:
        y = TableTensor(numerical=torch.randn(24, 1) * 100 + 20)
        recipe = Recipe(
            features=sp.ShuffleColumns(),
            target=sp.Standardize(),
            output=sp.AverageEstimators(),
        )
    with EnsembleParallel([copy.deepcopy(model)]) as serial:
        serial.fit(
            x,
            y,
            num_estimators=7,
            recipe=recipe,
            generator=torch.Generator().manual_seed(4),
            member_seed=42,
        )
        expected = serial.predict(x[:4])
    with BatchedEnsembleParallel(
        [copy.deepcopy(model) for _ in range(replicas)]
    ) as parallel:
        parallel.fit(
            x,
            y,
            num_estimators=7,
            recipe=recipe,
            estimator_batch_size=batch_size,
            generator=torch.Generator().manual_seed(4),
            member_seed=42,
        )
        actual = parallel.predict(x[:4])
        assert actual.columns == expected.columns
        torch.testing.assert_close(
            actual.numerical, expected.numerical, rtol=1e-5, atol=1e-5
        )
        if classes:
            assert actual.numerical.argmax(-1).equal(
                expected.numerical.argmax(-1)
            )
        if classes > 10:
            assert len(parallel._caches) == 7
