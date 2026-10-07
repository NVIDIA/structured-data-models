# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from research.multigpu.graph_ensemble import GraphEnsembleParallel

from sdm import CategoricalTensor, Recipe, TableTensor
from sdm.models import EnsembleParallel, KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


@pytest.mark.parametrize("replicas", [1, 2])
def test_graph_replay_shape_and_output_lifetime(
    replicas: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.device_count() < replicas:
        pytest.skip("needs CUDA devices for every replica")
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 16,
            "num_embedding_layers": 1,
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
    x = torch.randn(24, 6, device="cuda:0")
    y = TableTensor(
        categorical=CategoricalTensor.from_tensor(
            torch.arange(24, device="cuda:0").remainder(7).view(-1, 1)
        )
    )
    with (
        EnsembleParallel([copy.deepcopy(model).cuda(0)]) as reference,
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        reference.fit(x, y, num_estimators=4, recipe=Recipe())
        expected = [
            reference.predict(query) for query in (x[:4], x[4:8], x[:3])
        ]
    with (
        GraphEnsembleParallel(
            [copy.deepcopy(model).cuda(i) for i in range(replicas)]
        ) as graph,
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        graph.fit(x, y, num_estimators=4, recipe=Recipe())
        actual = [graph.predict(query) for query in (x[:4], x[4:8], x[:3])]
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                a.numerical, b.numerical, rtol=0, atol=0
            )
        assert graph.graph_count == 8
        graph.predict(x[:4])
        torch.testing.assert_close(
            actual[0].numerical, expected[0].numerical, rtol=0, atol=0
        )
        assert graph.graph_count == 8
        graph.fit(x, y, num_estimators=4, recipe=Recipe())
        assert graph.graph_count == 0
