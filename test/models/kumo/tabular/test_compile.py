# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("fullgraph", [False, True])
def test_inner_compile_public_workflows(
    monkeypatch: pytest.MonkeyPatch,
    task: Literal["classification", "regression"],
    fullgraph: bool,
) -> None:
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 8,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 4,
            "group_size": 3,
            "num_frequencies": 2,
            "num_readout_tokens": 2,
            "icl_channels": 16,
            "num_icl_layers": 1,
            "num_icl_heads": 2,
            "num_icl_key_value_heads_for_query": None,
        },
    )
    eager = KumoTabular(task=task, size="small", pretrained=False)
    for parameter in eager.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    compiled = copy.deepcopy(eager)
    for inner in compiled.models.values():
        inner.compile(backend="eager", fullgraph=fullgraph, dynamic=True)
    recipe = sp.Recipe(output=[sp.AverageEstimators()])
    try:
        for rows, features in [(6, 3), (8, 5)]:
            context = TableTensor.from_tensor(torch.randn(rows, features))
            values = torch.arange(rows)[:, None] % 3
            target = TableTensor.from_tensor(
                CategoricalTensor.from_tensor(values)
                if task == "classification"
                else torch.linspace(-1, 1, rows)[:, None]
            )
            for model in (eager, compiled):
                model.fit(context, target, recipe=recipe, num_estimators=1)
            for queries in (1, 4):
                query = context[:queries]
                expected = eager.predict(query)
                actual = compiled.predict(query)
                assert actual.columns == expected.columns
                torch.testing.assert_close(
                    actual.numerical, expected.numerical
                )
            expected = eager(context, target, context[:4], recipe=recipe)
            actual = compiled(context, target, context[:4], recipe=recipe)
            torch.testing.assert_close(actual.numerical, expected.numerical)
    finally:
        torch._dynamo.reset()
