# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from typing import Literal

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS

pytestmark = pytest.mark.skipif(
    "isolate_recompiles" not in inspect.signature(torch.compile).parameters,
    reason="KumoTabular compilation requires per-region recompile isolation",
)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_compile_prediction(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    task: Literal["classification", "regression"],
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
    compiled.compile(backend=backend, fullgraph=True, dynamic=True)

    try:
        for rows, columns in [(7, 3), (9, 5)]:
            x = TableTensor.from_tensor(torch.randn(rows, columns))
            y = TableTensor.from_tensor(torch.randn(rows, 1))
            if task == "classification":
                y = TableTensor.from_tensor(
                    CategoricalTensor.from_tensor(
                        torch.arange(rows)[:, None] % 3
                    )
                )
            for model in (eager, compiled):
                model.fit(
                    x,
                    y,
                    num_estimators=1,
                    generator=torch.Generator().manual_seed(0),
                )
            for queries in (4, 1):
                query = TableTensor.from_tensor(torch.randn(queries, columns))
                expected = eager.predict(query)
                actual = compiled.predict(query)
                assert actual.columns == expected.columns
                torch.testing.assert_close(
                    actual.numerical, expected.numerical, atol=1e-4, rtol=1e-4
                )
    finally:
        torch._dynamo.reset()
