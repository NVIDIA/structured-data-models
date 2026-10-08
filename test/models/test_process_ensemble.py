# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from research.multigpu._test_models import _RandomCacheModel
from research.multigpu.process_ensemble import ProcessEnsembleParallel

from sdm import RelatedTables, RelationalData, TableTensor
from sdm.models import EnsembleParallel, KumoRelational
from sdm.models.kumo.relational.model import _KumoRelational


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


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_process_relational_tables_and_two_hop_cache(
    relational_data: RelationalData, task: str
) -> None:
    model = KumoRelational(task=task, pretrained=False, device="meta")
    for key in model.models:
        model.models[key] = _KumoRelational(
            num_classes=10 if task == "classification" else 0,
            num_quantiles=999 if task == "regression" else 0,
            channels=16,
            num_embedding_layers=1,
            num_embedding_heads=4,
            num_inducing_points=8,
            num_readout_tokens=2,
            num_icl_layers=2,
            num_icl_heads=4,
        )
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.1)
    model.eval()
    x = TableTensor.from_columns(
        {"user_id": [0, 1, 2, 3]}, stypes={"user_id": "id"}
    )
    y = TableTensor.from_columns(
        {"target": [0, 1, 0, 1]},
        stypes={
            "target": "categorical"
            if task == "classification"
            else "numerical"
        },
    )
    related = RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )
    with EnsembleParallel([copy.deepcopy(model)]) as reference:
        reference.fit(
            x,
            y,
            related,
            num_estimators=3,
            num_hops=2,
            generator=torch.Generator().manual_seed(1729),
            member_seed=1729,
        )
        expected = reference.predict(x[:3], related)
    parallel = ProcessEnsembleParallel(
        [copy.deepcopy(model), copy.deepcopy(model)]
    )
    try:
        parallel.fit(
            x,
            y,
            related,
            num_estimators=3,
            num_hops=2,
            generator=torch.Generator().manual_seed(1729),
            member_seed=1729,
        )
        for _ in range(2):
            actual = parallel.predict(x[:3], related)
            assert actual.columns == expected.columns
            torch.testing.assert_close(
                actual.numerical, expected.numerical, rtol=0, atol=0
            )
    finally:
        parallel.close()
