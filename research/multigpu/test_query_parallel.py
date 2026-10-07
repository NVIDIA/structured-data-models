# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts; GPU/quality claims require the separate measured runs."""
# ruff: noqa: D101, D102, D103, TID253

import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from research.multigpu.data_parallel_adapter import (
    DataParallelAdapter,
    hybrid_factory,
)
from research.multigpu.process_factories import TabularProcessFactory
from research.multigpu.query_parallel import (
    ProcessQueryParallel,
    QueryBatch,
    QueryParallel,
)

from sdm import RelatedTables, TableTensor
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


class Echo:
    def __init__(self) -> None:
        self.active = False

    def predict(self, x, related_tables=None):
        assert torch.is_inference_mode_enabled()
        assert not self.active, "Concurrent entry into one fitted model"
        self.active = True
        try:
            time.sleep(0.002)
            out = x.numerical.clone()
            if related_tables is not None:
                out += related_tables.tables["orders"].numerical.sum()
            return TableTensor.from_tensor(out)
        finally:
            self.active = False


def echo_factory(worker, device):
    return Echo()


def batches():
    return [
        QueryBatch(
            tuple(range(start, start + size)),
            TableTensor.from_columns(
                {
                    "value": list(range(start, start + size)),
                    "kind": ["a"] * size,
                },
                stypes={"value": "numerical", "kind": "categorical"},
            ),
            RelatedTables(
                tables={
                    "orders": TableTensor.from_tensor(
                        torch.full((size + 2, 1), float(start))
                    )
                },
                relationships=[],
                task_links=[],
            ),
        )
        for start, size in [(12, 3), (3, 1), (91, 4), (2, 2), (7, 1)]
    ]


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_irregular_whole_graph_batches_and_order(workers):
    query = batches()
    with QueryParallel(
        [Echo() for _ in range(workers)], ["cpu"] * workers
    ) as executor:
        result = executor.predict(query)
    assert [r.row_ids for r in result] == [b.row_ids for b in query]
    with torch.inference_mode():
        for batch, item in zip(query, result, strict=True):
            torch.testing.assert_close(
                item.prediction.numerical,
                Echo().predict(batch.x, batch.related_tables).numerical,
            )


def test_concurrent_requests_serialize_each_replica():
    with QueryParallel([Echo()], ["cpu"]) as executor:
        submitted = []
        threads = [
            threading.Thread(
                target=lambda: submitted.extend(
                    executor.submit(b, 0) for b in batches()
                )
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len([future.result() for future in submitted]) == 20


def test_invalid_batch_and_shared_model():
    model = Echo()
    with pytest.raises(ValueError, match="independent"):
        QueryParallel([model, model], ["cpu", "cpu"])
    with QueryParallel([model], ["cpu"]) as executor:
        with pytest.raises(ValueError, match="row count"):
            executor.predict([QueryBatch((1,), batches()[0].x)])
        assert executor.predict([]) == []


def test_spawn_process_roundtrip():
    query = batches()
    with ProcessQueryParallel(echo_factory, ["cpu", "cpu"]) as executor:
        executor.ready()
        memory = executor.memory(reset_peak=True)
        assert len({item["pid"] for item in memory}) == 2
        result = executor.predict(query)
    assert [r.row_ids for r in result] == [b.row_ids for b in query]
    with torch.inference_mode():
        for batch, item in zip(query, result, strict=True):
            torch.testing.assert_close(
                item.prediction.numerical,
                Echo().predict(batch.x, batch.related_tables).numerical,
            )


@pytest.mark.parametrize("adapter", [False, True, "hybrid"])
def test_real_kumotabular_fitted_recipe_parity(monkeypatch, adapter):
    torch.set_num_threads(1)
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 16,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 4,
            "group_size": 3,
            "num_frequencies": 4,
            "num_readout_tokens": 2,
            "icl_channels": 32,
            "num_icl_layers": 1,
            "num_icl_heads": 2,
            "num_icl_key_value_heads_for_query": None,
        },
    )
    base = KumoTabular(task="classification", size="small", pretrained=False)
    for parameter in base.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    models = [
        copy.deepcopy(base) for _ in range(4 if adapter == "hybrid" else 2)
    ]
    context = TableTensor.from_columns(
        {"n": list(range(24)), "c": ["a", "b", "c"] * 8},
        stypes={"n": "numerical", "c": "categorical"},
    )
    target = TableTensor.from_columns(
        {"label": [False, True] * 12}, stypes={"label": "categorical"}
    )
    for model in models:
        model.fit(
            context,
            target,
            num_estimators=2,
            generator=torch.Generator().manual_seed(123),
        )
    query = [
        QueryBatch(tuple(range(i, i + count)), context[i : i + count])
        for i, count in [(2, 4), (13, 1), (7, 3)]
    ]
    reference = [models[0].predict(batch.x) for batch in query]
    if adapter:
        if adapter == "hybrid":
            executor = hybrid_factory(
                SimpleNamespace(seed=123, precision="float32"), models
            )
        else:
            executor = DataParallelAdapter(
                models, devices=[torch.device("cpu")] * 2, dtype=None
            )
        executor.fit(
            context,
            target,
            num_estimators=2,
            generator=torch.Generator().manual_seed(123),
        )
        result = executor.predict_batches(query)
        executor.close()
    else:
        with QueryParallel(models, ["cpu", "cpu"]) as executor:
            result = [item.prediction for item in executor.predict(query)]
    for actual, expected in zip(result, reference, strict=True):
        torch.testing.assert_close(
            actual.numerical, expected.numerical, rtol=0, atol=0
        )
        assert actual.columns == expected.columns
    assert reference[0].numerical.std() > 0


def test_tabular_process_factory_reproduces_weights_and_fit(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 16,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 4,
            "group_size": 3,
            "num_frequencies": 4,
            "num_readout_tokens": 2,
            "icl_channels": 32,
            "num_icl_layers": 1,
            "num_icl_heads": 2,
            "num_icl_key_value_heads_for_query": None,
        },
    )
    x = np.arange(72, dtype=np.float32).reshape(24, 3)
    y = np.arange(24) % 2
    np.save(tmp_path / "x_train.npy", x)
    np.save(tmp_path / "y_train.npy", y)
    factory = TabularProcessFactory(
        data=tmp_path,
        context=24,
        estimators=2,
        pretrained=False,
        precision="float32",
    )
    first = factory(0, torch.device("cpu"))
    second = factory(3, torch.device("cpu"))
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    query = TableTensor.from_tensor(torch.from_numpy(x[:3]))
    torch.testing.assert_close(
        first.predict(query).numerical,
        second.predict(query).numerical,
        rtol=0,
        atol=0,
    )
