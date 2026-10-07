# ruff: noqa: D103
"""Check explicit churn-positive scoring with reversed prediction columns."""

from types import SimpleNamespace

import torch
from research.multigpu.relational_bench import cache_sizes, score

import sdm
from sdm.cache import Cache


def test_auc_uses_churn_class_one_with_reversed_category_order() -> None:
    pred = sdm.TableTensor.from_tensor(
        torch.tensor([[0.49999997, 0.5], [0.5, 0.5]]),
        columns=["1", "0"],
    )
    target = sdm.TableTensor.from_columns(
        {"churn": [1, 0]}, stypes={"churn": "categorical"}
    )
    metrics = score(pred, target, "classification")
    assert metrics["positive_class"] == 1
    # Class 1 scores have a strict ordering; class 0 scores tie in float32.
    assert metrics["auroc"] == 0.0


def test_cache_measurement_counts_shared_storage_once() -> None:
    values = torch.arange(10, dtype=torch.float32)
    cache = Cache(a=values[:5], b=values[-3:])
    model = SimpleNamespace(_cache=cache)
    wrapper = SimpleNamespace(replicas=[model, model])
    assert cache_sizes([wrapper, model]) == {
        "logical_bytes": {"cpu": 32},
        "storage_bytes": {"cpu": 40},
    }
