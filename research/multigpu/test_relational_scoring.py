# ruff: noqa: D103
"""Check explicit churn-positive scoring with reversed prediction columns."""

import torch
from research.multigpu.relational_bench import score

import sdm


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
