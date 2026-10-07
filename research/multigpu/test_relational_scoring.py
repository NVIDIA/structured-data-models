# ruff: noqa: D103
"""Check explicit churn-positive scoring with reversed prediction columns."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from research.multigpu.relational_bench import (
    archive_prediction_repeats,
    cache_sizes,
    score,
)

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


def test_archive_and_quality_use_first_nondeterministic_repeat(
    tmp_path: Path,
) -> None:
    repeats = [
        sdm.TableTensor.from_tensor(values, columns=["1", "0"])
        for values in [
            torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        ]
    ]
    target = sdm.TableTensor.from_columns(
        {"churn": [1, 0]}, stypes={"churn": "categorical"}
    )
    reference = archive_prediction_repeats(repeats, tmp_path)
    archived_pt = torch.load(tmp_path / "predictions.pt", weights_only=False)
    archived_npy = sdm.TableTensor.from_tensor(
        torch.from_numpy(np.load(tmp_path / "predictions.npy")),
        columns=["1", "0"],
    )
    quality = score(reference, target, "classification")
    assert quality["auroc"] == 1.0
    assert score(repeats[-1], target, "classification")["auroc"] == 0.0
    assert quality == score(archived_pt, target, "classification")
    assert quality == score(archived_npy, target, "classification")
    for index, table in enumerate(repeats):
        np.testing.assert_array_equal(
            np.load(tmp_path / f"predictions-repeat-{index}.npy"),
            table.numerical.numpy(),
        )


def test_archive_rejects_missing_repeats(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="At least one"):
        archive_prediction_repeats([], tmp_path)
