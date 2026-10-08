# ruff: noqa: D103
"""Check explicit churn-positive scoring with reversed prediction columns."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from research.multigpu.relational_bench import (
    archive_prediction_repeats,
    cache_sizes,
    configure_gnn_blocks,
    runtime_environment,
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


def test_gnn_block_size_zero_preserves_native_and_negative_rejected() -> None:
    import pytest

    core = torch.nn.Module()
    core.gnn = torch.nn.Linear(2, 2)
    native = core.gnn
    replicas = [SimpleNamespace(models={"test": core})]
    configure_gnn_blocks(replicas, 0)
    assert core.gnn is native
    with pytest.raises(ValueError, match="nonnegative"):
        configure_gnn_blocks(replicas, -1)


def test_gnn_blocks_installed_on_all_replica_cores() -> None:
    cores = [torch.nn.Module() for _ in range(4)]
    replicas = [
        SimpleNamespace(models={"a": cores[0], "b": cores[1]}),
        SimpleNamespace(models={"a": cores[2], "b": cores[3]}),
    ]
    install = Mock()
    with patch.dict(
        "sys.modules",
        {
            "research.multigpu.blocked_gnn": SimpleNamespace(
                install_blocked_gnn=install
            )
        },
    ):
        configure_gnn_blocks(replicas, 17)
    assert len(install.call_args_list) == 4
    for call, core in zip(install.call_args_list, cores, strict=True):
        assert call.args == (core,)
        assert call.kwargs == {"block_size": 17}


def test_runtime_environment_records_allocator_without_credentials() -> None:
    with patch.dict(
        "os.environ",
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "8",
            "AWS_SECRET_ACCESS_KEY": "must-not-be-collected",
        },
        clear=True,
    ):
        receipt = runtime_environment()
    assert receipt["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert receipt["PYTORCH_CUDA_ALLOC_CONF"] is None
    assert receipt["OMP_NUM_THREADS"] == "8"
    assert "AWS_SECRET_ACCESS_KEY" not in receipt
