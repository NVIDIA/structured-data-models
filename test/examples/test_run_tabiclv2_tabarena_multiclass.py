from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from examples.benchmarking import run_tabiclv2_tabarena_multiclass as runner

pytestmark = pytest.mark.tabarena


def _task_grid() -> list[dict[str, object]]:
    return [
        {
            "dataset": "dataset",
            "task_id": 1,
            "split_index": "r0f0",
            "repeat": 0,
            "fold": 0,
            "sample": 0,
            "metric": "log_loss",
            "problem_type": "multiclass",
            "num_classes": 3,
            "metadata_train_rows": 6,
            "metadata_test_rows": 3,
            "metadata_features": 2,
        }
    ]


def _fake_worker(*, failing_outcome: bool = False):
    def run(
        spec: Mapping[str, Any],
        *,
        scratch_dir: Path,
        worker_timeout_s: float,
    ) -> dict[str, Any]:
        del scratch_dir, worker_timeout_s
        implementation = str(spec["implementation"])
        predictions = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float64,
        )
        if implementation == "sdm_tabicl":
            delta = 0.1 if failing_outcome else 1e-5
            predictions = predictions.copy()
            predictions[np.arange(3), np.arange(3)] -= delta
            predictions[np.arange(3), (np.arange(3) + 1) % 3] += delta
        targets = np.asarray([0, 1, 2], dtype=np.int64)
        labels = ("alpha", "beta", "gamma")
        prediction_path = spec.get("prediction_path")
        prediction_sha256 = None
        if prediction_path is not None:
            path = Path(str(prediction_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                predictions=predictions,
                targets=targets,
                class_labels=np.asarray(labels),
            )
            prediction_sha256 = runner.shared._sha256(path)
        return {
            "profile": spec["profile"],
            "implementation": implementation,
            "trial": spec["trial"],
            "execution_order": spec["execution_order"],
            "seed": spec["seed"],
            "dataset": spec["dataset"],
            "task_id": spec["task_id"],
            "tabarena_split": 0,
            "repeat": spec["repeat"],
            "fold": spec["fold"],
            "sample": spec["sample"],
            "train_rows": 6,
            "test_rows": 3,
            "features": 2,
            "metric": "log_loss",
            "metric_error": runner.local._multiclass_log_loss(
                predictions,
                targets,
            ),
            "train_target_std": None,
            "roc_auc": None,
            "accuracy": runner.local._classification_accuracy(
                predictions,
                targets,
            ),
            "class_labels": labels,
            "prediction_path": prediction_path,
            "prediction_sha256": prediction_sha256,
            "fit_time_s": 1.0,
            "first_inference_time_s": 1.0,
            "inference_times_s": [1.0] * int(spec["inference_repeats"]),
        }

    return run


def _main_args(checkpoint: Path, output_dir: Path) -> list[str]:
    return [
        "--checkpoint-path",
        str(checkpoint),
        "--checkpoint-sha256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "--checkpoint-repository",
        "caller/repository",
        "--checkpoint-revision",
        "caller-revision",
        "--output-dir",
        str(output_dir),
        "--num-cpus",
        "1",
        "--num-gpus",
        "0",
        "--outcome-margin",
        "0.0001",
        "--bootstrap-resamples",
        "100",
    ]


def test_installed_grid_matches_observed_multiclass_scope() -> None:
    full = runner._task_grid_records(subset=runner.MULTICLASS_SUBSET_ALL)
    lite = runner._task_grid_records(subset=runner.MULTICLASS_SUBSET_LITE)

    assert len(full) == 156
    assert len(lite) == 8
    assert Counter(row["dataset"] for row in full) == {
        "anneal": 30,
        "hiva_agnostic": 9,
        "maternal_health_risk": 30,
        "MIC": 30,
        "SDSS17": 9,
        "splice": 9,
        "students_dropout_and_academic_success": 9,
        "website_phishing": 30,
    }
    assert {row["num_classes"] for row in full} == {3, 5, 8}


def test_outcome_summary_requires_all_three_log_loss_gates() -> None:
    rows = [
        {
            "dataset": "a",
            "original_log_loss": 0.2,
            "sdm_log_loss": 0.20005,
            "outcome_gate_passed": True,
        },
        {
            "dataset": "b",
            "original_log_loss": 0.4,
            "sdm_log_loss": 0.40005,
            "outcome_gate_passed": True,
        },
    ]
    summary, passed = runner.summarize_outcomes(
        rows,
        outcome_margin=1e-4,
        bootstrap_resamples=100,
        seed=0,
    )

    assert passed is True
    overall = next(
        row for row in summary if row["scope"] == "all_multiclass_datasets"
    )
    assert overall["bootstrap_upper_95_log_loss_delta"] == pytest.approx(5e-5)


@pytest.mark.parametrize(
    ("failing_outcome", "expected_status", "expected_code"),
    [
        (False, "outcome_qualified", 0),
        (True, "outcome_gate_failed", 2),
    ],
)
def test_main_writes_outcome_only_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_outcome: bool,
    expected_status: str,
    expected_code: int,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "outcome"
    monkeypatch.setattr(
        runner,
        "_task_grid_records",
        lambda **_kwargs: _task_grid(),
    )
    monkeypatch.setattr(
        runner.local,
        "run_worker_subprocess",
        _fake_worker(failing_outcome=failing_outcome),
    )
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(
        runner.shared, "_git_commit", lambda _path: "sdm-commit"
    )
    monkeypatch.setattr(runner, "_tabarena_commit", lambda: "ta-commit")
    monkeypatch.setattr(
        runner.local,
        "environment_manifest",
        lambda **_kwargs: {"test": True},
    )

    assert runner.main(_main_args(checkpoint, output_dir)) == expected_code

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text())
    correctness = (output_dir / runner.CORRECTNESS_FILENAME).read_text()
    report = (output_dir / runner.REPORT_FILENAME).read_text()
    assert manifest["status"] == expected_status
    assert manifest["runtime_qualification"] == {
        "requested": False,
        "claim": "none",
    }
    assert len(manifest["prediction_artifacts"]) == 2
    assert "sdm_accuracy" in correctness
    assert "No runtime or speed claim" in report
    assert not any("timing" in path.name for path in output_dir.iterdir())
