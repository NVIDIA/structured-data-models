from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from examples.benchmarking import run_tabiclv2_tabarena_regression as runner

pytestmark = pytest.mark.tabarena


def _outcome_row(
    *,
    dataset: str,
    original_nrmse: float,
    sdm_nrmse: float,
    margin: float = 1e-4,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "original_nrmse": original_nrmse,
        "sdm_nrmse": sdm_nrmse,
        "outcome_gate_passed": sdm_nrmse - original_nrmse <= margin,
    }


def _timing_rows(
    *,
    sdm_ratio: float,
    calibration_ratio: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in range(3):
        for execution_order, multiplier in enumerate((1.0, calibration_ratio)):
            rows.append(
                {
                    "dataset": "dataset",
                    "task_id": 1,
                    "split_index": "r0f0",
                    "repeat": 0,
                    "fold": 0,
                    "phase": "calibration",
                    "trial": trial,
                    "execution_order": execution_order,
                    "implementation": "original_tabicl",
                    "fit_time_s": multiplier,
                    "first_inference_time_s": multiplier,
                    "warm_inference_time_s": multiplier,
                }
            )
        for implementation, multiplier in (
            ("original_tabicl", 1.0),
            ("sdm_tabicl", sdm_ratio),
        ):
            rows.append(
                {
                    "dataset": "dataset",
                    "task_id": 1,
                    "split_index": "r0f0",
                    "repeat": 0,
                    "fold": 0,
                    "phase": "paired",
                    "trial": trial,
                    "execution_order": 0,
                    "implementation": implementation,
                    "fit_time_s": multiplier,
                    "first_inference_time_s": multiplier,
                    "warm_inference_time_s": multiplier,
                }
            )
    return rows


def _task_grid() -> list[dict[str, object]]:
    return [
        {
            "dataset": "dataset",
            "task_id": 1,
            "split_index": "r0f0",
            "repeat": 0,
            "fold": 0,
            "sample": 0,
            "metric": "rmse",
            "problem_type": "regression",
            "metadata_train_rows": 3,
            "metadata_test_rows": 2,
            "metadata_features": 1,
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
        predictions = np.array([0.0, 1.0], dtype=np.float64)
        if implementation == "sdm_tabicl":
            predictions = predictions + (0.1 if failing_outcome else 5e-5)
        targets = np.array([0.0, 1.0], dtype=np.float64)
        prediction_path = spec.get("prediction_path")
        prediction_sha256 = None
        if prediction_path is not None:
            path = Path(str(prediction_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                predictions=predictions,
                targets=targets,
                train_target_std=np.array([1.0], dtype=np.float64),
            )
            prediction_sha256 = runner._sha256(path)
        multiplier = 1.0
        trial_name = str(spec["trial"])
        if trial_name.startswith("calibration"):
            multiplier = 1.001 if int(spec["execution_order"]) else 1.0
        if trial_name.startswith("timing") and implementation == "sdm_tabicl":
            multiplier = 1.004
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
            "train_rows": 3,
            "test_rows": 2,
            "features": 1,
            "metric": "rmse",
            "metric_error": float(
                np.sqrt(np.mean(np.square(predictions - targets)))
            ),
            "train_target_std": 1.0,
            "prediction_path": prediction_path,
            "prediction_sha256": prediction_sha256,
            "fit_time_s": multiplier,
            "first_inference_time_s": multiplier,
            "inference_times_s": [multiplier],
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
        "--bootstrap-resamples",
        "100",
    ]


def test_task_grid_fingerprint_is_order_sensitive_and_stable() -> None:
    grid = _task_grid()
    assert runner.task_grid_fingerprint(grid) == runner.task_grid_fingerprint(
        grid
    )
    changed = [dict(grid[0], fold=1)]
    assert runner.task_grid_fingerprint(grid) != runner.task_grid_fingerprint(
        changed
    )


def test_outcome_summary_requires_split_dataset_and_bootstrap_gates() -> None:
    passing = [
        _outcome_row(dataset="a", original_nrmse=0.2, sdm_nrmse=0.20005),
        _outcome_row(dataset="b", original_nrmse=0.4, sdm_nrmse=0.40005),
    ]
    summary, passed = runner.summarize_outcomes(
        passing,
        outcome_margin=1e-4,
        bootstrap_resamples=100,
        seed=0,
    )
    assert passed is True
    overall = next(
        row for row in summary if row["scope"] == "all_regression_datasets"
    )
    assert overall["bootstrap_upper_95_nrmse_delta"] == pytest.approx(5e-5)

    failing = [
        _outcome_row(dataset="a", original_nrmse=0.2, sdm_nrmse=0.2002),
        _outcome_row(dataset="b", original_nrmse=0.4, sdm_nrmse=0.40005),
    ]
    _, passed = runner.summarize_outcomes(
        failing,
        outcome_margin=1e-4,
        bootstrap_resamples=100,
        seed=0,
    )
    assert passed is False


def test_timing_summary_calibrates_noise_and_caps_slowdown() -> None:
    summary, passed = runner.summarize_timing(
        _timing_rows(sdm_ratio=1.004, calibration_ratio=1.001),
        timing_trials=3,
        calibration_trials=3,
        bootstrap_resamples=100,
        seed=0,
    )
    assert passed is True
    assert all(
        row["allowed_slowdown_ratio"] == pytest.approx(1.005)
        for row in summary
    )

    _, passed = runner.summarize_timing(
        _timing_rows(sdm_ratio=1.01, calibration_ratio=1.001),
        timing_trials=3,
        calibration_trials=3,
        bootstrap_resamples=100,
        seed=0,
    )
    assert passed is False


def test_main_writes_complete_outcome_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "outcome"
    monkeypatch.setattr(
        runner, "_task_grid_records", lambda **_kwargs: _task_grid()
    )
    monkeypatch.setattr(runner.local, "run_worker_subprocess", _fake_worker())
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(runner, "_tabarena_commit", lambda: "ta-commit")
    monkeypatch.setattr(
        runner.local,
        "environment_manifest",
        lambda **_kwargs: {"test": True},
    )

    assert runner.main(_main_args(checkpoint, output_dir)) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text())
    correctness = (output_dir / runner.CORRECTNESS_FILENAME).read_text()
    report = (output_dir / runner.REPORT_FILENAME).read_text()
    assert manifest["status"] == "outcome_qualified"
    assert manifest["outcome_gate"]["passed"] is True
    snapshot_path = output_dir / runner.SOURCE_SNAPSHOT_FILENAME
    snapshot = json.loads(snapshot_path.read_text())
    assert manifest["source_snapshot"]["sha256"] == runner._sha256(
        snapshot_path
    )
    assert snapshot["base_commit"] == "sdm-commit"
    assert snapshot["tracked_diff"]["path"] == runner.SOURCE_PATCH_FILENAME
    assert (output_dir / runner.SOURCE_PATCH_FILENAME).is_file()
    assert (
        output_dir
        / runner.SOURCE_SNAPSHOT_DIRECTORY
        / "examples/benchmarking/run_tabiclv2_tabarena_regression.py"
    ).is_file()
    assert len(manifest["prediction_artifacts"]) == 2
    assert "sdm_nrmse" in correctness
    assert "Not run" in report
    assert (output_dir / ".workers").is_dir()


def test_main_blocks_timing_when_outcome_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "failing"
    monkeypatch.setattr(
        runner, "_task_grid_records", lambda **_kwargs: _task_grid()
    )
    monkeypatch.setattr(
        runner.local,
        "run_worker_subprocess",
        _fake_worker(failing_outcome=True),
    )
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(runner, "_tabarena_commit", lambda: "ta-commit")
    monkeypatch.setattr(
        runner.local,
        "environment_manifest",
        lambda **_kwargs: {"test": True},
    )

    args = [*_main_args(checkpoint, output_dir), "--run-timing"]
    assert runner.main(args) == 2

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "outcome_gate_failed"
    assert not (output_dir / runner.TIMING_TRIALS_FILENAME).exists()


def test_main_runs_timing_only_after_passing_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "timing"
    monkeypatch.setattr(
        runner, "_task_grid_records", lambda **_kwargs: _task_grid()
    )
    monkeypatch.setattr(runner.local, "run_worker_subprocess", _fake_worker())
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(runner, "_tabarena_commit", lambda: "ta-commit")
    monkeypatch.setattr(
        runner.local,
        "environment_manifest",
        lambda **_kwargs: {"test": True},
    )

    args = [
        *_main_args(checkpoint, output_dir),
        "--run-timing",
        "--timing-trials",
        "3",
        "--timing-calibration-trials",
        "3",
        "--warm-inference-repeats",
        "1",
    ]
    assert runner.main(args) == 0

    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "completed"
    assert manifest["timing_gate"]["passed"] is True
    assert (output_dir / runner.TIMING_TRIALS_FILENAME).is_file()
    assert (output_dir / runner.TIMING_SUMMARY_FILENAME).is_file()
