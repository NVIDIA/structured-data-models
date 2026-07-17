"""Qualify paired SDM and original TabICLv2 on TabArena regression splits.

The default run discovers every regression split from the installed pinned
TabArena metadata. It writes immutable prediction artifacts and applies the
outcome gate before optionally running the much more expensive timing gate.
All work stays local to this repository; the sibling TabArena checkout is read
only and no results are uploaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from examples.benchmarking.run_tabiclv2_tabarena_smoke import (
    _git_commit,
    _tabarena_commit,
    _validate_checkpoint,
    _validate_provenance,
    _validate_seed,
)

DEFAULT_OUTCOME_MARGIN = 1e-4
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_TIMING_TRIALS = 30
DEFAULT_TIMING_CALIBRATION_TRIALS = 3
DEFAULT_WARM_INFERENCE_REPEATS = 20
DEFAULT_WORKER_TIMEOUT_S = 1_800.0
REGRESSION_SUBSET_ALL = "all"
REGRESSION_SUBSET_LITE = "lite"
MATCHED_PARITY = "matched_parity"
ORIGINAL = "original_tabicl"
SDM = "sdm_tabicl"
IMPLEMENTATIONS = (ORIGINAL, SDM)

TASK_GRID_FILENAME = "regression_task_grid.json"
CORRECTNESS_FILENAME = "regression_correctness.csv"
OUTCOME_SUMMARY_FILENAME = "regression_outcome_summary.csv"
TIMING_TRIALS_FILENAME = "regression_timing_trials.csv"
TIMING_SUMMARY_FILENAME = "regression_timing_summary.csv"
REPORT_FILENAME = "regression_qualification_report.md"
MANIFEST_FILENAME = "manifest.json"
PREDICTIONS_DIRNAME = "predictions"
SOURCE_SNAPSHOT_FILENAME = "source_snapshot.json"
SOURCE_PATCH_FILENAME = "source_snapshot.patch"
SOURCE_SNAPSHOT_DIRECTORY = "source_snapshot"
RUNTIME_SOURCE_PATHS = (
    "examples/benchmarking/run_tabiclv2_tabarena_regression.py",
    "examples/benchmarking/run_tabiclv2_local_comparison.py",
    "examples/benchmarking/run_tabiclv2_tabarena_smoke.py",
    "examples/benchmarking/tabiclv2_tabarena_model.py",
    "sdm/models/base.py",
    "sdm/models/tabiclv2/__init__.py",
    "sdm/models/tabiclv2/model.py",
    "sdm/models/tabiclv2/output.py",
    "sdm/models/tabiclv2/recipe.py",
    "sdm/models/tabiclv2/row_embedding.py",
    "sdm/processing/categorical_align.py",
    "sdm/tensor/categorical.py",
)

CORRECTNESS_COLUMNS = (
    "dataset",
    "task_id",
    "split_index",
    "tabarena_split",
    "repeat",
    "fold",
    "sample",
    "train_rows",
    "test_rows",
    "features",
    "original_rmse",
    "sdm_rmse",
    "train_target_std",
    "original_nrmse",
    "sdm_nrmse",
    "nrmse_delta_sdm_minus_original",
    "prediction_rmse_sdm_vs_original",
    "prediction_max_abs_delta",
    "outcome_margin",
    "outcome_gate_passed",
    "original_prediction_path",
    "original_prediction_sha256",
    "sdm_prediction_path",
    "sdm_prediction_sha256",
)
OUTCOME_SUMMARY_COLUMNS = (
    "scope",
    "dataset",
    "splits",
    "mean_original_nrmse",
    "mean_sdm_nrmse",
    "mean_nrmse_delta_sdm_minus_original",
    "outcome_margin",
    "outcome_gate_passed",
    "bootstrap_resamples",
    "bootstrap_upper_95_nrmse_delta",
)
TIMING_TRIAL_COLUMNS = (
    "dataset",
    "task_id",
    "split_index",
    "repeat",
    "fold",
    "phase",
    "trial",
    "execution_order",
    "implementation",
    "fit_time_s",
    "first_inference_time_s",
    "warm_inference_time_s",
)
TIMING_SUMMARY_COLUMNS = (
    "dataset",
    "task_id",
    "split_index",
    "repeat",
    "fold",
    "timing_metric",
    "timing_trials",
    "calibration_trials",
    "original_vs_original_geomean_ratio",
    "original_vs_original_upper_95_ratio",
    "allowed_slowdown_ratio",
    "sdm_vs_original_geomean_ratio",
    "sdm_vs_original_upper_95_ratio",
    "hardware_qualified",
    "timing_gate_passed",
)


class _QualificationInterrupted(Exception):
    """Represent a termination signal inside normal control flow."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"Qualification interrupted by signal {signum}.")
        self.signum = signum


def _termination_handler(signum: int, frame: Any) -> None:
    """Convert SIGTERM into an exception so the manifest can be finalized."""
    del frame

    raise _QualificationInterrupted(signum)


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_inventory(root: Path) -> dict[str, object]:
    """Return a content-addressed inventory of one runtime source tree."""
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Runtime source tree is missing: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"Runtime source tree is empty: {root}")
    return {"root": str(root), "files": files}


def _tabarena_package_root() -> Path:
    tabarena = importlib.import_module("tabarena")
    module_file = getattr(tabarena, "__file__", None)
    if module_file is None:
        raise RuntimeError("Cannot locate the imported TabArena package.")
    return Path(module_file).resolve().parent


def _runtime_source_state(repository_root: Path) -> dict[str, object]:
    """Fingerprint every SDM and TabArena file that may affect a run."""
    return {
        "schema_version": 1,
        "sdm": {
            "commit": _git_commit(repository_root),
            "trees": [
                _source_tree_inventory(repository_root / "sdm"),
                _source_tree_inventory(repository_root / "examples"),
            ],
        },
        "tabarena": {
            "commit": _tabarena_commit(),
            "trees": [_source_tree_inventory(_tabarena_package_root())],
        },
    }


def _assert_runtime_source_state(
    expected: Mapping[str, object],
    repository_root: Path,
) -> None:
    """Fail when code or packaged metadata changed after run start."""
    current = _runtime_source_state(repository_root)
    if current == expected:
        return
    changed = [
        name
        for name in ("sdm", "tabarena")
        if current.get(name) != expected.get(name)
    ]
    detail = ", ".join(changed) if changed else "runtime sources"
    raise RuntimeError(
        f"Runtime source drift detected in {detail}; refusing mixed-source "
        "qualification results. Start a new run after source changes finish."
    )


def _write_source_snapshot(
    repository_root: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Write the exact runtime-source and working-tree snapshot for a run."""
    runtime_source_state = _runtime_source_state(repository_root)
    snapshot_directory = output_dir / SOURCE_SNAPSHOT_DIRECTORY
    snapshot_directory.mkdir()
    runtime_sources: list[dict[str, str]] = []
    for relative_path in RUNTIME_SOURCE_PATHS:
        source_path = repository_root / relative_path
        if not source_path.is_file():
            raise RuntimeError(
                f"Runtime source is missing from snapshot: {relative_path}"
            )
        snapshot_path = snapshot_directory / relative_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(source_path.read_bytes())
        runtime_sources.append(
            {
                "path": relative_path,
                "snapshot_path": str(snapshot_path.relative_to(output_dir)),
                "sha256": _sha256(snapshot_path),
            }
        )

    patch_path = output_dir / SOURCE_PATCH_FILENAME
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    patch_path.write_text(patch)
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    snapshot_path = output_dir / SOURCE_SNAPSHOT_FILENAME
    _write_json(
        snapshot_path,
        {
            "schema_version": 1,
            "base_commit": _git_commit(repository_root),
            "git_status_short": status,
            "tracked_diff": {
                "path": SOURCE_PATCH_FILENAME,
                "sha256": _sha256(patch_path),
            },
            "runtime_sources": runtime_sources,
            "runtime_source_guard": runtime_source_state,
        },
    )
    return snapshot_path, patch_path, runtime_source_state


def _ensure_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_positive_int(value: int, *, option: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{option} must be an integer of at least 1.")
    return value


def _validate_non_negative_float(value: float, *, option: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{option} must be finite and non-negative.")
    return value


def _validate_timeout(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--worker-timeout-s must be finite and positive.")
    return value


def _local_comparison_module() -> Any:
    """Load optional benchmark integrations only when executing a run."""
    from examples.benchmarking import (
        run_tabiclv2_local_comparison as local_runtime,
    )

    return local_runtime


def _device_allocation(num_gpus: int) -> Any:
    """Resolve devices without importing AutoGluon during test collection."""
    from examples.benchmarking.tabiclv2_tabarena_model import DeviceAllocation

    return DeviceAllocation.from_num_gpus(num_gpus)


def _execution_order(job_index: int, trial_index: int) -> tuple[str, str]:
    """Alternate implementation order to reduce systematic order bias."""
    if (job_index + trial_index) % 2 == 0:
        return IMPLEMENTATIONS
    return IMPLEMENTATIONS[1], IMPLEMENTATIONS[0]


def _task_grid_records(*, subset: str) -> list[dict[str, object]]:
    """Discover one immutable record for every selected regression split."""
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str

    collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
    filters: dict[str, object] = {"problem_types": ["regression"]}
    if subset == REGRESSION_SUBSET_LITE:
        filters["split_indices"] = "lite"
    elif subset != REGRESSION_SUBSET_ALL:
        raise ValueError(f"Unknown regression subset: {subset!r}.")
    collection = collection.subset_tasks(**filters)

    records: list[dict[str, object]] = []
    for task in collection:
        if task.problem_type != "regression" or task.eval_metric != "rmse":
            raise RuntimeError("Regression grid contains a non-RMSE task.")
        if task.task_id_str is None or task.tabarena_task_name is None:
            raise RuntimeError(
                "Regression task metadata is missing an identity."
            )
        for split in task.splits_metadata.values():
            records.append(
                {
                    "dataset": task.tabarena_task_name,
                    "task_id": tid_from_task_id_str(task.task_id_str),
                    "split_index": split.split_index,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "sample": 0,
                    "metric": task.eval_metric,
                    "problem_type": task.problem_type,
                    "metadata_train_rows": split.num_instances_train,
                    "metadata_test_rows": split.num_instances_test,
                    "metadata_features": task.num_features,
                }
            )
    records.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["repeat"]),
            int(row["fold"]),
        )
    )
    if not records:
        raise RuntimeError("Regression task discovery returned no splits.")
    identities = {
        (row["task_id"], row["repeat"], row["fold"]) for row in records
    }
    if len(identities) != len(records):
        raise RuntimeError("Regression task grid contains duplicate splits.")
    return records


def task_grid_fingerprint(task_grid: Sequence[Mapping[str, Any]]) -> str:
    """Return the SHA-256 of a canonical discovered task grid."""
    serialized = json.dumps(
        list(task_grid),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _artifact_stem(task: Mapping[str, Any], implementation: str) -> str:
    dataset_digest = hashlib.sha256(str(task["dataset"]).encode()).hexdigest()[
        :12
    ]
    task_id = int(task["task_id"])
    split_index = str(task["split_index"])
    return f"{task_id}-{split_index}-{dataset_digest}-{implementation}.npz"


def _base_worker_spec(
    task: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    num_cpus: int,
    num_gpus: int,
    inference_repeats: int,
) -> dict[str, object]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "task_id": int(task["task_id"]),
        "dataset": str(task["dataset"]),
        "repeat": int(task["repeat"]),
        "fold": int(task["fold"]),
        "sample": int(task["sample"]),
        "inference_repeats": inference_repeats,
        "profile": MATCHED_PARITY,
    }


def _validate_task_identity(
    result: Mapping[str, Any],
    task: Mapping[str, Any],
) -> None:
    expected = {
        "dataset": task["dataset"],
        "task_id": task["task_id"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "sample": task["sample"],
        "metric": "rmse",
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(
                f"Worker result {field!r} does not match the frozen task grid."
            )


def _load_prediction_artifact(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, Path, str]:
    path_value = result.get("prediction_path")
    sha256 = result.get("prediction_sha256")
    if not isinstance(path_value, str) or not isinstance(sha256, str):
        raise RuntimeError(
            "Correctness worker did not return a prediction artifact."
        )
    path = Path(path_value)
    if not path.is_file() or _sha256(path) != sha256:
        raise RuntimeError(
            "Prediction artifact does not match its worker digest."
        )
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {
            "predictions",
            "targets",
            "train_target_std",
        }:
            raise RuntimeError("Prediction artifact has an unexpected schema.")
        predictions = np.asarray(artifact["predictions"], dtype=np.float64)
        targets = np.asarray(artifact["targets"], dtype=np.float64)
        train_target_std = np.asarray(
            artifact["train_target_std"],
            dtype=np.float64,
        )
    if predictions.ndim != 1 or targets.shape != predictions.shape:
        raise RuntimeError(
            "Prediction artifact does not contain paired vectors."
        )
    if train_target_std.shape != (1,):
        raise RuntimeError("Prediction artifact has an invalid target scale.")
    scale = float(train_target_std[0])
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise RuntimeError("Prediction artifact contains non-finite values.")
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            "NRMSE is undefined because the training target standard "
            "deviation is not positive."
        )
    return predictions, targets, scale, path, sha256


def _rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(predictions - targets))))
    if not math.isfinite(value):
        raise RuntimeError("RMSE must be finite.")
    return value


def _correctness_row(
    task: Mapping[str, Any],
    *,
    original: Mapping[str, Any],
    sdm: Mapping[str, Any],
    outcome_margin: float,
    output_dir: Path,
) -> dict[str, object]:
    (
        original_predictions,
        original_targets,
        original_scale,
        original_path,
        original_sha,
    ) = _load_prediction_artifact(original)
    sdm_predictions, sdm_targets, sdm_scale, sdm_path, sdm_sha = (
        _load_prediction_artifact(sdm)
    )
    if not np.array_equal(original_targets, sdm_targets):
        raise RuntimeError(
            "Paired implementations used different test targets."
        )
    if not math.isclose(original_scale, sdm_scale, rel_tol=0, abs_tol=0):
        raise RuntimeError(
            "Paired implementations used different target scales."
        )
    original_rmse = _rmse(original_predictions, original_targets)
    sdm_rmse = _rmse(sdm_predictions, sdm_targets)
    if not math.isclose(
        original_rmse,
        float(original["metric_error"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Original worker RMSE disagrees with its artifact.")
    if not math.isclose(
        sdm_rmse,
        float(sdm["metric_error"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("SDM worker RMSE disagrees with its artifact.")
    original_nrmse = original_rmse / original_scale
    sdm_nrmse = sdm_rmse / original_scale
    delta = sdm_nrmse - original_nrmse
    return {
        "dataset": task["dataset"],
        "task_id": task["task_id"],
        "split_index": task["split_index"],
        "tabarena_split": original["tabarena_split"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "sample": task["sample"],
        "train_rows": original["train_rows"],
        "test_rows": original["test_rows"],
        "features": original["features"],
        "original_rmse": original_rmse,
        "sdm_rmse": sdm_rmse,
        "train_target_std": original_scale,
        "original_nrmse": original_nrmse,
        "sdm_nrmse": sdm_nrmse,
        "nrmse_delta_sdm_minus_original": delta,
        "prediction_rmse_sdm_vs_original": _rmse(
            sdm_predictions,
            original_predictions,
        ),
        "prediction_max_abs_delta": float(
            np.max(np.abs(sdm_predictions - original_predictions))
        ),
        "outcome_margin": outcome_margin,
        "outcome_gate_passed": delta <= outcome_margin,
        "original_prediction_path": str(original_path.relative_to(output_dir)),
        "original_prediction_sha256": original_sha,
        "sdm_prediction_path": str(sdm_path.relative_to(output_dir)),
        "sdm_prediction_sha256": sdm_sha,
    }


def run_correctness_qualification(
    task_grid: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    num_cpus: int,
    num_gpus: int,
    outcome_margin: float,
    output_dir: Path,
    scratch_dir: Path,
    worker_timeout_s: float,
    source_guard: Callable[[], None],
) -> list[dict[str, object]]:
    """Run one alternating paired prediction capture for every frozen split."""
    predictions_dir = output_dir / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(exist_ok=False)
    rows: list[dict[str, object]] = []
    for job_index, task in enumerate(task_grid):
        results: dict[str, dict[str, Any]] = {}
        for order_index, implementation in enumerate(
            _execution_order(job_index, 0)
        ):
            artifact_path = predictions_dir / _artifact_stem(
                task, implementation
            )
            spec = {
                **_base_worker_spec(
                    task,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                    seed=seed,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    inference_repeats=1,
                ),
                "implementation": implementation,
                "trial": f"correctness_{job_index}",
                "execution_order": order_index,
                "prediction_path": str(artifact_path),
            }
            source_guard()
            result = _local_comparison_module().run_worker_subprocess(
                spec,
                scratch_dir=scratch_dir,
                worker_timeout_s=worker_timeout_s,
            )
            source_guard()
            _validate_task_identity(result, task)
            results[implementation] = result
        rows.append(
            _correctness_row(
                task,
                original=results[ORIGINAL],
                sdm=results[SDM],
                outcome_margin=outcome_margin,
                output_dir=output_dir,
            )
        )
    if len(rows) != len(task_grid):
        raise RuntimeError(
            "Correctness qualification did not cover every split."
        )
    return rows


def _bootstrap_upper_95(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError("Bootstrap requires at least one finite value.")
    if not np.isfinite(values).all():
        raise RuntimeError("Bootstrap values must be finite.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        low=0,
        high=values.size,
        size=(resamples, values.size),
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.95))


def summarize_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_margin: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], bool]:
    """Summarize split and dataset NRMSE gates with a dataset bootstrap."""
    if not rows:
        raise RuntimeError("Cannot summarize an empty correctness result.")
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    summary: list[dict[str, object]] = []
    dataset_deltas: list[float] = []
    dataset_original_nrmse: list[float] = []
    dataset_sdm_nrmse: list[float] = []
    all_split_passed = all(bool(row["outcome_gate_passed"]) for row in rows)
    all_dataset_passed = True
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        original = np.asarray(
            [float(row["original_nrmse"]) for row in dataset_rows],
            dtype=np.float64,
        )
        sdm = np.asarray(
            [float(row["sdm_nrmse"]) for row in dataset_rows],
            dtype=np.float64,
        )
        mean_original = float(np.mean(original))
        mean_sdm = float(np.mean(sdm))
        delta = mean_sdm - mean_original
        dataset_original_nrmse.append(mean_original)
        dataset_sdm_nrmse.append(mean_sdm)
        dataset_deltas.append(delta)
        passed = delta <= outcome_margin
        all_dataset_passed = all_dataset_passed and passed
        summary.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "splits": len(dataset_rows),
                "mean_original_nrmse": mean_original,
                "mean_sdm_nrmse": mean_sdm,
                "mean_nrmse_delta_sdm_minus_original": delta,
                "outcome_margin": outcome_margin,
                "outcome_gate_passed": passed,
                "bootstrap_resamples": None,
                "bootstrap_upper_95_nrmse_delta": None,
            }
        )
    dataset_delta_array = np.asarray(dataset_deltas, dtype=np.float64)
    bootstrap_upper = _bootstrap_upper_95(
        dataset_delta_array,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    overall_original = float(np.mean(dataset_original_nrmse))
    overall_sdm = float(np.mean(dataset_sdm_nrmse))
    overall_delta = overall_sdm - overall_original
    aggregate_passed = bootstrap_upper <= outcome_margin
    summary.append(
        {
            "scope": "all_regression_datasets",
            "dataset": "all",
            "splits": len(rows),
            "mean_original_nrmse": overall_original,
            "mean_sdm_nrmse": overall_sdm,
            "mean_nrmse_delta_sdm_minus_original": overall_delta,
            "outcome_margin": outcome_margin,
            "outcome_gate_passed": aggregate_passed,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_upper_95_nrmse_delta": bootstrap_upper,
        }
    )
    return (
        summary,
        all_split_passed and all_dataset_passed and aggregate_passed,
    )


def _timing_values(result: Mapping[str, Any]) -> dict[str, float]:
    warmed = result.get("inference_times_s")
    if not isinstance(warmed, list) or not warmed:
        raise RuntimeError(
            "Timing worker returned no warmed inference samples."
        )
    return {
        "fit_time_s": float(result["fit_time_s"]),
        "first_inference_time_s": float(result["first_inference_time_s"]),
        "warm_inference_time_s": float(np.median(warmed)),
    }


def _timing_row(
    task: Mapping[str, Any],
    *,
    phase: str,
    trial: int,
    execution_order: int,
    implementation: str,
    result: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "dataset": task["dataset"],
        "task_id": task["task_id"],
        "split_index": task["split_index"],
        "repeat": task["repeat"],
        "fold": task["fold"],
        "phase": phase,
        "trial": trial,
        "execution_order": execution_order,
        "implementation": implementation,
        **_timing_values(result),
    }


def run_timing_qualification(
    task_grid: Sequence[Mapping[str, Any]],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    num_cpus: int,
    num_gpus: int,
    timing_trials: int,
    calibration_trials: int,
    warm_inference_repeats: int,
    scratch_dir: Path,
    worker_timeout_s: float,
    source_guard: Callable[[], None],
) -> list[dict[str, object]]:
    """Run fresh-process original/original and original/SDM timing pairs."""
    rows: list[dict[str, object]] = []
    for job_index, task in enumerate(task_grid):
        base_spec = _base_worker_spec(
            task,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            inference_repeats=warm_inference_repeats,
        )
        for trial in range(calibration_trials):
            for execution_order in range(2):
                spec = {
                    **base_spec,
                    "implementation": ORIGINAL,
                    "trial": f"calibration_{job_index}_{trial}",
                    "execution_order": execution_order,
                }
                source_guard()
                result = _local_comparison_module().run_worker_subprocess(
                    spec,
                    scratch_dir=scratch_dir,
                    worker_timeout_s=worker_timeout_s,
                )
                source_guard()
                _validate_task_identity(result, task)
                rows.append(
                    _timing_row(
                        task,
                        phase="calibration",
                        trial=trial,
                        execution_order=execution_order,
                        implementation=ORIGINAL,
                        result=result,
                    )
                )
        for trial in range(timing_trials):
            for execution_order, implementation in enumerate(
                _execution_order(job_index, trial)
            ):
                spec = {
                    **base_spec,
                    "implementation": implementation,
                    "trial": f"timing_{job_index}_{trial}",
                    "execution_order": execution_order,
                }
                source_guard()
                result = _local_comparison_module().run_worker_subprocess(
                    spec,
                    scratch_dir=scratch_dir,
                    worker_timeout_s=worker_timeout_s,
                )
                source_guard()
                _validate_task_identity(result, task)
                rows.append(
                    _timing_row(
                        task,
                        phase="paired",
                        trial=trial,
                        execution_order=execution_order,
                        implementation=implementation,
                        result=result,
                    )
                )
    expected = len(task_grid) * 2 * (calibration_trials + timing_trials)
    if len(rows) != expected:
        raise RuntimeError("Timing qualification did not cover every pair.")
    return rows


def _paired_ratio_summary(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[float, float]:
    numerator_array = np.asarray(numerators, dtype=np.float64)
    denominator_array = np.asarray(denominators, dtype=np.float64)
    if (
        numerator_array.ndim != 1
        or numerator_array.shape != denominator_array.shape
        or numerator_array.size == 0
        or not np.isfinite(numerator_array).all()
        or not np.isfinite(denominator_array).all()
        or (numerator_array <= 0).any()
        or (denominator_array <= 0).any()
    ):
        raise RuntimeError(
            "Timing ratios require finite positive paired values."
        )
    log_ratios = np.log(numerator_array / denominator_array)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        low=0,
        high=log_ratios.size,
        size=(bootstrap_resamples, log_ratios.size),
    )
    upper = float(np.exp(np.quantile(log_ratios[indices].mean(axis=1), 0.95)))
    geometric_mean = float(np.exp(np.mean(log_ratios)))
    return geometric_mean, upper


def _paired_symmetric_ratio_summary(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a direction-invariant paired timing deviation and bound."""
    numerator_array = np.asarray(numerators, dtype=np.float64)
    denominator_array = np.asarray(denominators, dtype=np.float64)
    if (
        numerator_array.ndim != 1
        or numerator_array.shape != denominator_array.shape
        or numerator_array.size == 0
        or not np.isfinite(numerator_array).all()
        or not np.isfinite(denominator_array).all()
        or (numerator_array <= 0).any()
        or (denominator_array <= 0).any()
    ):
        raise RuntimeError(
            "Timing ratios require finite positive paired values."
        )
    log_ratios = np.log(numerator_array / denominator_array)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        low=0,
        high=log_ratios.size,
        size=(bootstrap_resamples, log_ratios.size),
    )
    bootstrap_deviations = np.abs(log_ratios[indices].mean(axis=1))
    upper = float(np.exp(np.quantile(bootstrap_deviations, 0.95)))
    geometric_deviation = float(np.exp(abs(np.mean(log_ratios))))
    return geometric_deviation, upper


def summarize_timing(
    rows: Sequence[Mapping[str, Any]],
    *,
    timing_trials: int,
    calibration_trials: int,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], bool]:
    """Apply calibrated paired timing gates to every split and timing mode."""
    by_task: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        key = (int(row["task_id"]), int(row["repeat"]), int(row["fold"]))
        by_task[key].append(row)
    summary: list[dict[str, object]] = []
    all_passed = True
    timing_fields = (
        "fit_time_s",
        "first_inference_time_s",
        "warm_inference_time_s",
    )
    for task_index, task_rows in enumerate(by_task.values()):
        calibration = [
            row for row in task_rows if row["phase"] == "calibration"
        ]
        paired = [row for row in task_rows if row["phase"] == "paired"]
        first_row = task_rows[0]
        for field_index, timing_field in enumerate(timing_fields):
            calibration_by_trial: dict[int, dict[int, Mapping[str, Any]]] = {}
            for row in calibration:
                calibration_by_trial.setdefault(int(row["trial"]), {})[
                    int(row["execution_order"])
                ] = row
            if set(calibration_by_trial) != set(range(calibration_trials)):
                raise RuntimeError("Timing calibration is missing trial IDs.")
            calibration_first: list[float] = []
            calibration_second: list[float] = []
            for trial in range(calibration_trials):
                pair = calibration_by_trial[trial]
                if set(pair) != {0, 1}:
                    raise RuntimeError(
                        "Timing calibration pair is incomplete."
                    )
                calibration_first.append(float(pair[0][timing_field]))
                calibration_second.append(float(pair[1][timing_field]))

            paired_by_trial: dict[int, dict[str, Mapping[str, Any]]] = {}
            for row in paired:
                paired_by_trial.setdefault(int(row["trial"]), {})[
                    str(row["implementation"])
                ] = row
            if set(paired_by_trial) != set(range(timing_trials)):
                raise RuntimeError("Paired timing is missing trial IDs.")
            original_values: list[float] = []
            sdm_values: list[float] = []
            for trial in range(timing_trials):
                pair = paired_by_trial[trial]
                if set(pair) != {ORIGINAL, SDM}:
                    raise RuntimeError("Paired timing trial is incomplete.")
                original_values.append(float(pair[ORIGINAL][timing_field]))
                sdm_values.append(float(pair[SDM][timing_field]))

            calibration_mean, calibration_upper = (
                _paired_symmetric_ratio_summary(
                    calibration_second,
                    calibration_first,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed + task_index * 31 + field_index,
                )
            )
            sdm_mean, sdm_upper = _paired_ratio_summary(
                sdm_values,
                original_values,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + task_index * 31 + field_index + 1,
            )
            hardware_qualified = calibration_upper <= 1.02
            allowed_slowdown = min(1.02, max(1.005, calibration_upper))
            passed = hardware_qualified and sdm_upper <= allowed_slowdown
            all_passed = all_passed and passed
            summary.append(
                {
                    "dataset": first_row["dataset"],
                    "task_id": first_row["task_id"],
                    "split_index": first_row["split_index"],
                    "repeat": first_row["repeat"],
                    "fold": first_row["fold"],
                    "timing_metric": timing_field,
                    "timing_trials": timing_trials,
                    "calibration_trials": calibration_trials,
                    "original_vs_original_geomean_ratio": calibration_mean,
                    "original_vs_original_upper_95_ratio": calibration_upper,
                    "allowed_slowdown_ratio": allowed_slowdown,
                    "sdm_vs_original_geomean_ratio": sdm_mean,
                    "sdm_vs_original_upper_95_ratio": sdm_upper,
                    "hardware_qualified": hardware_qualified,
                    "timing_gate_passed": passed,
                }
            )
    return summary, all_passed


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    correctness: Sequence[Mapping[str, Any]],
    outcome_summary: Sequence[Mapping[str, Any]],
    *,
    outcome_margin: float,
    timing_summary: Sequence[Mapping[str, Any]] | None,
    calibration_trials: int,
) -> str:
    """Render a compact report that separates outcome and timing claims."""
    overall = next(
        row
        for row in outcome_summary
        if row["scope"] == "all_regression_datasets"
    )
    outcome_rows = []
    for row in outcome_summary:
        if row["scope"] != "dataset":
            continue
        outcome_rows.append(
            [
                str(row["dataset"]),
                str(row["splits"]),
                "{:.8f}".format(float(row["mean_original_nrmse"])),
                "{:.8f}".format(float(row["mean_sdm_nrmse"])),
                "{:+.8f}".format(
                    float(row["mean_nrmse_delta_sdm_minus_original"])
                ),
                "pass" if bool(row["outcome_gate_passed"]) else "fail",
            ]
        )
    total_splits = len(correctness)
    passed_splits = sum(
        bool(row["outcome_gate_passed"]) for row in correctness
    )
    bootstrap_upper = float(overall["bootstrap_upper_95_nrmse_delta"])
    aggregate_passed = bool(overall["outcome_gate_passed"])
    sections = [
        "# TabICLv2 all-regression qualification report",
        "",
        "This run evaluates only the frozen, locally discovered regression "
        "grid. The outcome gate is SDM NRMSE minus original NRMSE no greater "
        f"than {outcome_margin:.1e} on every split and dataset aggregate. "
        "NRMSE divides RMSE by the training-target population standard "
        "deviation, so it is "
        "comparable across targets with different units.",
        "",
        "Raw prediction deltas are retained as diagnostics; they are not a "
        "hard benchmark gate because the agreed claim is non-inferior outcome "
        "quality, not bitwise implementation identity.",
        "",
        "## Outcome gate",
        "",
        f"- Split results: {passed_splits}/{total_splits} passed.",
        f"- Dataset-bootstrap one-sided 95% upper NRMSE delta: "
        f"{bootstrap_upper:+.8f}.",
        "- Aggregate gate: {}.".format("pass" if aggregate_passed else "fail"),
        "",
        _markdown_table(
            [
                "Dataset",
                "Splits",
                "Original mean NRMSE",
                "SDM mean NRMSE",
                "SDM - original",
                "Gate",
            ],
            outcome_rows,
        ),
        "",
    ]
    if timing_summary is None:
        sections.extend(
            [
                "## Timing gate",
                "",
                "Not run. Timing qualification is intentionally blocked until "
                "the complete outcome gate passes.",
                "",
            ]
        )
        return "\n".join(sections)

    timing_passed = sum(
        bool(row["timing_gate_passed"]) for row in timing_summary
    )
    timing_rows = [
        [
            str(row["dataset"]),
            str(row["split_index"]),
            str(row["timing_metric"]),
            "{:.5f}x".format(
                float(row["original_vs_original_upper_95_ratio"])
            ),
            "{:.5f}x".format(float(row["allowed_slowdown_ratio"])),
            "{:.5f}x".format(float(row["sdm_vs_original_upper_95_ratio"])),
            "pass" if bool(row["timing_gate_passed"]) else "fail",
        ]
        for row in timing_summary
    ]
    sections.extend(
        [
            "## Timing gate",
            "",
            "Every timing value is measured in a fresh process. For each "
            f"split and timing mode, the configured {calibration_trials} "
            "original-vs-original pairs estimate a symmetric two-sided 95% "
            "hardware-instability bound. The allowed SDM slowdown "
            "is max(0.5%, that bound), capped at 2%; a noise bound above 2% "
            "invalidates the environment for that measurement.",
            "",
            f"- Timing results: {timing_passed}/{len(timing_summary)} passed.",
            "",
            _markdown_table(
                [
                    "Dataset",
                    "Split",
                    "Timing mode",
                    "A/A upper 95%",
                    "Allowed",
                    "SDM/original upper 95%",
                    "Gate",
                ],
                timing_rows,
            ),
            "",
        ]
    )
    return "\n".join(sections)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Digest verified against the local checkpoint before execution.",
    )
    parser.add_argument(
        "--checkpoint-repository",
        required=True,
        help=("Caller-asserted source repository; recorded but not verified."),
    )
    parser.add_argument(
        "--checkpoint-revision",
        required=True,
        help=(
            "Caller-asserted source revision; recorded but not independently "
            "verified."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-cpus", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--subset",
        choices=(REGRESSION_SUBSET_ALL, REGRESSION_SUBSET_LITE),
        default=REGRESSION_SUBSET_ALL,
        help=(
            "Use every regression split by default; lite is an explicit "
            "smoke subset."
        ),
    )
    parser.add_argument(
        "--outcome-margin",
        type=float,
        default=DEFAULT_OUTCOME_MARGIN,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--run-timing",
        action="store_true",
        help="Run timing only after all outcome gates pass.",
    )
    parser.add_argument(
        "--timing-trials",
        type=int,
        default=DEFAULT_TIMING_TRIALS,
    )
    parser.add_argument(
        "--timing-calibration-trials",
        type=int,
        default=DEFAULT_TIMING_CALIBRATION_TRIALS,
    )
    parser.add_argument(
        "--warm-inference-repeats",
        type=int,
        default=DEFAULT_WARM_INFERENCE_REPEATS,
    )
    parser.add_argument(
        "--worker-timeout-s",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_S,
    )
    return parser.parse_args(argv)


def _artifact_entries(
    paths: Sequence[Path],
) -> dict[str, dict[str, str]]:
    return {
        path.name: {"path": str(path), "sha256": _sha256(path)}
        for path in paths
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run all-regression outcome qualification and optional timing gates."""
    args = _parse_args(argv)
    seed = _validate_seed(args.seed)
    num_cpus = _validate_positive_int(args.num_cpus, option="--num-cpus")
    outcome_margin = _validate_non_negative_float(
        args.outcome_margin,
        option="--outcome-margin",
    )
    bootstrap_resamples = _validate_positive_int(
        args.bootstrap_resamples,
        option="--bootstrap-resamples",
    )
    timing_trials = _validate_positive_int(
        args.timing_trials,
        option="--timing-trials",
    )
    calibration_trials = _validate_positive_int(
        args.timing_calibration_trials,
        option="--timing-calibration-trials",
    )
    warm_inference_repeats = _validate_positive_int(
        args.warm_inference_repeats,
        option="--warm-inference-repeats",
    )
    worker_timeout_s = _validate_timeout(args.worker_timeout_s)
    checkpoint_repository = _validate_provenance(
        args.checkpoint_repository,
        option="--checkpoint-repository",
    )
    checkpoint_revision = _validate_provenance(
        args.checkpoint_revision,
        option="--checkpoint-revision",
    )
    checkpoint_path, checkpoint_sha256 = _validate_checkpoint(
        args.checkpoint_path,
        args.checkpoint_sha256,
    )
    local_runtime = _local_comparison_module()
    allocation = _device_allocation(args.num_gpus)
    output_dir = _ensure_output_dir(args.output_dir)
    repository_root = Path(__file__).resolve().parents[2]
    source_snapshot_path, source_patch_path, runtime_source_state = (
        _write_source_snapshot(repository_root, output_dir)
    )

    def source_guard() -> None:
        _assert_runtime_source_state(runtime_source_state, repository_root)

    source_guard()
    task_grid = _task_grid_records(subset=args.subset)
    source_guard()
    task_grid_path = output_dir / TASK_GRID_FILENAME
    _write_json(task_grid_path, task_grid)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sdm": {"commit": _git_commit(repository_root)},
        "source_snapshot": {
            "path": str(source_snapshot_path.relative_to(output_dir)),
            "sha256": _sha256(source_snapshot_path),
            "tracked_diff_path": str(
                source_patch_path.relative_to(output_dir)
            ),
            "tracked_diff_sha256": _sha256(source_patch_path),
        },
        "tabarena": {"commit": _tabarena_commit()},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "repository": checkpoint_repository,
            "revision": checkpoint_revision,
            "artifact_verification": "local_sha256_verified",
            "source_metadata_verification": "caller_asserted_unverified",
        },
        "resources": {
            "num_cpus": num_cpus,
            "num_gpus": args.num_gpus,
            "device": str(allocation.device),
        },
        "task_grid": {
            "subset": args.subset,
            "splits": len(task_grid),
            "datasets": len({str(task["dataset"]) for task in task_grid}),
            "path": str(task_grid_path),
            "sha256": task_grid_fingerprint(task_grid),
        },
        "configuration": {
            "profile": MATCHED_PARITY,
            "original": local_runtime.resolved_configuration(
                profile=MATCHED_PARITY,
                implementation=ORIGINAL,
                checkpoint_path=checkpoint_path,
                seed=seed,
            ),
            "sdm": local_runtime.resolved_configuration(
                profile=MATCHED_PARITY,
                implementation=SDM,
                checkpoint_path=checkpoint_path,
                seed=seed,
            ),
        },
        "outcome_gate": {
            "per_split_nrmse_margin": outcome_margin,
            "per_dataset_mean_nrmse_margin": outcome_margin,
            "aggregate": "dataset-level paired bootstrap upper one-sided 95%",
            "bootstrap_resamples": bootstrap_resamples,
        },
        "timing_gate": {
            "requested": args.run_timing,
            "fresh_subprocess_per_measurement": True,
            "calibration": "original versus original",
            "calibration_trials": calibration_trials,
            "timing_trials": timing_trials,
            "warm_inference_repeats": warm_inference_repeats,
            "minimum_allowed_slowdown_ratio": 1.005,
            "maximum_allowed_slowdown_ratio": 1.02,
            "statistic": (
                "symmetric original/original absolute-log bootstrap upper "
                "two-sided 95%; SDM/original log-ratio upper one-sided 95%"
            ),
        },
        "environment": local_runtime.environment_manifest(
            num_gpus=args.num_gpus
        ),
    }
    _write_json(manifest_path, manifest)

    correctness_path = output_dir / CORRECTNESS_FILENAME
    outcome_summary_path = output_dir / OUTCOME_SUMMARY_FILENAME
    timing_trials_path = output_dir / TIMING_TRIALS_FILENAME
    timing_summary_path = output_dir / TIMING_SUMMARY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    artifact_paths: list[Path] = [
        task_grid_path,
        source_snapshot_path,
        source_patch_path,
    ]
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _termination_handler)
    try:
        scratch_dir = output_dir / ".workers"
        scratch_dir.mkdir(exist_ok=False)
        correctness = run_correctness_qualification(
            task_grid,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            num_cpus=num_cpus,
            num_gpus=args.num_gpus,
            outcome_margin=outcome_margin,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            worker_timeout_s=worker_timeout_s,
            source_guard=source_guard,
        )
        outcome_summary, outcome_passed = summarize_outcomes(
            correctness,
            outcome_margin=outcome_margin,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        _write_csv(correctness_path, correctness, CORRECTNESS_COLUMNS)
        _write_csv(
            outcome_summary_path,
            outcome_summary,
            OUTCOME_SUMMARY_COLUMNS,
        )
        artifact_paths.extend([correctness_path, outcome_summary_path])
        report_path.write_text(
            render_report(
                correctness,
                outcome_summary,
                outcome_margin=outcome_margin,
                timing_summary=None,
                calibration_trials=calibration_trials,
            )
        )
        artifact_paths.append(report_path)
        prediction_artifacts = {
            str(row["original_prediction_path"]): str(
                row["original_prediction_sha256"]
            )
            for row in correctness
        }
        prediction_artifacts.update(
            {
                str(row["sdm_prediction_path"]): str(
                    row["sdm_prediction_sha256"]
                )
                for row in correctness
            }
        )
        manifest["prediction_artifacts"] = prediction_artifacts
        manifest["outcome_gate"]["passed"] = outcome_passed
        if not outcome_passed:
            source_guard()
            manifest.update(
                {
                    "status": "outcome_gate_failed",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "artifacts": _artifact_entries(artifact_paths),
                }
            )
            _write_json(manifest_path, manifest)
            return 2

        if not args.run_timing:
            source_guard()
            manifest.update(
                {
                    "status": "outcome_qualified",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "artifacts": _artifact_entries(artifact_paths),
                }
            )
            _write_json(manifest_path, manifest)
            return 0

        timing_rows = run_timing_qualification(
            task_grid,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            num_cpus=num_cpus,
            num_gpus=args.num_gpus,
            timing_trials=timing_trials,
            calibration_trials=calibration_trials,
            warm_inference_repeats=warm_inference_repeats,
            scratch_dir=scratch_dir,
            worker_timeout_s=worker_timeout_s,
            source_guard=source_guard,
        )
        timing_summary, timing_passed = summarize_timing(
            timing_rows,
            timing_trials=timing_trials,
            calibration_trials=calibration_trials,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        _write_csv(timing_trials_path, timing_rows, TIMING_TRIAL_COLUMNS)
        _write_csv(timing_summary_path, timing_summary, TIMING_SUMMARY_COLUMNS)
        artifact_paths.extend([timing_trials_path, timing_summary_path])
        report_path.write_text(
            render_report(
                correctness,
                outcome_summary,
                outcome_margin=outcome_margin,
                timing_summary=timing_summary,
                calibration_trials=calibration_trials,
            )
        )
        manifest["timing_gate"]["passed"] = timing_passed
        source_guard()
        manifest.update(
            {
                "status": "completed"
                if timing_passed
                else "timing_gate_failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": _artifact_entries(artifact_paths),
            }
        )
        _write_json(manifest_path, manifest)
        return 0 if timing_passed else 3
    except (KeyboardInterrupt, _QualificationInterrupted) as error:
        if isinstance(error, _QualificationInterrupted):
            signal_name = signal.Signals(error.signum).name
            exit_code = 128 + error.signum
        else:
            signal_name = signal.Signals(signal.SIGINT).name
            exit_code = 128 + signal.SIGINT
        existing_paths = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ]
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                "interruption": {
                    "signal": signal_name,
                    "exit_code": exit_code,
                },
                "artifacts": _artifact_entries(existing_paths),
            }
        )
        _write_json(manifest_path, manifest)
        return exit_code
    except Exception as error:
        existing_paths = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ]
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "artifacts": _artifact_entries(existing_paths),
            }
        )
        _write_json(manifest_path, manifest)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    raise SystemExit(main())
