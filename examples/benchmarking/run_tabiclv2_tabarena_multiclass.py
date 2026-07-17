"""Qualify predictive parity on the installed TabArena multiclass grid.

This outcome-only runner reuses the binary qualification infrastructure for
immutable artifacts, source provenance, paired isolated workers, and bootstrap
gates. It intentionally does not expose or aggregate timing measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from examples.benchmarking import run_tabiclv2_local_comparison as local
from examples.benchmarking import run_tabiclv2_tabarena_binary as shared
from examples.benchmarking.run_tabiclv2_tabarena_smoke import (
    _git_commit,
    _tabarena_commit,
    _validate_checkpoint,
    _validate_provenance,
    _validate_seed,
)
from examples.benchmarking.tabiclv2_tabarena_model import DeviceAllocation

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_WORKER_TIMEOUT_S = 1_800.0
MULTICLASS_SUBSET_ALL = "all"
MULTICLASS_SUBSET_LITE = "lite"
TASK_GRID_FILENAME = "multiclass_task_grid.json"
CORRECTNESS_FILENAME = "multiclass_correctness.csv"
OUTCOME_SUMMARY_FILENAME = "multiclass_outcome_summary.csv"
REPORT_FILENAME = "multiclass_qualification_report.md"
MANIFEST_FILENAME = "manifest.json"
PREDICTIONS_DIRNAME = "predictions"
RUNTIME_SOURCE_PATHS = (
    "examples/benchmarking/run_tabiclv2_tabarena_multiclass.py",
    "examples/benchmarking/run_tabiclv2_tabarena_binary.py",
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
    "sdm/processing/postprocess.py",
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
    "num_classes",
    "original_log_loss",
    "sdm_log_loss",
    "log_loss_delta_sdm_minus_original",
    "original_accuracy",
    "sdm_accuracy",
    "accuracy_delta_sdm_minus_original",
    "probability_rmse_sdm_vs_original",
    "probability_max_abs_delta",
    "class_labels",
    "outcome_margin",
    "outcome_gate_passed",
    "original_prediction_path",
    "original_prediction_sha256",
    "sdm_prediction_path",
    "sdm_prediction_sha256",
)
OUTCOME_SUMMARY_COLUMNS = shared.OUTCOME_SUMMARY_COLUMNS


def _task_grid_records(*, subset: str) -> list[dict[str, object]]:
    """Discover every selected multiclass split from pinned metadata."""
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str

    collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
    filters: dict[str, object] = {"problem_types": ["multiclass"]}
    if subset == MULTICLASS_SUBSET_LITE:
        filters["split_indices"] = "lite"
    elif subset != MULTICLASS_SUBSET_ALL:
        raise ValueError(f"Unknown multiclass subset: {subset!r}.")
    collection = collection.subset_tasks(**filters)

    records: list[dict[str, object]] = []
    for task in collection:
        if task.problem_type != "multiclass" or task.eval_metric != "log_loss":
            raise RuntimeError("Multiclass grid contains a non-log-loss task.")
        if task.task_id_str is None or task.tabarena_task_name is None:
            raise RuntimeError(
                "Multiclass task metadata is missing an identity."
            )
        if task.num_classes is None or not 3 <= task.num_classes <= 10:
            raise RuntimeError(
                "Multiclass grid contains a task outside the supported "
                "3-to-10-class range."
            )
        if not task.class_consistency_over_splits:
            raise RuntimeError(
                "Multiclass grid contains inconsistent split class "
                "vocabularies."
            )
        for split in task.splits_metadata.values():
            if split.num_classes_train != task.num_classes:
                raise RuntimeError(
                    "Multiclass split does not contain the complete class "
                    "vocabulary."
                )
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
                    "num_classes": task.num_classes,
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
        raise RuntimeError("Multiclass task discovery returned no splits.")
    identities = {
        (row["task_id"], row["repeat"], row["fold"]) for row in records
    }
    if len(identities) != len(records):
        raise RuntimeError("Multiclass task grid contains duplicate splits.")
    return records


def _base_worker_spec(
    task: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    num_cpus: int,
    num_gpus: int,
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
        "inference_repeats": 0,
        "problem_type": "multiclass",
        "profile": local.MATCHED_PARITY,
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
        "metric": "log_loss",
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(
                f"Worker result {field!r} does not match the frozen task grid."
            )


def _load_prediction_artifact(
    result: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], Path, str]:
    path_value = result.get("prediction_path")
    sha256 = result.get("prediction_sha256")
    if not isinstance(path_value, str) or not isinstance(sha256, str):
        raise RuntimeError(
            "Correctness worker did not return a prediction artifact."
        )
    path = Path(path_value)
    if not path.is_file() or shared._sha256(path) != sha256:
        raise RuntimeError(
            "Prediction artifact does not match its worker digest."
        )
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"predictions", "targets", "class_labels"}:
            raise RuntimeError("Prediction artifact has an unexpected schema.")
        probabilities = np.asarray(artifact["predictions"], dtype=np.float64)
        targets = np.asarray(artifact["targets"], dtype=np.int64)
        labels = tuple(
            str(value) for value in artifact["class_labels"].tolist()
        )
    if probabilities.ndim != 2 or not 3 <= probabilities.shape[1] <= 10:
        raise RuntimeError(
            "Prediction artifact does not contain supported multiclass "
            "probabilities."
        )
    if (
        targets.shape != (probabilities.shape[0],)
        or len(labels) != probabilities.shape[1]
        or not np.isin(targets, np.arange(probabilities.shape[1])).all()
    ):
        raise RuntimeError(
            "Prediction artifact does not contain aligned multiclass labels."
        )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
        )
    ):
        raise RuntimeError(
            "Prediction artifact contains invalid probabilities."
        )
    return probabilities, targets, labels, path, sha256


def _correctness_row(
    task: Mapping[str, Any],
    *,
    original: Mapping[str, Any],
    sdm: Mapping[str, Any],
    outcome_margin: float,
    output_dir: Path,
) -> dict[str, object]:
    (
        original_probabilities,
        original_targets,
        original_labels,
        original_path,
        original_sha,
    ) = _load_prediction_artifact(original)
    (
        sdm_probabilities,
        sdm_targets,
        sdm_labels,
        sdm_path,
        sdm_sha,
    ) = _load_prediction_artifact(sdm)
    if not np.array_equal(original_targets, sdm_targets):
        raise RuntimeError(
            "Paired implementations used different test targets."
        )
    if original_labels != sdm_labels:
        raise RuntimeError(
            "Paired implementations used different class orders."
        )
    if len(original_labels) != int(task["num_classes"]):
        raise RuntimeError(
            "Prediction class count does not match the frozen task grid."
        )
    original_log_loss = local._multiclass_log_loss(
        original_probabilities,
        original_targets,
    )
    sdm_log_loss = local._multiclass_log_loss(
        sdm_probabilities,
        sdm_targets,
    )
    original_accuracy = local._classification_accuracy(
        original_probabilities,
        original_targets,
    )
    sdm_accuracy = local._classification_accuracy(
        sdm_probabilities,
        sdm_targets,
    )
    for implementation, result, log_loss, accuracy in (
        ("original", original, original_log_loss, original_accuracy),
        ("SDM", sdm, sdm_log_loss, sdm_accuracy),
    ):
        if not math.isclose(
            log_loss,
            float(result["metric_error"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{implementation} worker log loss disagrees with its "
                "artifact."
            )
        if not math.isclose(
            accuracy,
            float(result["accuracy"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{implementation} worker accuracy disagrees with its "
                "artifact."
            )
    delta = sdm_log_loss - original_log_loss
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
        "num_classes": len(original_labels),
        "original_log_loss": original_log_loss,
        "sdm_log_loss": sdm_log_loss,
        "log_loss_delta_sdm_minus_original": delta,
        "original_accuracy": original_accuracy,
        "sdm_accuracy": sdm_accuracy,
        "accuracy_delta_sdm_minus_original": (
            sdm_accuracy - original_accuracy
        ),
        "probability_rmse_sdm_vs_original": shared._probability_rmse(
            sdm_probabilities,
            original_probabilities,
        ),
        "probability_max_abs_delta": float(
            np.max(np.abs(sdm_probabilities - original_probabilities))
        ),
        "class_labels": json.dumps(original_labels),
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
) -> list[dict[str, object]]:
    """Capture one paired prediction artifact for every frozen split."""
    predictions_dir = output_dir / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(exist_ok=False)
    rows: list[dict[str, object]] = []
    for job_index, task in enumerate(task_grid):
        results: dict[str, dict[str, Any]] = {}
        for order_index, implementation in enumerate(
            local.execution_order(job_index, 0)
        ):
            artifact_path = predictions_dir / shared._artifact_stem(
                task,
                implementation,
            )
            spec = {
                **_base_worker_spec(
                    task,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                    seed=seed,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                ),
                "implementation": implementation,
                "trial": f"correctness_{job_index}",
                "execution_order": order_index,
                "prediction_path": str(artifact_path),
            }
            result = local.run_worker_subprocess(
                spec,
                scratch_dir=scratch_dir,
                worker_timeout_s=worker_timeout_s,
            )
            _validate_task_identity(result, task)
            results[implementation] = result
        rows.append(
            _correctness_row(
                task,
                original=results[local.ORIGINAL],
                sdm=results[local.SDM],
                outcome_margin=outcome_margin,
                output_dir=output_dir,
            )
        )
    if len(rows) != len(task_grid):
        raise RuntimeError(
            "Correctness qualification did not cover every split."
        )
    return rows


def summarize_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_margin: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], bool]:
    """Apply split, dataset, and dataset-bootstrap log-loss gates."""
    if not rows:
        raise RuntimeError("Cannot summarize an empty correctness result.")
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    summary: list[dict[str, object]] = []
    dataset_deltas: list[float] = []
    all_split_passed = all(bool(row["outcome_gate_passed"]) for row in rows)
    all_dataset_passed = True
    for dataset in sorted(by_dataset):
        dataset_rows = by_dataset[dataset]
        original = np.asarray(
            [float(row["original_log_loss"]) for row in dataset_rows],
            dtype=np.float64,
        )
        sdm = np.asarray(
            [float(row["sdm_log_loss"]) for row in dataset_rows],
            dtype=np.float64,
        )
        delta = float(np.mean(sdm - original))
        dataset_deltas.append(delta)
        passed = delta <= outcome_margin
        all_dataset_passed = all_dataset_passed and passed
        summary.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "splits": len(dataset_rows),
                "mean_original_log_loss": float(np.mean(original)),
                "mean_sdm_log_loss": float(np.mean(sdm)),
                "mean_log_loss_delta_sdm_minus_original": delta,
                "outcome_margin": outcome_margin,
                "outcome_gate_passed": passed,
                "bootstrap_resamples": None,
                "bootstrap_upper_95_log_loss_delta": None,
            }
        )
    dataset_delta_array = np.asarray(dataset_deltas, dtype=np.float64)
    bootstrap_upper = shared._bootstrap_upper_95(
        dataset_delta_array,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    aggregate_passed = bootstrap_upper <= outcome_margin
    summary.append(
        {
            "scope": "all_multiclass_datasets",
            "dataset": "all",
            "splits": len(rows),
            "mean_original_log_loss": float(
                np.mean([float(row["original_log_loss"]) for row in rows])
            ),
            "mean_sdm_log_loss": float(
                np.mean([float(row["sdm_log_loss"]) for row in rows])
            ),
            "mean_log_loss_delta_sdm_minus_original": float(
                np.mean(dataset_delta_array)
            ),
            "outcome_margin": outcome_margin,
            "outcome_gate_passed": aggregate_passed,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_upper_95_log_loss_delta": bootstrap_upper,
        }
    )
    return (
        summary,
        all_split_passed and all_dataset_passed and aggregate_passed,
    )


def render_report(
    correctness: Sequence[Mapping[str, Any]],
    outcome_summary: Sequence[Mapping[str, Any]],
    *,
    outcome_margin: float,
) -> str:
    """Render an outcome-only multiclass qualification report."""
    overall = next(
        row
        for row in outcome_summary
        if row["scope"] == "all_multiclass_datasets"
    )
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in correctness:
        by_dataset[str(row["dataset"])].append(row)
    summary_by_dataset = {
        str(row["dataset"]): row
        for row in outcome_summary
        if row["scope"] == "dataset"
    }
    rows = []
    for dataset in sorted(by_dataset):
        result_rows = by_dataset[dataset]
        outcome = summary_by_dataset[dataset]
        rows.append(
            [
                dataset,
                str(outcome["splits"]),
                "{:.8f}".format(float(outcome["mean_original_log_loss"])),
                "{:.8f}".format(float(outcome["mean_sdm_log_loss"])),
                "{:+.8f}".format(
                    float(outcome["mean_log_loss_delta_sdm_minus_original"])
                ),
                "{:+.8f}".format(
                    float(
                        np.mean(
                            [
                                float(row["accuracy_delta_sdm_minus_original"])
                                for row in result_rows
                            ]
                        )
                    )
                ),
                "pass" if bool(outcome["outcome_gate_passed"]) else "fail",
            ]
        )
    passed_splits = sum(
        bool(row["outcome_gate_passed"]) for row in correctness
    )
    return "\n".join(
        [
            "# TabICLv2 all-multiclass qualification report",
            "",
            "This outcome-only run evaluates the frozen, locally discovered "
            "multiclass grid. The primary gate is SDM log loss minus original "
            f"log loss at most {outcome_margin:.1e} on every split, every "
            "dataset mean, and the dataset-bootstrap upper bound.",
            "",
            "Top-1 accuracy and raw probability differences are diagnostics "
            "only. No runtime or speed claim is made.",
            "",
            "## Outcome gate",
            "",
            f"- Split results: {passed_splits}/{len(correctness)} passed.",
            "- Dataset-bootstrap one-sided 95% upper log-loss delta: "
            "{:+.8f}.".format(
                float(overall["bootstrap_upper_95_log_loss_delta"])
            ),
            "- Aggregate gate: {}.".format(
                "pass" if bool(overall["outcome_gate_passed"]) else "fail"
            ),
            "",
            shared._markdown_table(
                [
                    "Dataset",
                    "Splits",
                    "Original log loss",
                    "SDM log loss",
                    "Log-loss delta",
                    "Accuracy delta",
                    "Gate",
                ],
                rows,
            ),
            "",
        ]
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-repository", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-cpus", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--subset",
        choices=(MULTICLASS_SUBSET_ALL, MULTICLASS_SUBSET_LITE),
        default=MULTICLASS_SUBSET_ALL,
        help=(
            "Use every multiclass split by default; lite selects one split "
            "per dataset."
        ),
    )
    parser.add_argument(
        "--outcome-margin",
        required=True,
        type=float,
        help="Frozen absolute log-loss non-inferiority margin.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--worker-timeout-s",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_S,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run multiclass predictive-outcome qualification."""
    args = _parse_args(argv)
    seed = _validate_seed(args.seed)
    num_cpus = shared._validate_positive_int(
        args.num_cpus,
        option="--num-cpus",
    )
    outcome_margin = shared._validate_non_negative_float(
        args.outcome_margin,
        option="--outcome-margin",
    )
    bootstrap_resamples = shared._validate_positive_int(
        args.bootstrap_resamples,
        option="--bootstrap-resamples",
    )
    worker_timeout_s = shared._validate_timeout(args.worker_timeout_s)
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
    allocation = DeviceAllocation.from_num_gpus(args.num_gpus)
    output_dir = shared._ensure_output_dir(args.output_dir)
    task_grid = _task_grid_records(subset=args.subset)
    task_grid_path = output_dir / TASK_GRID_FILENAME
    shared._write_json(task_grid_path, task_grid)

    repository_root = Path(__file__).resolve().parents[2]
    source_snapshot_path, source_patch_path = shared._write_source_snapshot(
        repository_root,
        output_dir,
        runtime_source_paths=RUNTIME_SOURCE_PATHS,
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "sdm": {"commit": _git_commit(repository_root)},
        "source_snapshot": {
            "path": str(source_snapshot_path.relative_to(output_dir)),
            "sha256": shared._sha256(source_snapshot_path),
            "tracked_diff_path": str(
                source_patch_path.relative_to(output_dir)
            ),
            "tracked_diff_sha256": shared._sha256(source_patch_path),
        },
        "tabarena": {"commit": _tabarena_commit()},
        "checkpoint": {
            "variant": "classifier",
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "repository": checkpoint_repository,
            "revision": checkpoint_revision,
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
            "sha256": shared.task_grid_fingerprint(task_grid),
        },
        "configuration": {
            "problem_type": "multiclass",
            "profile": local.MATCHED_PARITY,
            "original": local.resolved_configuration(
                profile=local.MATCHED_PARITY,
                implementation=local.ORIGINAL,
                checkpoint_path=checkpoint_path,
                seed=seed,
                problem_type="multiclass",
            ),
            "sdm": local.resolved_configuration(
                profile=local.MATCHED_PARITY,
                implementation=local.SDM,
                checkpoint_path=checkpoint_path,
                seed=seed,
                problem_type="multiclass",
            ),
        },
        "outcome_gate": {
            "per_split_log_loss_margin": outcome_margin,
            "per_dataset_mean_log_loss_margin": outcome_margin,
            "aggregate": "dataset-level paired bootstrap upper one-sided 95%",
            "bootstrap_resamples": bootstrap_resamples,
        },
        "runtime_qualification": {
            "requested": False,
            "claim": "none",
        },
        "environment": local.environment_manifest(num_gpus=args.num_gpus),
    }
    shared._write_json(manifest_path, manifest)

    correctness_path = output_dir / CORRECTNESS_FILENAME
    outcome_summary_path = output_dir / OUTCOME_SUMMARY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    artifact_paths = [
        task_grid_path,
        source_snapshot_path,
        source_patch_path,
    ]
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
        )
        outcome_summary, outcome_passed = summarize_outcomes(
            correctness,
            outcome_margin=outcome_margin,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        shared._write_csv(
            correctness_path,
            correctness,
            CORRECTNESS_COLUMNS,
        )
        shared._write_csv(
            outcome_summary_path,
            outcome_summary,
            OUTCOME_SUMMARY_COLUMNS,
        )
        report_path.write_text(
            render_report(
                correctness,
                outcome_summary,
                outcome_margin=outcome_margin,
            )
        )
        artifact_paths.extend(
            [correctness_path, outcome_summary_path, report_path]
        )
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
        manifest.update(
            {
                "status": (
                    "outcome_qualified"
                    if outcome_passed
                    else "outcome_gate_failed"
                ),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "artifacts": shared._artifact_entries(artifact_paths),
            }
        )
        shared._write_json(manifest_path, manifest)
        return 0 if outcome_passed else 2
    except Exception as error:
        existing_paths = [
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path != manifest_path
        ]
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "artifacts": shared._artifact_entries(existing_paths),
            }
        )
        shared._write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
