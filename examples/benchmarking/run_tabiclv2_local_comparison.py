"""Benchmark original and SDM TabICLv2 on one controlled local split.

The runner measures two local configuration profiles in fresh subprocesses and
adds one archived TabArena result as clearly separated context:

* matched parity: the only profile eligible for fair speedup claims;
* local native defaults: same machine, intentionally different workloads;
* historical TabArena: cached external context from an uncontrolled machine.

The generated artifacts are local files only. TabArena may download its cached
historical result through its normal read path; this script never uploads
results or changes the TabArena repository.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from sdm import Stype, TableTensor
from sdm.models import TabICLv2
from sdm.processing import (
    ConstantFilter,
    Identity,
    MeanImpute,
    Processor,
    Recipe,
    SigmaClip,
    StandardScale,
)

from examples.benchmarking.run_tabiclv2_tabarena_smoke import (
    _git_commit,
    _tabarena_commit,
    _validate_checkpoint,
    _validate_provenance,
    _validate_seed,
)
from examples.benchmarking.tabiclv2_tabarena_model import (
    DeviceAllocation,
    SDMTabICLv2Model,
)

DEFAULT_DATASET = "QSAR_fish_toxicity"
DEFAULT_TASK_ID = 363698
DEFAULT_SEED = 0
DEFAULT_WARMUP_PAIRS = 1
DEFAULT_TRIALS = 30
DEFAULT_INFERENCE_REPEATS = 20
DEFAULT_WORKER_TIMEOUT_S = 1800.0
HISTORICAL_METHOD = "TabICLv2"
HISTORICAL_CONFIG_TYPE = "TABICLV2"
HISTORICAL_METHOD_SUBTYPE = "default"
MATCHED_PARITY = "matched_parity"
LOCAL_NATIVE_DEFAULT = "local_native_default"
HISTORICAL_TABARENA = "historical_tabarena"
ORIGINAL = "original_tabicl"
SDM = "sdm_tabicl"
LOCAL_PROFILES: Final = (MATCHED_PARITY, LOCAL_NATIVE_DEFAULT)
IMPLEMENTATIONS: Final = (ORIGINAL, SDM)
TRIALS_FILENAME = "tabiclv2_local_trials.csv"
INFERENCE_FILENAME = "tabiclv2_local_inference.csv"
SUMMARY_FILENAME = "tabiclv2_comparison_summary.csv"
REPORT_FILENAME = "tabiclv2_comparison_report.md"
MANIFEST_FILENAME = "manifest.json"

TRIAL_COLUMNS: Final = (
    "comparison_group",
    "execution_source",
    "timing_comparability",
    "eligible_for_fair_speedup",
    "configuration_profile",
    "implementation",
    "trial",
    "execution_order",
    "seed",
    "dataset",
    "task_id",
    "tabarena_split",
    "repeat",
    "fold",
    "sample",
    "train_rows",
    "test_rows",
    "features",
    "metric",
    "metric_error",
    "fit_time_s",
    "first_inference_time_s",
    "inference_time_s",
)
INFERENCE_COLUMNS: Final = (
    "comparison_group",
    "configuration_profile",
    "implementation",
    "trial",
    "execution_order",
    "inference_repeat",
    "duration_s",
)
SUMMARY_COLUMNS: Final = (
    "comparison_group",
    "execution_source",
    "timing_comparability",
    "eligible_for_fair_speedup",
    "configuration_profile",
    "implementation",
    "method",
    "dataset",
    "task_id",
    "tabarena_split",
    "repeat",
    "fold",
    "sample",
    "metric",
    "trials",
    "inference_repeats",
    "timing_definition",
    "metric_error_mean",
    "metric_error_std",
    "metric_error",
    "metric_error_q1",
    "metric_error_q3",
    "fit_time_s_mean",
    "fit_time_s_std",
    "fit_time_s",
    "fit_time_s_q1",
    "fit_time_s_q3",
    "first_inference_time_s_mean",
    "first_inference_time_s_std",
    "first_inference_time_s",
    "first_inference_time_s_q1",
    "first_inference_time_s_q3",
    "inference_time_s_mean",
    "inference_time_s_std",
    "inference_time_s",
    "inference_time_s_q1",
    "inference_time_s_q3",
    "fit_speedup_vs_original_mean",
    "fit_speedup_vs_original_std",
    "fit_speedup_vs_original",
    "fit_speedup_vs_original_q1",
    "fit_speedup_vs_original_q3",
    "first_inference_speedup_vs_original_mean",
    "first_inference_speedup_vs_original_std",
    "first_inference_speedup_vs_original",
    "first_inference_speedup_vs_original_q1",
    "first_inference_speedup_vs_original_q3",
    "inference_speedup_vs_original_mean",
    "inference_speedup_vs_original_std",
    "inference_speedup_vs_original",
    "inference_speedup_vs_original_q1",
    "inference_speedup_vs_original_q3",
    "historical_suite",
    "historical_config",
)


class _FixedRangeClip(Processor):
    """Apply the reference implementation's fixed z-score safety bounds."""

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.clamp(min=-100.0, max=100.0)
        return table.replace_blocks(numerical=numerical)


def matched_parity_recipe() -> Recipe:
    """Return the deterministic single-estimator parity recipe."""
    return Recipe(
        features=[
            MeanImpute(),
            ConstantFilter(),
            StandardScale(epsilon=1e-6),
            _FixedRangeClip(),
            Identity(),
            SigmaClip(threshold=4.0),
            Identity(),
        ],
        target=[StandardScale()],
        output=[Identity()],
    )


class MatchedParitySDMTabICLv2Model(SDMTabICLv2Model):
    """SDM adapter variant with the explicit matched-parity recipe."""

    def _build_recipe(self, model: TabICLv2) -> Recipe:
        del model
        return matched_parity_recipe()


def comparison_labels(profile: str) -> dict[str, object]:
    """Return report classification fields for a local profile."""
    if profile == MATCHED_PARITY:
        return {
            "comparison_group": MATCHED_PARITY,
            "execution_source": "local_measured",
            "timing_comparability": "matched",
            "eligible_for_fair_speedup": True,
            "configuration_profile": MATCHED_PARITY,
        }
    if profile == LOCAL_NATIVE_DEFAULT:
        return {
            "comparison_group": LOCAL_NATIVE_DEFAULT,
            "execution_source": "local_measured",
            "timing_comparability": "different_workload",
            "eligible_for_fair_speedup": False,
            "configuration_profile": LOCAL_NATIVE_DEFAULT,
        }
    raise ValueError(f"Unknown local comparison profile: {profile!r}.")


def resolved_configuration(
    *,
    profile: str,
    implementation: str,
    checkpoint_path: Path,
    seed: int,
) -> dict[str, object]:
    """Describe the complete algorithm configuration shown in the manifest."""
    if implementation == ORIGINAL:
        artifact = {
            "model_path": str(checkpoint_path),
            "allow_auto_download": False,
        }
    elif implementation == SDM:
        artifact = {
            "checkpoint_path": str(checkpoint_path),
            "allow_auto_download": False,
        }
    else:
        raise ValueError(f"Unknown implementation: {implementation!r}.")
    if implementation == ORIGINAL and profile == MATCHED_PARITY:
        return {
            **artifact,
            "n_estimators": 1,
            "norm_methods": "none",
            "feat_shuffle_method": "none",
            "outlier_threshold": 4.0,
            "batch_size": 1,
            "kv_cache": "kv",
            "random_state": seed,
            "use_amp": False,
            "use_fa3": False,
            "offload_mode": False,
        }
    if implementation == SDM and profile == MATCHED_PARITY:
        return {
            **artifact,
            "num_estimators": 1,
            "seed": seed,
            "cache": True,
            "recipe": [
                "mean_impute",
                "constant_filter",
                "standard_scale_epsilon_1e-6",
                "fixed_clip_-100_100",
                "identity_normalization",
                "sigma_clip_4",
                "identity_feature_permutation",
                "target_standard_scale",
            ],
        }
    if implementation == ORIGINAL and profile == LOCAL_NATIVE_DEFAULT:
        return {
            **artifact,
            "n_estimators": 8,
            "norm_methods": ["none", "power"],
            "feat_shuffle_method": "latin",
            "outlier_threshold": 4.0,
            "batch_size": 8,
            "kv_cache": False,
            "random_state": 42,
            "use_amp": "auto",
            "use_fa3": "auto",
            "offload_mode": "auto",
        }
    if implementation == SDM and profile == LOCAL_NATIVE_DEFAULT:
        return {
            **artifact,
            "num_estimators": 1,
            "seed": seed,
            "cache": True,
            "recipe": "sdm_tabiclv2_default_recipe",
        }
    raise ValueError(
        "Unknown benchmark configuration pair: "
        f"profile={profile!r}, implementation={implementation!r}."
    )


def model_hyperparameters(
    *,
    profile: str,
    implementation: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
) -> dict[str, object]:
    """Return hyperparameters passed to the two AutoGluon adapters."""
    config = resolved_configuration(
        profile=profile,
        implementation=implementation,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
    if implementation == ORIGINAL:
        result = dict(config)
        result.pop("recipe", None)
        result.pop("cache", None)
        return result
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "num_estimators": 1,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-repository", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-cpus", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int, choices=(0, 1))
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--warmup-pairs",
        type=int,
        default=DEFAULT_WARMUP_PAIRS,
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--inference-repeats",
        type=int,
        default=DEFAULT_INFERENCE_REPEATS,
    )
    parser.add_argument(
        "--worker-timeout-s",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_S,
    )
    return parser.parse_args(argv)


def _parse_worker_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser.parse_args(argv)


def _validate_positive_int(value: int, *, option: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{option} must be an integer of at least 1.")
    return value


def _validate_non_negative_int(value: int, *, option: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{option} must be a non-negative integer.")
    return value


def _validate_timeout(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--worker-timeout-s must be finite and positive.")
    return value


def _sync_cuda(num_gpus: int) -> None:
    if num_gpus:
        torch.cuda.synchronize()


def _timed_call(
    call: Callable[[], Any],
    *,
    num_gpus: int,
) -> tuple[Any, float]:
    """Measure a call with explicit accelerator synchronization."""
    _sync_cuda(num_gpus)
    started_ns = time.perf_counter_ns()
    value = call()
    _sync_cuda(num_gpus)
    finished_ns = time.perf_counter_ns()
    elapsed = (finished_ns - started_ns) / 1_000_000_000
    if elapsed <= 0:
        raise RuntimeError("Measured duration must be positive.")
    return value, elapsed


def execution_order(profile_index: int, trial_index: int) -> tuple[str, str]:
    """Alternate implementation order to reduce systematic order bias."""
    if (profile_index + trial_index) % 2 == 0:
        return IMPLEMENTATIONS
    return IMPLEMENTATIONS[1], IMPLEMENTATIONS[0]


def _load_split(spec: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, int]:
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    from tabarena.benchmark.task.utils import get_split_idx

    task = OpenMLTaskWrapper.from_task_id(task_id=int(spec["task_id"]))
    if task.problem_type != "regression":
        raise RuntimeError(
            f"Expected a regression task, got {task.problem_type!r}."
        )
    dataset_name = task.dataset_name
    if dataset_name != spec["dataset"]:
        raise RuntimeError(
            f"Task dataset mismatch: expected {spec['dataset']!r}, "
            f"got {dataset_name!r}."
        )
    repeat = int(spec["repeat"])
    fold = int(spec["fold"])
    sample = int(spec["sample"])
    dimensions = task.get_split_dimensions()
    train_indices, test_indices = task.get_split_indices(
        repeat=repeat,
        fold=fold,
        sample=sample,
    )
    split_idx = get_split_idx(
        repeat=repeat,
        fold=fold,
        sample=sample,
        n_repeats=dimensions[0],
        n_folds=dimensions[1],
        n_samples=dimensions[2],
    )
    X_train = task.X.iloc[train_indices].reset_index(drop=True)
    y_train = task.y.iloc[train_indices].reset_index(drop=True)
    X_test = task.X.iloc[test_indices].reset_index(drop=True)
    y_test = task.y.iloc[test_indices].reset_index(drop=True)
    return X_train, y_train, X_test, y_test, split_idx


def _construct_model(spec: Mapping[str, Any]) -> Any:
    implementation = str(spec["implementation"])
    profile = str(spec["profile"])
    if implementation == ORIGINAL:
        from tabarena.models.tabicl.model import TabICLv2Model

        model_cls = TabICLv2Model
    elif implementation == SDM and profile == MATCHED_PARITY:
        model_cls = MatchedParitySDMTabICLv2Model
    elif implementation == SDM:
        model_cls = SDMTabICLv2Model
    else:
        raise ValueError(f"Unknown implementation: {implementation!r}.")

    hyperparameters = model_hyperparameters(
        profile=profile,
        implementation=implementation,
        checkpoint_path=Path(spec["checkpoint_path"]),
        checkpoint_sha256=str(spec["checkpoint_sha256"]),
        seed=int(spec["seed"]),
    )
    return model_cls(
        path="",
        name=f"{profile}_{implementation}",
        problem_type="regression",
        eval_metric=None,
        hyperparameters=hyperparameters,
    )


def _validate_predictions(
    predictions: Any, *, expected_rows: int
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (expected_rows,):
        raise RuntimeError(
            f"Expected {expected_rows} scalar predictions, got "
            f"shape {values.shape}."
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Model produced non-finite predictions.")
    return values


def _worker_run(spec: Mapping[str, Any]) -> dict[str, object]:
    num_cpus = _validate_positive_int(
        int(spec["num_cpus"]),
        option="worker num_cpus",
    )
    num_gpus = int(spec["num_gpus"])
    DeviceAllocation.from_num_gpus(num_gpus)
    torch.set_num_threads(num_cpus)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    X_train, y_train, X_test, y_test, split_idx = _load_split(spec)

    def construct_and_fit() -> Any:
        model = _construct_model(spec)
        model.fit(
            X=X_train,
            y=y_train,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
        return model

    model, fit_time_s = _timed_call(
        construct_and_fit,
        num_gpus=num_gpus,
    )
    first_raw, first_time_s = _timed_call(
        lambda: model.predict(X_test),
        num_gpus=num_gpus,
    )
    predictions = _validate_predictions(first_raw, expected_rows=len(X_test))

    inference_times: list[float] = []
    for _ in range(int(spec["inference_repeats"])):
        repeated_raw, duration = _timed_call(
            lambda: model.predict(X_test),
            num_gpus=num_gpus,
        )
        repeated = _validate_predictions(
            repeated_raw,
            expected_rows=len(X_test),
        )
        if not np.allclose(repeated, predictions, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Repeated inference changed model predictions.")
        inference_times.append(duration)

    target = np.asarray(y_test, dtype=np.float64)
    metric_error = float(np.sqrt(np.mean(np.square(predictions - target))))
    if not math.isfinite(metric_error):
        raise RuntimeError("RMSE must be finite.")

    return {
        "profile": spec["profile"],
        "implementation": spec["implementation"],
        "trial": spec["trial"],
        "execution_order": spec["execution_order"],
        "seed": spec["seed"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "tabarena_split": split_idx,
        "repeat": spec["repeat"],
        "fold": spec["fold"],
        "sample": spec["sample"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X_train.shape[1],
        "metric": "rmse",
        "metric_error": metric_error,
        "fit_time_s": fit_time_s,
        "first_inference_time_s": first_time_s,
        "inference_times_s": inference_times,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _worker_main(argv: Sequence[str]) -> int:
    args = _parse_worker_args(argv)
    spec = json.loads(args.spec.read_text())
    result = _worker_run(spec)
    _write_json(args.result, result)
    return 0


def _tail(value: str, *, limit: int = 3000) -> str:
    return value if len(value) <= limit else value[-limit:]


def run_worker_subprocess(
    spec: Mapping[str, Any],
    *,
    scratch_dir: Path,
    worker_timeout_s: float,
) -> dict[str, Any]:
    """Execute exactly one implementation/trial in a fresh process."""
    stem = (
        f"{spec['profile']}-{spec['implementation']}-"
        f"{spec['trial']}-{spec['execution_order']}"
    )
    spec_path = scratch_dir / f"{stem}-spec.json"
    result_path = scratch_dir / f"{stem}-result.json"
    _write_json(spec_path, spec)

    env = os.environ.copy()
    threads = str(spec["num_cpus"])
    env.update(
        {
            "PYTHONHASHSEED": str(spec["seed"]),
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        }
    )
    command = [
        sys.executable,
        "-m",
        "examples.benchmarking.run_tabiclv2_local_comparison",
        "_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=worker_timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Benchmark worker failed "
            f"(profile={spec['profile']}, "
            f"implementation={spec['implementation']}, "
            f"trial={spec['trial']}):\n"
            f"stdout:\n{_tail(completed.stdout)}\n"
            f"stderr:\n{_tail(completed.stderr)}"
        )
    if not result_path.is_file():
        raise RuntimeError("Benchmark worker did not write its result file.")
    result = json.loads(result_path.read_text())
    _validate_worker_result(result, spec=spec)
    return result


def _require_finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Worker field {field!r} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Worker field {field!r} must be numeric."
        ) from error
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(
            f"Worker field {field!r} must be finite and positive."
        )
    return number


def _validate_worker_result(
    result: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> None:
    for field in ("profile", "implementation", "trial", "execution_order"):
        if result.get(field) != spec.get(field):
            raise RuntimeError(
                f"Worker result field {field!r} does not match its spec."
            )
    _require_finite_positive(result.get("fit_time_s"), field="fit_time_s")
    _require_finite_positive(
        result.get("first_inference_time_s"),
        field="first_inference_time_s",
    )
    metric_error = result.get("metric_error")
    if (
        isinstance(metric_error, bool)
        or not isinstance(metric_error, (int, float))
        or not math.isfinite(float(metric_error))
        or float(metric_error) < 0
    ):
        raise RuntimeError("Worker field 'metric_error' must be finite.")
    inference_times = result.get("inference_times_s")
    if not isinstance(inference_times, list) or len(inference_times) != int(
        spec["inference_repeats"]
    ):
        raise RuntimeError(
            "Worker returned the wrong number of inference measurements."
        )
    for index, duration in enumerate(inference_times):
        _require_finite_positive(
            duration,
            field=f"inference_times_s[{index}]",
        )


WorkerRunner = Callable[[Mapping[str, Any]], dict[str, Any]]


def run_local_benchmarks(
    base_spec: Mapping[str, Any],
    *,
    warmup_pairs: int,
    trials: int,
    worker_runner: WorkerRunner,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run discarded warm-up pairs and alternating measured trials."""
    trial_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []

    for profile_index, profile in enumerate(LOCAL_PROFILES):
        for warmup in range(warmup_pairs):
            order = execution_order(profile_index, warmup)
            for order_index, implementation in enumerate(order):
                spec = {
                    **base_spec,
                    "profile": profile,
                    "implementation": implementation,
                    "trial": f"warmup_{warmup}",
                    "execution_order": order_index,
                }
                worker_runner(spec)

        for trial in range(trials):
            order = execution_order(profile_index, trial)
            for order_index, implementation in enumerate(order):
                spec = {
                    **base_spec,
                    "profile": profile,
                    "implementation": implementation,
                    "trial": trial,
                    "execution_order": order_index,
                }
                result = worker_runner(spec)
                _validate_worker_result(result, spec=spec)
                labels = comparison_labels(profile)
                inference_times = [
                    float(value) for value in result["inference_times_s"]
                ]
                trial_rows.append(
                    {
                        **labels,
                        "implementation": implementation,
                        "trial": trial,
                        "execution_order": order_index,
                        "seed": result["seed"],
                        "dataset": result["dataset"],
                        "task_id": result["task_id"],
                        "tabarena_split": result["tabarena_split"],
                        "repeat": result["repeat"],
                        "fold": result["fold"],
                        "sample": result["sample"],
                        "train_rows": result["train_rows"],
                        "test_rows": result["test_rows"],
                        "features": result["features"],
                        "metric": result["metric"],
                        "metric_error": result["metric_error"],
                        "fit_time_s": result["fit_time_s"],
                        "first_inference_time_s": result[
                            "first_inference_time_s"
                        ],
                        "inference_time_s": float(np.median(inference_times)),
                    }
                )
                for repeat_index, duration in enumerate(inference_times):
                    inference_rows.append(
                        {
                            "comparison_group": profile,
                            "configuration_profile": profile,
                            "implementation": implementation,
                            "trial": trial,
                            "execution_order": order_index,
                            "inference_repeat": repeat_index,
                            "duration_s": duration,
                        }
                    )

    return trial_rows, inference_rows


def _distribution_fields(
    values: Sequence[float], *, field: str
) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError(
            "Cannot summarize empty or non-finite measurements."
        )
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else None
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        f"{field}_mean": mean,
        f"{field}_std": std,
        field: float(median),
        f"{field}_q1": float(q1),
        f"{field}_q3": float(q3),
    }


def aggregate_local_results(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    trials: int,
    inference_repeats: int,
) -> list[dict[str, object]]:
    """Aggregate trial distributions and attach paired matched speedups."""
    summary: list[dict[str, object]] = []
    indexed_local_rows: dict[
        tuple[str, str], dict[int, Mapping[str, Any]]
    ] = {}
    for profile in LOCAL_PROFILES:
        for implementation in IMPLEMENTATIONS:
            rows = [
                row
                for row in trial_rows
                if row["comparison_group"] == profile
                and row["implementation"] == implementation
            ]
            if len(rows) != trials:
                raise RuntimeError(
                    f"Expected {trials} rows for {profile}/{implementation}, "
                    f"got {len(rows)}."
                )
            by_trial: dict[int, Mapping[str, Any]] = {}
            for row in rows:
                trial = int(row["trial"])
                if trial in by_trial:
                    raise RuntimeError(
                        f"Duplicate trial {trial} for "
                        f"{profile}/{implementation}."
                    )
                by_trial[trial] = row
            expected_trials = set(range(trials))
            if set(by_trial) != expected_trials:
                raise RuntimeError(
                    f"Expected trial IDs {sorted(expected_trials)!r} for "
                    f"{profile}/{implementation}, got {sorted(by_trial)!r}."
                )
            rows = [by_trial[trial] for trial in range(trials)]
            indexed_local_rows[(profile, implementation)] = by_trial
            first_row = rows[0]
            summary.append(
                {
                    **comparison_labels(profile),
                    "implementation": implementation,
                    "method": (
                        "Original TabICLv2"
                        if implementation == ORIGINAL
                        else "SDM TabICLv2"
                    ),
                    "dataset": first_row["dataset"],
                    "task_id": first_row["task_id"],
                    "tabarena_split": first_row["tabarena_split"],
                    "repeat": first_row["repeat"],
                    "fold": first_row["fold"],
                    "sample": first_row["sample"],
                    "metric": "rmse",
                    "trials": trials,
                    "inference_repeats": inference_repeats,
                    "timing_definition": (
                        "local end-to-end fit; first predict; "
                        "median warmed predict per trial"
                    ),
                    **_distribution_fields(
                        [float(row["metric_error"]) for row in rows],
                        field="metric_error",
                    ),
                    **_distribution_fields(
                        [float(row["fit_time_s"]) for row in rows],
                        field="fit_time_s",
                    ),
                    **_distribution_fields(
                        [float(row["first_inference_time_s"]) for row in rows],
                        field="first_inference_time_s",
                    ),
                    **_distribution_fields(
                        [float(row["inference_time_s"]) for row in rows],
                        field="inference_time_s",
                    ),
                    "fit_speedup_vs_original": None,
                    "first_inference_speedup_vs_original": None,
                    "inference_speedup_vs_original": None,
                    "historical_suite": None,
                    "historical_config": None,
                }
            )

    matched = {
        row["implementation"]: row
        for row in summary
        if row["comparison_group"] == MATCHED_PARITY
    }
    original_rows = indexed_local_rows[(MATCHED_PARITY, ORIGINAL)]
    speedup_fields = (
        ("fit_time_s", "fit_speedup_vs_original"),
        (
            "first_inference_time_s",
            "first_inference_speedup_vs_original",
        ),
        ("inference_time_s", "inference_speedup_vs_original"),
    )
    for implementation, row in matched.items():
        implementation_rows = indexed_local_rows[
            (MATCHED_PARITY, str(implementation))
        ]
        for timing_field, speedup_field in speedup_fields:
            paired_speedups: list[float] = []
            for trial in range(trials):
                baseline = _require_finite_positive(
                    original_rows[trial][timing_field], field=timing_field
                )
                duration = _require_finite_positive(
                    implementation_rows[trial][timing_field],
                    field=timing_field,
                )
                paired_speedups.append(baseline / duration)
            row.update(
                _distribution_fields(
                    paired_speedups,
                    field=speedup_field,
                )
            )
    return summary


def select_historical_baseline(
    results: Any,
    *,
    dataset: str,
    tabarena_split: int,
    task_id: int,
    repeat: int,
    fold: int,
    sample: int,
    suite: str,
    config: str,
) -> dict[str, object]:
    """Select and classify exactly one archived TabArena default row."""
    import pandas as pd

    if not isinstance(results, pd.DataFrame):
        raise RuntimeError("Expected historical TabArena results as a frame.")
    required = {
        "dataset",
        "fold",
        "method",
        "metric_error",
        "time_train_s",
        "time_infer_s",
        "metric",
        "problem_type",
        "method_subtype",
        "config_type",
        "ta_suite",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise RuntimeError(
            f"Historical TabArena results are missing columns: {missing!r}."
        )
    selected = results.loc[
        (results["dataset"] == dataset)
        & (results["fold"] == tabarena_split)
        & (results["config_type"] == HISTORICAL_CONFIG_TYPE)
        & (results["method_subtype"] == HISTORICAL_METHOD_SUBTYPE)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one historical TabArena TabICLv2 default row, "
            f"got {len(selected)}."
        )
    row = selected.iloc[0]
    if row["problem_type"] != "regression" or row["metric"] != "rmse":
        raise RuntimeError("Expected a historical TabICLv2 regression RMSE.")
    if row["ta_suite"] != suite:
        raise RuntimeError(
            f"Historical suite mismatch: expected {suite!r}, "
            f"got {row['ta_suite']!r}."
        )
    metric_error = float(row["metric_error"])
    fit_time = _require_finite_positive(
        row["time_train_s"],
        field="historical time_train_s",
    )
    inference_time = _require_finite_positive(
        row["time_infer_s"],
        field="historical time_infer_s",
    )
    if not math.isfinite(metric_error) or metric_error < 0:
        raise RuntimeError("Historical RMSE must be finite and non-negative.")

    return {
        "comparison_group": HISTORICAL_TABARENA,
        "execution_source": "tabarena_cached",
        "timing_comparability": "different_environment",
        "eligible_for_fair_speedup": False,
        "configuration_profile": HISTORICAL_TABARENA,
        "implementation": "historical_original_tabicl",
        "method": row["method"],
        "dataset": dataset,
        "task_id": task_id,
        "tabarena_split": tabarena_split,
        "repeat": repeat,
        "fold": fold,
        "sample": sample,
        "metric": "rmse",
        "trials": None,
        "inference_repeats": None,
        "timing_definition": (
            "archived TabArena-reported timing; hardware uncontrolled"
        ),
        "metric_error_mean": None,
        "metric_error_std": None,
        "metric_error": metric_error,
        "metric_error_q1": None,
        "metric_error_q3": None,
        "fit_time_s_mean": None,
        "fit_time_s_std": None,
        "fit_time_s": fit_time,
        "fit_time_s_q1": None,
        "fit_time_s_q3": None,
        "first_inference_time_s_mean": None,
        "first_inference_time_s_std": None,
        "first_inference_time_s": None,
        "first_inference_time_s_q1": None,
        "first_inference_time_s_q3": None,
        "inference_time_s_mean": None,
        "inference_time_s_std": None,
        "inference_time_s": inference_time,
        "inference_time_s_q1": None,
        "inference_time_s_q3": None,
        "fit_speedup_vs_original_mean": None,
        "fit_speedup_vs_original_std": None,
        "fit_speedup_vs_original": None,
        "fit_speedup_vs_original_q1": None,
        "fit_speedup_vs_original_q3": None,
        "first_inference_speedup_vs_original_mean": None,
        "first_inference_speedup_vs_original_std": None,
        "first_inference_speedup_vs_original": None,
        "first_inference_speedup_vs_original_q1": None,
        "first_inference_speedup_vs_original_q3": None,
        "inference_speedup_vs_original_mean": None,
        "inference_speedup_vs_original_std": None,
        "inference_speedup_vs_original": None,
        "inference_speedup_vs_original_q1": None,
        "inference_speedup_vs_original_q3": None,
        "historical_suite": suite,
        "historical_config": config,
    }


def load_historical_baseline(
    *,
    dataset: str,
    tabarena_split: int,
    task_id: int,
    repeat: int,
    fold: int,
    sample: int,
) -> dict[str, object]:
    """Read the official cached baseline without registering local results."""
    from tabarena.contexts import TabArenaContext

    context = TabArenaContext()
    metadata = context.method_metadata(method=HISTORICAL_METHOD)
    results = context.load_results(
        methods=[HISTORICAL_METHOD],
        download_results="auto",
    )
    return select_historical_baseline(
        results,
        dataset=dataset,
        tabarena_split=tabarena_split,
        task_id=task_id,
        repeat=repeat,
        fold=fold,
        sample=sample,
        suite=str(metadata.suite),
        config=str(metadata.config_default),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: Any, *, digits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_value(value: Any, *, digits: int, suffix: str) -> str:
    text = _format_number(value, digits=digits)
    return text if text == "n/a" else text + suffix


def _format_distribution(
    row: Mapping[str, Any],
    field: str,
    *,
    digits: int = 6,
    suffix: str = "",
) -> str:
    mean = _format_value(
        row.get(f"{field}_mean"), digits=digits, suffix=suffix
    )
    std = _format_value(row.get(f"{field}_std"), digits=digits, suffix=suffix)
    median = _format_value(row[field], digits=digits, suffix=suffix)
    q1 = row.get(f"{field}_q1")
    q3 = row.get(f"{field}_q3")
    if q1 is None or q3 is None:
        interval = median
    else:
        interval = (
            f"{median} [{_format_value(q1, digits=digits, suffix=suffix)}, "
            f"{_format_value(q3, digits=digits, suffix=suffix)}]"
        )
    return f"{mean} ± {std}; {interval}"


def _markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    summary: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    warmup_pairs: int,
    trials: int,
    inference_repeats: int,
) -> str:
    """Render three deliberately disjoint result tables."""
    matched = [
        row for row in summary if row["comparison_group"] == MATCHED_PARITY
    ]
    native = [
        row
        for row in summary
        if row["comparison_group"] == LOCAL_NATIVE_DEFAULT
    ]
    historical = [
        row
        for row in summary
        if row["comparison_group"] == HISTORICAL_TABARENA
    ]
    if len(matched) != 2 or len(native) != 2 or len(historical) != 1:
        raise RuntimeError(
            "Report requires two local rows per profile and one "
            "historical row."
        )

    matched_rows = [
        [
            str(row["method"]),
            _format_distribution(row, "metric_error"),
            _format_distribution(row, "fit_time_s"),
            _format_distribution(row, "first_inference_time_s"),
            _format_distribution(row, "inference_time_s"),
            _format_distribution(
                row, "fit_speedup_vs_original", digits=2, suffix="x"
            ),
            _format_distribution(
                row,
                "first_inference_speedup_vs_original",
                digits=2,
                suffix="x",
            ),
            _format_distribution(
                row, "inference_speedup_vs_original", digits=2, suffix="x"
            ),
        ]
        for row in matched
    ]
    native_rows = [
        [
            str(row["method"]),
            "8" if row["implementation"] == ORIGINAL else "1",
            _format_distribution(row, "metric_error"),
            _format_distribution(row, "fit_time_s"),
            _format_distribution(row, "first_inference_time_s"),
            _format_distribution(row, "inference_time_s"),
        ]
        for row in native
    ]
    archived = historical[0]
    historical_rows = [
        [
            str(archived["method"]),
            _format_number(archived["metric_error"]),
            _format_number(archived["fit_time_s"]),
            _format_number(archived["inference_time_s"]),
            str(archived["historical_suite"]),
            str(archived["historical_config"]),
        ]
    ]
    trial_word = "trial" if trials == 1 else "trials"
    call_word = "call" if inference_repeats == 1 else "calls"
    pair_word = "pair" if warmup_pairs == 1 else "pairs"

    sections = [
        "# TabICLv2 local comparison report",
        "",
        "This report keeps three result classes separate. A baseline-enriched "
        "report means that an archived TabArena baseline is loaded and shown "
        "beside newly measured local results for context. The local results "
        "are not recorded by TabArena, uploaded, or shared with another "
        "repository. TabArena may only download its own cached historical "
        "result through its normal read-only result API.",
        "",
        f"Local statistics use {trials} measured {trial_word} after "
        f"{warmup_pairs} discarded warm-up {pair_word}. Each trial records "
        "first inference separately and "
        f"then {inference_repeats} warmed inference {call_word}. "
        "Each local value is mean ± sample SD; median [Q1, Q3] across trials. "
        "Warmed inference first reduces each trial's calls to one trial "
        "median. CUDA work is synchronized around every timed region.",
        "",
        "## 1. Matched parity — authoritative same-workload comparison",
        "",
        "These are the only rows eligible for fair speedup claims. Both use "
        "one estimator, the same checkpoint, deterministic no-normalization "
        "and no-permutation preprocessing, caching, seed, split, resources, "
        "and disabled automatic acceleration/offload choices.",
        "",
        _markdown_table(
            [
                "Implementation",
                "RMSE",
                "Fit/setup s",
                "First inference s",
                "Warmed inference s",
                "Fit speedup",
                "First speedup",
                "Warmed speedup",
            ],
            matched_rows,
        ),
        "",
        "Matched speedups are computed per same-index trial as original "
        "duration divided by implementation duration, then summarized. "
        "Values above 1x are faster than the matched original row.",
        "",
        "## 2. Local native defaults — same machine, different workloads",
        "",
        "Warning: these rows use each implementation's native algorithm "
        "defaults. Original TabICLv2 uses eight estimators, while SDM "
        "TabICLv2 uses one. They are useful for out-of-box context but must "
        "not be treated as a parity comparison, ranking, or fair speedup.",
        "",
        _markdown_table(
            [
                "Implementation",
                "Estimators",
                "RMSE",
                "Fit/setup s",
                "First inference s",
                "Warmed inference s",
            ],
            native_rows,
        ),
        "",
        "Original native defaults also retain the none/power normalization "
        "mix, Latin feature shuffling, batch size eight, no KV cache, and "
        "automatic AMP, FlashAttention, and offload selection. SDM retains "
        "its current default recipe, one estimator, and cache-backed "
        "inference. Checkpoint and machine resources remain pinned.",
        "",
        "## 3. Historical TabArena baseline — external context, not "
        "timing comparable",
        "",
        "Warning: this row is loaded from TabArena's archived result cache. "
        "It was not rerun in this process, its machine and timing conditions "
        "are uncontrolled here, and it is excluded from every local speedup "
        "and statistical aggregate.",
        "",
        _markdown_table(
            [
                "Method",
                "Archived RMSE",
                "Archived fit s",
                "Archived inference s",
                "TabArena suite",
                "Config",
            ],
            historical_rows,
        ),
        "",
        "## Reproducibility",
        "",
        f"- Checkpoint SHA-256: {checkpoint_sha256}",
        "- Raw local trials: " + TRIALS_FILENAME,
        "- Raw warmed inference calls: " + INFERENCE_FILENAME,
        "- Machine-readable summary: " + SUMMARY_FILENAME,
        "- Full resolved configurations and environment: " + MANIFEST_FILENAME,
        "",
    ]
    return "\n".join(sections)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def environment_manifest(*, num_gpus: int) -> dict[str, object]:
    """Capture package, platform, accelerator, and thread provenance."""
    gpu = None
    if num_gpus:
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "device_count_visible": torch.cuda.device_count(),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "autogluon": _package_version("autogluon.tabular"),
        "tabarena": _package_version("tabarena"),
        "tabicl": _package_version("tabicl"),
        "gpu": gpu,
        "thread_environment": {
            "OMP_NUM_THREADS": "set per worker",
            "MKL_NUM_THREADS": "set per worker",
            "OPENBLAS_NUM_THREADS": "set per worker",
            "NUMEXPR_NUM_THREADS": "set per worker",
        },
    }


def _ensure_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_split_identity(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    split_values = {int(row["tabarena_split"]) for row in rows}
    identity = {
        (
            row["dataset"],
            int(row["task_id"]),
            int(row["repeat"]),
            int(row["fold"]),
            int(row["sample"]),
            int(row["train_rows"]),
            int(row["test_rows"]),
            int(row["features"]),
        )
        for row in rows
    }
    if len(split_values) != 1 or len(identity) != 1:
        raise RuntimeError(
            "Local workers did not use one identical task split."
        )
    return split_values.pop()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent benchmark or one internal worker invocation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_worker":
        return _worker_main(arguments[1:])

    args = _parse_args(arguments)
    seed = _validate_seed(args.seed)
    num_cpus = _validate_positive_int(args.num_cpus, option="--num-cpus")
    _validate_non_negative_int(args.task_id, option="--task-id")
    _validate_non_negative_int(args.repeat, option="--repeat")
    _validate_non_negative_int(args.fold, option="--fold")
    _validate_non_negative_int(args.sample, option="--sample")
    warmup_pairs = _validate_non_negative_int(
        args.warmup_pairs,
        option="--warmup-pairs",
    )
    trials = _validate_positive_int(args.trials, option="--trials")
    inference_repeats = _validate_positive_int(
        args.inference_repeats,
        option="--inference-repeats",
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
    DeviceAllocation.from_num_gpus(args.num_gpus)
    output_dir = _ensure_output_dir(args.output_dir)

    repository_root = Path(__file__).resolve().parents[2]
    manifest_path = output_dir / MANIFEST_FILENAME
    configurations = {
        profile: {
            implementation: resolved_configuration(
                profile=profile,
                implementation=implementation,
                checkpoint_path=checkpoint_path,
                seed=seed,
            )
            for implementation in IMPLEMENTATIONS
        }
        for profile in LOCAL_PROFILES
    }
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "sdm": {"commit": _git_commit(repository_root)},
        "tabarena": {"commit": _tabarena_commit()},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "repository": checkpoint_repository,
            "revision": checkpoint_revision,
        },
        "task": {
            "dataset": args.dataset,
            "task_id": args.task_id,
            "repeat": args.repeat,
            "fold": args.fold,
            "sample": args.sample,
        },
        "resources": {
            "num_cpus": num_cpus,
            "num_gpus": args.num_gpus,
            "device": str(
                DeviceAllocation.from_num_gpus(args.num_gpus).device
            ),
        },
        "protocol": {
            "fresh_subprocess_per_trial": True,
            "alternating_order": True,
            "warmup_pairs": warmup_pairs,
            "trials": trials,
            "inference_repeats": inference_repeats,
            "timer": "time.perf_counter_ns",
            "cuda_synchronize": True,
            "worker_timeout_s": worker_timeout_s,
        },
        "configurations": configurations,
        "historical_tabarena": {
            "method": HISTORICAL_METHOD,
            "config_type": HISTORICAL_CONFIG_TYPE,
            "method_subtype": HISTORICAL_METHOD_SUBTYPE,
            "access": "read-only cached result; auto-download on cache miss",
            "local_results_uploaded": False,
        },
        "environment": environment_manifest(num_gpus=args.num_gpus),
    }
    _write_json(manifest_path, manifest)

    try:
        base_spec = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "seed": seed,
            "num_cpus": num_cpus,
            "num_gpus": args.num_gpus,
            "task_id": args.task_id,
            "dataset": args.dataset,
            "repeat": args.repeat,
            "fold": args.fold,
            "sample": args.sample,
            "inference_repeats": inference_repeats,
        }
        with tempfile.TemporaryDirectory(
            prefix=".workers-",
            dir=output_dir,
        ) as scratch:
            scratch_dir = Path(scratch)

            def worker_runner(spec: Mapping[str, Any]) -> dict[str, Any]:
                return run_worker_subprocess(
                    spec,
                    scratch_dir=scratch_dir,
                    worker_timeout_s=worker_timeout_s,
                )

            trial_rows, inference_rows = run_local_benchmarks(
                base_spec,
                warmup_pairs=warmup_pairs,
                trials=trials,
                worker_runner=worker_runner,
            )

        expected_trials = len(LOCAL_PROFILES) * len(IMPLEMENTATIONS) * trials
        expected_inference = expected_trials * inference_repeats
        if len(trial_rows) != expected_trials:
            raise RuntimeError(
                f"Expected {expected_trials} local trial rows, "
                f"got {len(trial_rows)}."
            )
        if len(inference_rows) != expected_inference:
            raise RuntimeError(
                f"Expected {expected_inference} inference rows, "
                f"got {len(inference_rows)}."
            )
        tabarena_split = _validate_split_identity(trial_rows)
        summary = aggregate_local_results(
            trial_rows,
            trials=trials,
            inference_repeats=inference_repeats,
        )
        historical = load_historical_baseline(
            dataset=args.dataset,
            tabarena_split=tabarena_split,
            task_id=args.task_id,
            repeat=args.repeat,
            fold=args.fold,
            sample=args.sample,
        )
        summary.append(historical)

        trials_path = output_dir / TRIALS_FILENAME
        inference_path = output_dir / INFERENCE_FILENAME
        summary_path = output_dir / SUMMARY_FILENAME
        report_path = output_dir / REPORT_FILENAME
        _write_csv(trials_path, trial_rows, TRIAL_COLUMNS)
        _write_csv(inference_path, inference_rows, INFERENCE_COLUMNS)
        _write_csv(summary_path, summary, SUMMARY_COLUMNS)
        report_path.write_text(
            render_report(
                summary,
                checkpoint_sha256=checkpoint_sha256,
                warmup_pairs=warmup_pairs,
                trials=trials,
                inference_repeats=inference_repeats,
            )
        )

        artifact_paths = (
            trials_path,
            inference_path,
            summary_path,
            report_path,
        )
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "task": {
                    **manifest["task"],
                    "tabarena_split": tabarena_split,
                },
                "artifacts": {
                    path.name: {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in artifact_paths
                },
            }
        )
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_json(manifest_path, manifest)
        raise

    print(f"Report: {report_path}")  # noqa: T201
    print(f"Summary: {summary_path}")  # noqa: T201
    print(f"Manifest: {manifest_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
