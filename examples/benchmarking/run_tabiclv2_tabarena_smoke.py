"""Run the SDM-owned TabICLv2 regression smoke through TabArena.

From the repository root, in the dedicated TabArena environment:

.. code-block:: console

    python -m examples.benchmarking.run_tabiclv2_tabarena_smoke \
        --checkpoint-path /path/to/tabicl-regressor-v2-20260212.ckpt \
        --checkpoint-sha256 \
        0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a \
        --checkpoint-repository jingang/TabICL \
        --checkpoint-revision \
        4dcd344ece2c00be9e831fdd35bed57b5ad83e19 \
        --seed 0

The checkpoint is verified locally before a job starts; this runner never
downloads weights.  Invoke it as a module so worker processes can import the
model class at ``examples.benchmarking.tabiclv2_tabarena_model``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from examples.benchmarking.tabiclv2_tabarena_model import (
    DeviceAllocation,
    SDMTabICLv2Model,
)

DATASET = "QSAR_fish_toxicity"
COMPARISON_FILENAME = "tabiclv2_comparison.csv"
NEW_RESULT_PREFIX = "[New] "
ORIGINAL_CONFIG_TYPE = "TABICLV2"
ORIGINAL_METHOD_SUBTYPE = "default"
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = Path("artifacts/tabiclv2_tabarena_smoke")
MAX_SEED = 2**63 - 1
NUM_ESTIMATORS = 1


def build_smoke_experiments(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
    num_cpus: int | None,
    num_gpus: int,
):
    """Build the one outer/no-preprocessing experiment without executing it."""
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.utils.config_utils import ConfigGenerator

    config = ConfigGenerator(
        search_space={},
        model_cls=SDMTabICLv2Model,
        manual_configs=[
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "seed": seed,
                "num_estimators": NUM_ESTIMATORS,
            }
        ],
    )
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[(config, 0)],
        outer_experiments=True,
        model_agnostic_preprocessing=False,
    )
    experiments = bundle.build_experiments(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )

    # In the pinned TabArena outer-model path, this flag is owned by the
    # execution wrapper rather than the bundle's feature-generator kwargs.
    # Keep both switches explicit: no AutoGluon feature generator runs before
    # SDM validates and transforms the raw frame.
    for experiment in experiments:
        experiment.method_kwargs["preprocess_data"] = False

    if len(experiments) != 1:
        raise RuntimeError(
            f"Expected one TabArena smoke experiment, got {len(experiments)}."
        )
    if experiments[0].method_kwargs.get("preprocess_data") is not False:
        raise RuntimeError(
            "Failed to disable TabArena model-agnostic preprocessing."
        )
    return experiments


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=Path,
        help="Local TabICLv2 regression checkpoint; no download is attempted.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected SHA-256 of --checkpoint-path.",
    )
    parser.add_argument(
        "--checkpoint-repository",
        required=True,
        help="Caller-supplied repository or source for the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-revision",
        required=True,
        help="Caller-supplied revision for the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fit-time seed for deterministic SDM preprocessing (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "New directory for this run's manifest, result cache, and "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help=(
            "CPUs allocated to the AutoGluon model "
            "(default: local physical-core count)."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use CPU (0, default) or the worker's single CUDA device (1).",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help=(
            "Run TabArena jobs in-process instead of through its normal "
            "worker path."
        ),
    )
    return parser.parse_args(argv)


def _resolve_num_cpus(num_cpus: int | None) -> int:
    """Resolve the CPU allocation once so the manifest records it exactly."""
    if num_cpus is not None:
        if num_cpus < 1:
            raise ValueError("--num-cpus must be at least 1.")
        return num_cpus

    from autogluon.common.utils.resource_utils import ResourceManager

    return ResourceManager.get_cpu_count(only_physical_cores=True)


def _validate_seed(seed: int) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SEED
    ):
        raise ValueError(
            "--seed must be an integer between 0 and "
            f"{MAX_SEED} (got {seed!r})."
        )
    return seed


def _validate_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[Path, str]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist or is not a file: {checkpoint_path}"
        )

    expected_sha256 = expected_sha256.strip().lower()
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ValueError(
            "--checkpoint-sha256 must be a 64-character hex digest."
        )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    return checkpoint_path, actual_sha256


def _validate_provenance(value: str, *, option: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{option} must not be empty.")
    return value


def _require_finite_value(
    value: object,
    *,
    field: str,
    non_negative: bool = True,
) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Expected finite numeric result field '{field}'.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Expected finite numeric result field '{field}'."
        ) from error
    if not math.isfinite(number):
        raise RuntimeError(f"Result field '{field}' must be finite.")
    if non_negative and number < 0:
        raise RuntimeError(f"Result field '{field}' must be non-negative.")
    return number


def _validate_run_results(
    results: Sequence[Mapping[str, Any]],
) -> str:
    """Validate the one raw TabArena result and return its registered name."""
    if len(results) != 1:
        raise RuntimeError(
            f"Expected exactly one SDM TabICLv2 result, got {len(results)}."
        )

    result = results[0]
    if not isinstance(result, Mapping):
        raise RuntimeError("Expected the TabArena result to be a mapping.")

    framework = result.get("framework")
    expected_framework = f"{SDMTabICLv2Model.ag_name}_c1"
    if not isinstance(framework, str) or not framework:
        raise RuntimeError("TabArena result is missing its framework name.")
    if framework != expected_framework:
        raise RuntimeError(
            "Expected the executed framework to be the one configured SDM "
            f"TabICLv2 method {expected_framework!r}, got {framework!r}."
        )

    task_metadata = result.get("task_metadata")
    if not isinstance(task_metadata, Mapping):
        raise RuntimeError("TabArena result is missing task metadata.")
    expected_task = {
        "name": DATASET,
        "fold": 0,
        "repeat": 0,
        "split_idx": 0,
    }
    for field, expected in expected_task.items():
        actual = task_metadata.get(field)
        if actual != expected:
            raise RuntimeError(
                f"Expected task metadata '{field}' to be {expected!r}, "
                f"got {actual!r}."
            )

    if result.get("problem_type") != "regression":
        raise RuntimeError("Expected one regression result from TabArena.")
    if result.get("metric") != "rmse":
        raise RuntimeError("Expected the TabArena result metric to be RMSE.")
    for field in ("metric_error", "time_train_s", "time_infer_s"):
        _require_finite_value(result.get(field), field=field)

    return f"{NEW_RESULT_PREFIX}{framework}"


def _require_frame_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{frame_name} is missing required columns: {missing!r}."
        )


def _validate_new_comparison(
    leaderboard: pd.DataFrame,
    results: pd.DataFrame,
    *,
    expected_method: str,
) -> None:
    """Require one registered SDM leaderboard and per-split result row."""
    import pandas as pd

    if not isinstance(leaderboard, pd.DataFrame) or len(leaderboard) != 1:
        count = (
            len(leaderboard) if isinstance(leaderboard, pd.DataFrame) else 0
        )
        raise RuntimeError(
            f"Expected exactly one new SDM leaderboard row, got {count}."
        )
    if "method" in leaderboard.columns:
        leaderboard_methods = leaderboard["method"].tolist()
    else:
        leaderboard_methods = leaderboard.index.tolist()
    if leaderboard_methods != [expected_method]:
        raise RuntimeError(
            "New leaderboard row does not match the executed SDM method: "
            f"expected {expected_method!r}, got {leaderboard_methods!r}."
        )

    if not isinstance(results, pd.DataFrame) or len(results) != 1:
        count = len(results) if isinstance(results, pd.DataFrame) else 0
        raise RuntimeError(
            f"Expected exactly one new SDM evaluated result row, got {count}."
        )
    _require_frame_columns(
        results,
        (
            "method",
            "dataset",
            "fold",
            "metric",
            "problem_type",
            "metric_error",
            "time_train_s",
            "time_infer_s",
        ),
        frame_name="New SDM results",
    )
    result = results.iloc[0]
    expected = {
        "method": expected_method,
        "dataset": DATASET,
        "fold": 0,
        "metric": "rmse",
        "problem_type": "regression",
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise RuntimeError(
                f"Expected new result '{field}' to be {expected_value!r}, "
                f"got {result[field]!r}."
            )
    for field in ("metric_error", "time_train_s", "time_infer_s"):
        _require_finite_value(result[field], field=field)


def _write_tabiclv2_comparison(
    path: Path,
    all_results: pd.DataFrame,
    new_results: pd.DataFrame,
    *,
    expected_new_method: str,
) -> None:
    """Write the paired original-versus-SDM per-split comparison."""
    import pandas as pd

    if not isinstance(all_results, pd.DataFrame):
        raise RuntimeError("Expected TabArena comparison results as a frame.")
    required = (
        "method",
        "dataset",
        "fold",
        "metric",
        "problem_type",
        "metric_error",
        "time_train_s",
        "time_infer_s",
        "config_type",
        "method_subtype",
    )
    _require_frame_columns(
        all_results,
        required,
        frame_name="Baseline-enriched results",
    )

    task_results = all_results.loc[
        (all_results["dataset"] == DATASET) & (all_results["fold"] == 0)
    ]
    original = task_results.loc[
        (task_results["config_type"] == ORIGINAL_CONFIG_TYPE)
        & (task_results["method_subtype"] == ORIGINAL_METHOD_SUBTYPE)
    ].copy()
    sdm = task_results.loc[
        task_results["method"] == expected_new_method
    ].copy()
    if len(original) != 1:
        raise RuntimeError(
            "Expected exactly one original TabArena TabICLv2 default result, "
            f"got {len(original)}."
        )
    if len(sdm) != 1:
        raise RuntimeError(
            "Expected exactly one SDM TabICLv2 result in the full comparison, "
            f"got {len(sdm)}."
        )

    new_result = new_results.iloc[0]
    for field in (
        "method",
        "dataset",
        "fold",
        "metric",
        "metric_error",
        "time_train_s",
        "time_infer_s",
    ):
        if sdm.iloc[0][field] != new_result[field]:
            raise RuntimeError(
                "Full and new-only SDM comparison results disagree on "
                f"'{field}'."
            )

    for frame in (original, sdm):
        row = frame.iloc[0]
        if row["problem_type"] != "regression" or row["metric"] != "rmse":
            raise RuntimeError("Expected paired TabICLv2 RMSE results.")
        for field in ("metric_error", "time_train_s", "time_infer_s"):
            _require_finite_value(row[field], field=field)

    original.insert(0, "implementation", "original_tabarena")
    sdm.insert(0, "implementation", "sdm")
    columns = (
        "implementation",
        "method",
        "dataset",
        "fold",
        "metric",
        "metric_error",
        "time_train_s",
        "time_infer_s",
    )
    comparison = pd.concat([original, sdm], ignore_index=True).loc[:, columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)


def _git_commit(path: Path) -> str:
    """Return a repository commit, or a concrete unavailable marker."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _find_git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _tabarena_commit() -> str:
    tabarena = importlib.import_module("tabarena")
    module_file = getattr(tabarena, "__file__", None)
    if module_file is None:
        return "unavailable"
    module_path = Path(module_file).resolve()
    root = _find_git_root(module_path.parent)
    return _git_commit(root) if root is not None else "unavailable"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the smoke benchmark and write its provenance manifest."""
    args = _parse_args(argv)
    seed = _validate_seed(args.seed)
    num_cpus = _resolve_num_cpus(args.num_cpus)
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

    output_dir = args.output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    if results_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing TabArena results at {results_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "task": DATASET,
        "sdm": {"commit": _git_commit(repo_root)},
        "tabarena": {"commit": _tabarena_commit()},
        "checkpoint": {
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
        "configuration": {
            "outer_experiments": True,
            "model_agnostic_preprocessing": False,
            "preprocess_data": False,
            "model": SDMTabICLv2Model.__name__,
            "seed": seed,
            "num_estimators": NUM_ESTIMATORS,
            "debug_mode": args.debug_mode,
        },
    }
    # This precedes TabArena job execution and therefore model timing.
    _write_manifest(manifest_path, manifest)

    try:
        experiments = build_smoke_experiments(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            seed=seed,
            num_cpus=num_cpus,
            num_gpus=args.num_gpus,
        )

        from tabarena.contexts import TabArenaContext

        context = TabArenaContext()
        run_results = context.build_and_run_jobs(
            experiments,
            expname=str(results_dir),
            build_kwargs={
                "dataset_names": [DATASET],
                "split_indices": "lite",
            },
            new_result_prefix=NEW_RESULT_PREFIX,
            debug_mode=args.debug_mode,
        )
        expected_new_method = _validate_run_results(run_results)

        evaluation_dir = output_dir / "evaluation"
        # The full comparison enriches the new local result with TabArena's
        # historical baselines.  It supplies comparison context, never proof
        # that this run completed; the raw result and new-only view provide
        # that proof.
        _, all_results = context.compare(
            output_dir=evaluation_dir,
            return_results=True,
        )
        leaderboard, new_results = context.compare(
            output_dir=None,
            new_methods_only=True,
            return_results=True,
        )
        _validate_new_comparison(
            leaderboard,
            new_results,
            expected_method=expected_new_method,
        )

        comparison_path = evaluation_dir / COMPARISON_FILENAME
        _write_tabiclv2_comparison(
            comparison_path,
            all_results,
            new_results,
            expected_new_method=expected_new_method,
        )
        leaderboard_website = context.leaderboard_to_website_format(
            leaderboard=leaderboard
        )
        if len(leaderboard_website) != 1:
            raise RuntimeError(
                "Expected exactly one formatted SDM leaderboard row, "
                f"got {len(leaderboard_website)}."
            )

        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "results_dir": str(results_dir),
                "evaluation_dir": str(evaluation_dir),
                "comparison_path": str(comparison_path),
            }
        )
        _write_manifest(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_manifest(manifest_path, manifest)
        raise

    print("\n=== TabArena SDM leaderboard row ===")  # noqa: T201
    print(leaderboard_website.to_markdown(index=False))  # noqa: T201
    print(f"\nComparison: {comparison_path}")  # noqa: T201
    print(f"\nManifest: {manifest_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
