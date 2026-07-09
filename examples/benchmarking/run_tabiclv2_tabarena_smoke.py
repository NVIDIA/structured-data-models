"""Run the SDM-owned TabICLv2 regression smoke through TabArena.

From the repository root, in the dedicated TabArena environment:

.. code-block:: console

    python -m examples.benchmarking.run_tabiclv2_tabarena_smoke \
        --checkpoint-path /path/to/tabicl-regressor-v2-20260212.ckpt \
        --checkpoint-sha256 \
        0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a \
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
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from examples.benchmarking.tabiclv2_tabarena_model import (
    DeviceAllocation,
    SDMTabICLv2Model,
)

DATASET = "QSAR_fish_toxicity"
CHECKPOINT_REPOSITORY = "jingang/TabICL"
CHECKPOINT_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
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
            "repository": CHECKPOINT_REPOSITORY,
            "revision": CHECKPOINT_REVISION,
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
        },
    }
    # This precedes TabArena job execution and therefore model timing.
    _write_manifest(manifest_path, manifest)

    experiments = build_smoke_experiments(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=args.num_gpus,
    )

    from tabarena.contexts import TabArenaContext

    context = TabArenaContext()
    try:
        context.build_and_run_jobs(
            experiments,
            expname=str(results_dir),
            build_kwargs={
                "dataset_names": [DATASET],
                "split_indices": "lite",
            },
            new_result_prefix="[New] ",
            debug_mode=args.debug_mode,
        )
        evaluation_dir = output_dir / "evaluation"
        leaderboard = context.compare(output_dir=evaluation_dir)
        leaderboard_website = context.leaderboard_to_website_format(
            leaderboard=leaderboard
        )
        if leaderboard_website.empty:
            raise RuntimeError("TabArena completed without a result row.")

        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "results_dir": str(results_dir),
                "evaluation_dir": str(evaluation_dir),
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

    print("\n=== TabArena leaderboard ===")  # noqa: T201
    print(leaderboard_website.to_markdown(index=False))  # noqa: T201
    print(f"\nManifest: {manifest_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
