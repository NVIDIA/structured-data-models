"""Shared full-suite TabArena runner for the local TabICLv2 examples."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from examples.tabiclv2_tabarena.model import SDMTabICLv2Model

if TYPE_CHECKING:
    import pandas as pd
    from tabarena.benchmark.experiment.job import Job


@dataclass(frozen=True)
class RunConfig:
    """Configuration shared by the local and Ray-backed examples."""

    output_root: Path
    num_estimators: int
    num_cpus: int | None
    num_gpus: int | None
    resume: bool
    allow_partial: bool
    outer: bool
    subset: list[str] | None
    datasets: list[str] | None


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by both example entry points."""
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--outer", action="store_true")
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--datasets", nargs="+")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Validate CLI arguments and convert them into an immutable config."""
    if args.num_estimators < 1:
        raise ValueError("'--num-estimators' must be positive")
    if args.num_cpus is not None and args.num_cpus < 1:
        raise ValueError("'--num-cpus' must be positive")
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("'--num-gpus' cannot be negative")
    return RunConfig(
        output_root=args.output_root.resolve(),
        num_estimators=args.num_estimators,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        resume=args.resume,
        allow_partial=args.allow_partial,
        outer=args.outer,
        subset=args.subset,
        datasets=args.datasets,
    )


def run(config: RunConfig, *, debug_mode: bool) -> None:
    """Run TabArena, then write results and reports to ``output_root``."""
    _prepare_output_root(config.output_root, resume=config.resume)
    _validate_or_write_run_signature(config=config, debug_mode=debug_mode)
    _write_run_metadata(config=config, debug_mode=debug_mode)

    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.benchmark.task.metadata.collection import TaskSubset
    from tabarena.contexts import TabArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    generator = ConfigGenerator(
        search_space={},
        model_cls=SDMTabICLv2Model,
        manual_configs=[{"num_estimators": config.num_estimators}],
    )
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(generator, 0)],
        outer_experiments=config.outer,
    ).build_experiments(
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
    )
    context = TabArenaContext()
    task_subset = TaskSubset(
        subset=config.subset,
        dataset_names=config.datasets,
    )
    jobs = context.build_jobs(experiments, task_subset=task_subset)
    _write_job_manifest(config.output_root, jobs=jobs)

    results = _run_context_jobs(
        context, jobs, config=config, debug_mode=debug_mode
    )
    complete = len(results) == len(jobs)
    _write_completion_metadata(
        config.output_root,
        planned_jobs=len(jobs),
        completed_jobs=len(results),
        complete=complete,
    )

    report_dir = config.output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_run_report(context, report_dir)

    if not complete and not config.allow_partial:
        raise RuntimeError(
            f"Only {len(results)} of {len(jobs)} planned jobs completed. "
            f"See {config.output_root} for cached results and diagnostics."
        )


def _write_run_report(context: Any, report_dir: Path) -> None:
    """Write completed SDM results, including empty partial-run status."""
    new_results = context._registered_new_results()
    if new_results is None:
        _write_report_status(
            report_dir,
            leaderboard_generated=False,
            reason="No SDM jobs completed successfully.",
        )
        return
    _write_results_report(new_results, report_dir)
    try:
        leaderboard = context.compare(
            output_dir=report_dir,
            only_valid_tasks=True,
            new_methods_only=True,
            fillna=None,
        )
    except (AssertionError, ValueError) as error:
        _write_report_status(
            report_dir, leaderboard_generated=False, reason=str(error)
        )
    else:
        leaderboard.to_csv(report_dir / "leaderboard.csv")
        leaderboard.to_markdown(report_dir / "leaderboard.md")
        context.leaderboard_to_website_format(leaderboard=leaderboard).to_csv(
            report_dir / "leaderboard_website.csv", index=False
        )
        _write_report_status(report_dir, leaderboard_generated=True)


def _run_context_jobs(
    context: Any, jobs: list[Job], *, config: RunConfig, debug_mode: bool
) -> list[dict[str, Any]]:
    """Run jobs without discarding completed work after a permitted failure."""
    return context.run_jobs(
        jobs,
        expname=config.output_root,
        new_result_prefix="[SDM] ",
        debug_mode=debug_mode,
        # Preserve completed jobs; do not raise at the first failure.
        raise_on_failure=not config.allow_partial,
    )


def _prepare_output_root(output_root: Path, *, resume: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(
            f"Output root {output_root} is non-empty. "
            "Pass '--resume' to reuse TabArena's cached results."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_or_write_run_signature(
    *, config: RunConfig, debug_mode: bool
) -> None:
    """Reject ``--resume`` when it would reuse a different result cache."""
    path = config.output_root / "run_signature.json"
    signature = _run_signature(config=config, debug_mode=debug_mode)
    if not config.resume:
        _write_json(path, signature)
        return
    if not path.is_file():
        raise RuntimeError(
            f"Cannot safely resume {config.output_root}: "
            "run_signature.json is missing. Use a fresh output root."
        )
    previous = json.loads(path.read_text())
    if previous != signature:
        raise RuntimeError(
            f"Cannot safely resume {config.output_root}: "
            "its run signature differs. Use a fresh output root "
            "for a different benchmark configuration."
        )


def _run_signature(
    *, config: RunConfig, debug_mode: bool
) -> dict[str, object]:
    """Return fields that affect TabArena cache identity or result meaning."""
    return {
        "signature_version": 1,
        "num_estimators": config.num_estimators,
        "num_cpus": config.num_cpus,
        "num_gpus": config.num_gpus,
        "outer": config.outer,
        "subset": sorted(config.subset) if config.subset is not None else None,
        "datasets": sorted(config.datasets)
        if config.datasets is not None
        else None,
        "debug_mode": debug_mode,
        "sdm_revision": _git_revision(),
    }


def _write_run_metadata(*, config: RunConfig, debug_mode: bool) -> None:
    metadata = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
        "debug_mode": debug_mode,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sdm_revision": _git_revision(),
    }

    _write_json(config.output_root / "run_metadata.json", metadata)


def _write_job_manifest(output_root: Path, *, jobs: list[Job]) -> None:
    rows = [
        {
            "experiment": job.experiment.name,
            "dataset": job.task.dataset,
            "fold": job.task.fold,
            "repeat": job.task.repeat,
        }
        for job in jobs
    ]

    _write_json(output_root / "planned_jobs.json", rows)


def _write_completion_metadata(
    output_root: Path,
    *,
    planned_jobs: int,
    completed_jobs: int,
    complete: bool,
) -> None:

    _write_json(
        output_root / "completion.json",
        {
            "planned_jobs": planned_jobs,
            "completed_jobs": completed_jobs,
            "complete": complete,
        },
    )


def _write_results_report(results: pd.DataFrame, report_dir: Path) -> None:
    results.to_csv(report_dir / "results_per_split.csv", index=False)
    summary = (
        results.groupby("problem_type", dropna=False)["metric_error"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary.to_csv(report_dir / "metric_summary.csv", index=False)


def _write_report_status(
    report_dir: Path, *, leaderboard_generated: bool, reason: str | None = None
) -> None:
    value: dict[str, object] = {"leaderboard_generated": leaderboard_generated}
    if reason is not None:
        value["reason"] = reason
    _write_json(report_dir / "report_status.json", value)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
