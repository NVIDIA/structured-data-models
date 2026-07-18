"""Run TabArena one dataset at a time and combine successful SDM results."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from examples.tabiclv2_tabarena.model import SDMTabICLv2Model
from examples.tabiclv2_tabarena.runner import (
    RunConfig,
    _prepare_output_root,
    add_common_arguments,
    config_from_args,
)

DEFAULT_EXCLUDED_DATASETS = ("APSFailure",)


@dataclass(frozen=True)
class DatasetRun:
    """Terminal state for one independently executed dataset."""

    dataset: str
    output_root: Path
    returncode: int
    complete: bool


def main() -> None:
    """Execute the selected suite in failure-isolated dataset subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=list(DEFAULT_EXCLUDED_DATASETS),
        help="Datasets to skip (APSFailure is excluded by default).",
    )
    args = parser.parse_args()
    run_campaign(
        config_from_args(args), excluded_datasets=args.exclude_datasets
    )


def run_campaign(config: RunConfig, *, excluded_datasets: list[str]) -> None:
    """Run isolated TabArena datasets and combine their report."""
    _prepare_output_root(config.output_root, resume=config.resume)
    datasets = _discover_datasets(config)
    excluded = set(excluded_datasets)
    selected = [dataset for dataset in datasets if dataset not in excluded]
    if not selected:
        raise ValueError(
            "No datasets remain after applying '--exclude-datasets'"
        )

    _write_json(
        config.output_root / "campaign_metadata.json",
        {
            "config": {
                **asdict(config),
                "output_root": str(config.output_root),
            },
            "excluded_datasets": sorted(excluded),
            "selected_datasets": selected,
        },
    )
    records = [_run_dataset(config, dataset) for dataset in selected]
    _write_campaign_report(
        config.output_root, records, excluded_datasets=excluded
    )
    _raise_if_campaign_incomplete(records, allow_partial=config.allow_partial)


def _discover_datasets(config: RunConfig) -> list[str]:
    """Resolve dataset names through TabArena without a copied static list."""
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
    ).build_experiments(num_cpus=config.num_cpus, num_gpus=config.num_gpus)
    jobs = TabArenaContext().build_jobs(
        experiments,
        task_subset=TaskSubset(
            subset=config.subset,
            dataset_names=config.datasets,
        ),
    )
    return sorted({job.task.dataset for job in jobs})


def _run_dataset(config: RunConfig, dataset: str) -> DatasetRun:
    """Run one dataset in a subprocess so OOM failures are isolated."""
    slug = _dataset_slug(dataset)
    output_root = config.output_root / "datasets" / slug
    log_path = config.output_root / "logs" / f"{slug}.log"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = _dataset_command(config, dataset, output_root)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).parents[2],
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    complete = _is_complete(output_root)
    return DatasetRun(
        dataset=dataset,
        output_root=output_root,
        returncode=completed.returncode,
        complete=complete,
    )


def _dataset_command(
    config: RunConfig, dataset: str, output_root: Path
) -> list[str]:
    """Return the exact child command for one dataset."""
    command = [
        sys.executable,
        "-m",
        "examples.tabiclv2_tabarena.run_local",
        "--output-root",
        str(output_root),
        "--num-estimators",
        str(config.num_estimators),
        "--datasets",
        dataset,
    ]
    if config.num_cpus is not None:
        command.extend(["--num-cpus", str(config.num_cpus)])
    if config.num_gpus is not None:
        command.extend(["--num-gpus", str(config.num_gpus)])
    if config.resume:
        command.append("--resume")
    if config.allow_partial:
        command.append("--allow-partial")
    if config.outer:
        command.append("--outer")
    if config.subset is not None:
        command.extend(["--subset", *config.subset])
    return command


def _write_campaign_report(
    output_root: Path,
    records: list[DatasetRun],
    *,
    excluded_datasets: set[str],
) -> None:
    """Combine completed per-dataset reports without filling missing values."""
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    completed = [record for record in records if record.complete]
    failed = [record for record in records if not record.complete]
    frames = [
        pd.read_csv(record.output_root / "report" / "results_per_split.csv")
        for record in completed
    ]
    if frames:
        results = pd.concat(frames, ignore_index=True)
        results.to_csv(report_dir / "results_per_split.csv", index=False)
        (
            results.groupby("problem_type", dropna=False)["metric_error"]
            .agg(["mean", "count"])
            .reset_index()
            .to_csv(report_dir / "metric_summary.csv", index=False)
        )
    _write_json(
        report_dir / "campaign_status.json",
        {
            "completed_datasets": [record.dataset for record in completed],
            "failed_datasets": [
                {
                    "dataset": record.dataset,
                    "log": str(
                        output_root
                        / "logs"
                        / f"{_dataset_slug(record.dataset)}.log"
                    ),
                    "output_root": str(record.output_root),
                    "returncode": record.returncode,
                }
                for record in failed
            ],
            "excluded_datasets": sorted(excluded_datasets),
        },
    )


def _raise_if_campaign_incomplete(
    records: list[DatasetRun], *, allow_partial: bool
) -> None:
    """Make failed dataset subprocesses visible to shell automation."""
    failed = [record.dataset for record in records if not record.complete]
    if failed and not allow_partial:
        raise RuntimeError(
            f"{len(failed)} dataset runs failed: " + ", ".join(failed) + ". "
            "Inspect report/campaign_status.json, or pass "
            "'--allow-partial' to accept the completed-dataset report."
        )


def _is_complete(output_root: Path) -> bool:
    path = output_root / "completion.json"
    if not path.is_file():
        return False
    return bool(json.loads(path.read_text())["complete"])


def _dataset_slug(dataset: str) -> str:
    """Create a stable, filesystem-safe directory name from a dataset name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", dataset).strip("-")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
