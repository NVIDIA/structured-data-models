# ruff: noqa: PLC0415

"""Run the local TabICLv2 model on TabArena in one process."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
    from tabarena.benchmark.experiment.job import Job


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one local TabArena run."""

    output_root: Path
    num_estimators: int
    num_cpus: int | None
    num_gpus: int | None
    mode: Literal["autogluon-compatible", "sdm-native"]
    outer: bool
    subset: list[str] | None
    datasets: list[str] | None


def main() -> None:
    """Parse command-line arguments and run the selected TabArena jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(config_from_args(parser.parse_args()))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for the local TabArena runner."""
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument(
        "--mode",
        choices=("autogluon-compatible", "sdm-native"),
        default="autogluon-compatible",
        help="Choose AutoGluon-compatible or SDM-native preprocessing.",
    )
    parser.add_argument(
        "--outer",
        action="store_true",
        help="Use TabArena outer experiments in AutoGluon-compatible mode.",
    )
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--datasets", nargs="+")


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Validate command-line arguments and return a run configuration."""
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
        mode=args.mode,
        outer=args.outer,
        subset=args.subset,
        datasets=args.datasets,
    )


def run(config: RunConfig) -> None:
    """Run selected TabArena jobs locally and write their result records."""
    _prepare_output_root(config.output_root)

    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.benchmark.task.metadata.collection import TaskSubset
    from tabarena.contexts import TabArenaContext

    experiments = _build_experiments(
        config,
        bundle_cls=TabArenaV0pt1ExperimentBundle,
    )
    context = TabArenaContext()
    jobs = context.build_jobs(
        experiments,
        task_subset=TaskSubset(
            subset=config.subset,
            dataset_names=config.datasets,
        ),
    )
    _run_jobs(context, jobs, output_root=config.output_root)
    results = context._registered_new_results()
    if results is None:
        raise RuntimeError("No TabArena jobs completed successfully")
    _write_results_report(
        results,
        config.output_root / "report",
        mode=config.mode,
    )


def _build_experiments(config: RunConfig, *, bundle_cls: type) -> list:
    """Build the selected TabArena integration mode with fixed parameters."""
    if config.mode == "autogluon-compatible":
        from examples.tabiclv2_tabarena.model import SDMTabICLv2Model
        from tabarena.utils.config_utils import ConfigGenerator

        generator = ConfigGenerator(
            search_space={},
            model_cls=SDMTabICLv2Model,
            manual_configs=[{"num_estimators": config.num_estimators}],
        )
        return bundle_cls(
            models=[(generator, 0)],
            outer_experiments=config.outer,
        ).build_experiments(
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus,
        )

    from examples.tabiclv2_tabarena.sdm_system import SDMTabICLv2System
    from tabarena.utils.config_utils import SystemConfigGenerator

    generator = SystemConfigGenerator(
        model_cls=SDMTabICLv2System,
        name="SDMTabICLv2System",
        manual_configs=[{"num_estimators": config.num_estimators}],
    )
    return bundle_cls(
        models=[(generator, 0)],
        system_experiments=True,
    ).build_experiments(
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
    )


def _prepare_output_root(output_root: Path) -> None:
    """Require a fresh output directory so TabArena results are not mixed."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root {output_root} is non-empty. Choose a fresh path."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _run_jobs(
    context: Any,
    jobs: list[Job],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Execute every selected job through TabArena's local debug path."""
    return context.run_jobs(
        jobs,
        expname=output_root,
        new_result_prefix="[SDM] ",
        debug_mode=True,
    )


def _write_results_report(
    results: pd.DataFrame,
    report_dir: Path,
    *,
    mode: str | None = None,
) -> None:
    """Write the completed SDM result records as a concise CSV report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report = results.copy()
    if mode is not None:
        report.insert(0, "integration_mode", mode)
    report.to_csv(report_dir / "results_per_split.csv", index=False)


if __name__ == "__main__":
    main()
