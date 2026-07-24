"""Run SDM-native TabICLv2 with the official eight-member ensemble policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from examples.tabiclv2_tabarena.run_local import (
    _prepare_output_root,
    _run_jobs,
    _write_results_report,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class OfficialPolicyRunConfig:
    """Configuration for the SDM-native official-policy control."""

    output_root: Path
    num_cpus: int | None
    num_gpus: int | None
    subset: list[str] | None
    datasets: list[str] | None
    random_state: int


def main() -> None:
    """Parse command-line arguments and execute the policy control."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(config_from_args(parser.parse_args()))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments unique to the fixed official-policy control."""
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--random-state", type=int, default=42)


def config_from_args(args: argparse.Namespace) -> OfficialPolicyRunConfig:
    """Validate command-line arguments and return the run configuration."""
    if args.num_cpus is not None and args.num_cpus < 1:
        raise ValueError("'--num-cpus' must be positive")
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("'--num-gpus' cannot be negative")
    return OfficialPolicyRunConfig(
        output_root=args.output_root.resolve(),
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        subset=args.subset,
        datasets=args.datasets,
        random_state=args.random_state,
    )


def run(config: OfficialPolicyRunConfig) -> None:
    """Run the policy-equivalent SDM-native TabArena control."""
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
    results: pd.DataFrame | None = context._registered_new_results()
    if results is None:
        raise RuntimeError("No TabArena jobs completed successfully")
    _write_results_report(
        results,
        config.output_root / "report",
        mode="sdm-native-official-policy",
    )


def _build_experiments(
    config: OfficialPolicyRunConfig, *, bundle_cls: type
) -> list:
    """Build the distinct TabArena external-system configuration."""
    from examples.tabiclv2_tabarena.sdm_system import (
        SDMTabICLv2OfficialPolicySystem,
    )
    from tabarena.utils.config_utils import SystemConfigGenerator

    generator = SystemConfigGenerator(
        model_cls=SDMTabICLv2OfficialPolicySystem,
        name="SDMTabICLv2OfficialPolicySystem",
        manual_configs=[{"random_state": config.random_state}],
    )
    return bundle_cls(
        models=[(generator, 0)],
        system_experiments=True,
    ).build_experiments(
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
    )


if __name__ == "__main__":
    main()
