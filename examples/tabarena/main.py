r"""Run TabICLv2 on TabArena."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import SDMTabICLv2System
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    help="Run only the selected TabArena dataset.",
)
args = parser.parse_args()

result_dir = Path(__file__).parent.parent / "tabarena_out" / "TabICLv2"
result_dir.mkdir(parents=True, exist_ok=True)

generator = SystemConfigGenerator(
    model_cls=SDMTabICLv2System,
    name="SDMTabICLv2System",
    manual_configs=[{}],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    system_experiments=True,
).build_experiments()

context = TabArenaContext()
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    register=False,
    build_kwargs=(
        {"dataset_names": [args.dataset]} if args.dataset is not None else None
    ),
)
