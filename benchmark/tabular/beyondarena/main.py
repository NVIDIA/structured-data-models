r"""Run an SDM tabular model on BeyondArena."""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

from benchmark.tabular.system import MODEL_CONFIGS, SDMSystem

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    default="tabiclv2",
    help="SDM model to benchmark.",
)
parser.add_argument(
    "--dataset",
    help="Run only the selected BeyondArena dataset.",
)
parser.add_argument(
    "--subset",
    action="append",
    help="Filter tasks; repeat to combine filters (default: core).",
)
parser.add_argument(
    "--max_context_size",
    type=int,
    help="Subsample the context to at most this many rows.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Prediction batch size.",
)
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    Path(__file__).parent.parent / "beyondarena_out" / model_config.name
)
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": args.model,
    "max_context_size": args.max_context_size,
    "batch_size": args.batch_size,
}

generator = SystemConfigGenerator(
    model_cls=SDMSystem,
    name=model_config.system_name,
    manual_configs=[config],
)
experiments = BeyondArenaExperimentBundle(
    models=[(generator, 0)],
    system_experiments=True,
    text_cache_mode="off",
).build_experiments()

context = BeyondArenaContext()
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    subset=args.subset or ["core"],
    register=False,
    build_kwargs=(
        {"dataset_names": [args.dataset]} if args.dataset is not None else None
    ),
)
