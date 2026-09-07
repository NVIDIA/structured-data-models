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
    "--checkpoint",
    help="Load a local KumoTFM trainer checkpoint.",
)
parser.add_argument(
    "--name",
    help="Use this result name for a local checkpoint.",
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
    "--max_columns",
    type=int,
    help="Select at most this many columns per estimator.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Prediction batch size.",
)
args = parser.parse_args()
if args.checkpoint is not None and args.name is None:
    parser.error("--name is required with --checkpoint")

model_config = MODEL_CONFIGS[args.model]
run_name = args.name or model_config.name
result_dir = Path(__file__).parent.parent / "beyondarena_out" / run_name
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": args.model,
    "checkpoint": args.checkpoint,
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
    "batch_size": args.batch_size,
}

generator = SystemConfigGenerator(
    model_cls=SDMSystem,
    name=f"SDM{''.join(char for char in run_name if char.isalnum())}System",
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
