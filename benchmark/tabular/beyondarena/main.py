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
    "--max_cells",
    type=int,
    help="Limit context rows times selected columns per estimator.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Prediction batch size.",
)
parser.add_argument(
    "--many_class",
    action="store_true",
    help="Use ECOC when KumoTabular receives more than 10 classes.",
)
parser.add_argument(
    "--num_shards",
    type=int,
    default=1,
    help="Split the selected jobs into this many shards.",
)
parser.add_argument(
    "--shard_index",
    type=int,
    default=0,
    help="Run this zero-based shard.",
)
args = parser.parse_args()
if args.checkpoint is not None and args.name is None:
    parser.error("--name is required with --checkpoint")
if args.num_shards < 1:
    parser.error("--num_shards must be positive")
if not 0 <= args.shard_index < args.num_shards:
    parser.error("--shard_index must be smaller than --num_shards")

model_config = MODEL_CONFIGS[args.model]
run_name = args.name or model_config.name
result_dir = Path(__file__).parent.parent / "beyondarena_out" / run_name
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": args.model,
    "checkpoint": args.checkpoint,
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
    "max_cells": args.max_cells,
    "batch_size": args.batch_size,
    "many_class": args.many_class,
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
jobs = context.build_jobs(
    experiments,
    subset=args.subset or ["core"],
    **({"dataset_names": [args.dataset]} if args.dataset is not None else {}),
)
context.run_jobs(
    jobs[args.shard_index :: args.num_shards],
    expname=result_dir,
    register=False,
    raise_on_failure=False,
)
