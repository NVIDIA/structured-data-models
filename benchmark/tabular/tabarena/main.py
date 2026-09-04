r"""Run an SDM tabular model on TabArena."""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import (
    ModelConstraints,
    TabArenaV0pt1ExperimentBundle,
)
from tabarena.contexts import TabArenaContext
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
    help="Run only the selected TabArena dataset.",
)
parser.add_argument(
    "--subset",
    action="append",
    help="Filter tasks; repeat to combine filters.",
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
parser.add_argument(
    "--checkpoint-path",
    help="Local Kumo-SCM checkpoint.",
)
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]
if model_config.checkpoint_task is not None and args.checkpoint_path is None:
    parser.error(f"--model {args.model} requires --checkpoint-path")
result_dir = Path(__file__).parent.parent / "tabarena_out" / model_config.name
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": args.model,
    "max_context_size": args.max_context_size,
    "batch_size": args.batch_size,
}
if model_config.checkpoint_task is not None:
    config["checkpoint_path"] = args.checkpoint_path

generator = SystemConfigGenerator(
    model_cls=SDMSystem,
    name=model_config.system_name,
    manual_configs=[config],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    system_experiments=True,
    custom_model_constraints=(
        {
            "SDMSystem": ModelConstraints(
                max_n_features=1024,
                max_n_classes=10,
            )
        }
        if model_config.checkpoint_task is not None
        else {}
    ),
).build_experiments()

context = TabArenaContext()
build_kwargs: dict[str, object] = {}
if model_config.checkpoint_task is not None:
    build_kwargs["problem_types"] = [model_config.checkpoint_task]
if args.dataset is not None:
    build_kwargs["dataset_names"] = [args.dataset]
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    subset=args.subset,
    register=False,
    build_kwargs=build_kwargs,
)
