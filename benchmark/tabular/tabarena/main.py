r"""Run an SDM tabular model on TabArena."""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.model import (
    MODEL_CONFIGS,
    SDMExperimentRunner,
    SDMModelWrapper,
)

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
    "--max_columns",
    type=int,
    help="Select at most this many columns per estimator.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Prediction batch size.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Local kumo-scm checkpoint for --model kumo-tabular.",
)
parser.add_argument(
    "--numerical_missing",
    choices=("nan", "impute", "mix"),
    default="nan",
    help="KumoTabular NaN handling: keep, mean-impute, or alternate both.",
)
parser.add_argument(
    "--recipe_ensemble",
    help="Comma-separated KumoTabular recipe names weighted by context "
    "cross-validation, e.g. mix,mix+catshuffle30,mix+quantile,mix+inter.",
)
parser.add_argument(
    "--name",
    help="Result directory name (default: the model name).",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path(__file__).parent.parent / "tabarena_out",
    help="Directory holding one result directory per run.",
)
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    args.output_root / (args.name or model_config.name) / "outer_model"
)
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
    "checkpoint": args.checkpoint,
    "numerical_missing": args.numerical_missing,
    "recipe_ensemble": (
        args.recipe_ensemble.split(",") if args.recipe_ensemble else None
    ),
}
if args.batch_size is not None:
    config["ag.max_batch_size"] = args.batch_size

generator = ConfigGenerator(
    search_space={},
    model_cls=model_config.model_cls,
    manual_configs=[config],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    outer_experiments=True,
).build_experiments()
for experiment in experiments:
    experiment.method_cls = SDMModelWrapper
    experiment.experiment_cls = SDMExperimentRunner

context = TabArenaContext()
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    subset=args.subset,
    register=False,
    build_kwargs=(
        {"dataset_names": [args.dataset]} if args.dataset is not None else None
    ),
)
