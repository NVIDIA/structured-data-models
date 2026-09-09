r"""Run an SDM tabular model on BeyondArena."""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import (
    BeyondArenaExperimentBundle,
    ModelConstraints,
)
from tabarena.contexts import BeyondArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.system import MODEL_CONFIGS, SDMModel

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
    Path(__file__).parent.parent / "beyondarena_model_out" / model_config.name
)
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "model": args.model,
    "max_context_size": args.max_context_size,
}

generator = ConfigGenerator(
    search_space={},
    model_cls=SDMModel,
    manual_configs=[config],
)
experiments = BeyondArenaExperimentBundle(
    models=[(generator, 0)],
    outer_experiments=True,
    max_predict_batch_size=args.batch_size,
    custom_model_constraints={
        SDMModel.ag_key: ModelConstraints(
            max_n_classes=model_config.max_classes,
        )
    },
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
