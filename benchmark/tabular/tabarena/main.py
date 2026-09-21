# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM tabular model on TabArena."""

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.caching import CacheConfig
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
    "--num_estimators",
    type=int,
    help="Ensemble members per fit (default: the model's).",
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
    help="Local checkpoint loaded instead of the published weights.",
)
parser.add_argument(
    "--size",
    choices=("small", "large"),
    help="kumo-tabular architecture matching --checkpoint (default: large).",
)
parser.add_argument(
    "--numerical_missing",
    choices=("dispatch", "nan", "mix", "impute"),
    help="How kumo-tabular passes missing numerical cells to the model "
    "(default: nan).",
)
parser.add_argument(
    "--regression_reduction",
    choices=("scalar_trim", "quantile_trim"),
    help="Whether kumo-tabular trims the members' point predictions or "
    "every quantile before averaging (default: scalar_trim).",
)
parser.add_argument(
    "--name",
    help="Name of the result directory (default: the model name).",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path(__file__).parent.parent / "tabarena_out",
    help="Directory holding one result directory per run.",
)
parser.add_argument(
    "--cache_root",
    type=Path,
    help="Parent directory of every TabArena cache.",
)
args = parser.parse_args()
if args.model != "kumo-tabular" and (
    args.checkpoint is not None
    or args.size is not None
    or args.numerical_missing is not None
    or args.regression_reduction is not None
):
    parser.error(
        "--checkpoint, --size, --numerical_missing, --regression_reduction "
        "need --model kumo-tabular"
    )

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    args.output_root / (args.name or model_config.name) / "outer_model"
)
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
}
if args.num_estimators is not None:
    config["num_estimators"] = args.num_estimators
if args.checkpoint is not None:
    config["checkpoint"] = args.checkpoint
if args.size is not None:
    config["size"] = args.size
if args.numerical_missing is not None:
    config["numerical_missing"] = args.numerical_missing
if args.regression_reduction is not None:
    config["regression_reduction"] = args.regression_reduction
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

context = TabArenaContext(
    cache_config=(
        CacheConfig.from_root(args.cache_root)
        if args.cache_root is not None
        else None
    ),
)
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    subset=args.subset,
    register=False,
    build_kwargs=(
        {"dataset_names": [args.dataset]} if args.dataset is not None else None
    ),
)
