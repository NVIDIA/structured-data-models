# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM tabular model on TabArena."""

import argparse
import json
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.caching import CacheConfig
from tabarena.contexts import TabArenaContext
from tabarena.models.utils import get_model_info_from_name
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.model import MODEL_CONFIGS, SDMModelWrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    default="tabiclv2",
    help="SDM model to benchmark.",
)
parser.add_argument(
    "--no_kv_cache",
    action="store_true",
    help="kumo-tabular: run the context inside predict instead of caching "
    "it at fit, like the other in-context wrappers.",
)
parser.add_argument(
    "--registry_hp",
    action="append",
    metavar="KEY=JSON",
    help="Hyperparameter of --registry_model (repeat); e.g. n_estimators=1.",
)
parser.add_argument(
    "--registry_model",
    help="Benchmark this tabarena registry model (e.g. TabPFN-3.5) with its "
    "upstream wrapper instead of an SDM model, under the same settings.",
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
    "--split_indices",
    action="append",
    help="Run only these splits (r<repeat>f<fold>, e.g. r2f1); repeat the flag for several.",
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
    choices=("small", "large", "xlarge"),
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
    "--validation",
    choices=("outer", "official"),
    default="outer",
    help="'outer' fits once on all training rows; 'official' runs the "
    "arena's bagged protocol (eight fold fits and a refit), as the hosted "
    "methods do.",
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
if args.registry_model is not None and args.num_estimators is not None:
    parser.error("--num_estimators applies to SDM models only")
if args.registry_hp and args.registry_model is None:
    parser.error("--registry_hp needs --registry_model")
if args.model != "kumo-tabular" and (
    args.checkpoint is not None
    or args.size is not None
    or args.numerical_missing is not None
    or args.regression_reduction is not None
    or args.no_kv_cache
):
    parser.error(
        "--checkpoint, --size, --numerical_missing, --regression_reduction, "
        "--no_kv_cache need --model kumo-tabular"
    )

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    args.output_root
    / (args.name or args.registry_model or model_config.name)
    / f"{args.validation}_model"
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
if args.no_kv_cache:
    config["kv_cache"] = False

if args.registry_hp:
    # The registry model with pinned hyperparameters, as one manual config.
    info = get_model_info_from_name(args.registry_model)
    hyperparameters = {
        key: json.loads(value)
        for key, _, value in (hp.partition("=") for hp in args.registry_hp)
    }
    models = [
        (
            ConfigGenerator(
                search_space={},
                model_cls=info.model_cls,
                manual_configs=[hyperparameters],
            ),
            0,
        )
    ]
elif args.registry_model is not None:
    models = [(args.registry_model, 0)]
else:
    models = [
        (
            ConfigGenerator(
                search_space={},
                model_cls=model_config.model_cls,
                manual_configs=[config],
            ),
            0,
        )
    ]
# One GPU per fit, as the hosted methods were run.
experiments = TabArenaV0pt1ExperimentBundle(
    models=models,
    outer_experiments=args.validation == "outer",
).build_experiments(num_gpus=1)
for experiment in experiments:
    experiment.experiment_kwargs["cleanup_on_failure"] = True
    if args.validation == "outer" and args.registry_model is None:
        experiment.method_cls = SDMModelWrapper

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
    build_kwargs={
        **({"dataset_names": [args.dataset]} if args.dataset is not None else {}),
        **({"split_indices": args.split_indices} if args.split_indices else {}),
    }
    or None,
)
