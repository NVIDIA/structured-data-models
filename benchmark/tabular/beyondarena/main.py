# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM tabular model on BeyondArena."""

import argparse
import gc
from pathlib import Path

import torch
from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
from tabarena.contexts import BeyondArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.model import MODEL_CONFIGS

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

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    Path(__file__).parent.parent
    / "beyondarena_out"
    / model_config.name
    / "outer_model"
)
result_dir.mkdir(parents=True, exist_ok=True)

config = {
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
}
if args.batch_size is not None:
    config["ag.max_batch_size"] = args.batch_size

generator = ConfigGenerator(
    search_space={},
    model_cls=model_config.model_cls,
    manual_configs=[config],
)
experiments = BeyondArenaExperimentBundle(
    models=[(generator, 0)],
    outer_experiments=True,
).build_experiments()
context = BeyondArenaContext()
jobs = context.build_jobs(
    experiments=experiments,
    subset=args.subset or ["core"],
    dataset_names=[args.dataset] if args.dataset is not None else None,
)
for job in jobs:
    try:
        context.run_jobs(jobs=[job], expname=result_dir, register=False)
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch._C._host_emptyCache()
            torch.cuda.empty_cache()
