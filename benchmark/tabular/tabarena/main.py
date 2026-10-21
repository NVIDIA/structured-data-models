# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM tabular model on TabArena."""

import argparse
import gc
import os
from pathlib import Path

import torch
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator

from benchmark.tabular.model import (
    MODEL_CONFIGS,
    add_finetune_args,
    finetune_config_overrides,
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
add_finetune_args(parser)
parser.add_argument(
    "--enable-kv-cache",
    action="store_true",
    help="Cache context key/value projections at fit and reuse them "
    "across prediction batches.",
)
args = parser.parse_args()

os.environ["SDM_ENABLE_CATEGORY_CHECKS"] = "0"

model_config = MODEL_CONFIGS[args.model]
result_dir = (
    Path(__file__).parent.parent
    / "tabarena_out"
    / model_config.name
    / "outer_model"
)

config = {
    "max_context_size": args.max_context_size,
    "max_columns": args.max_columns,
    "kv_cache": args.enable_kv_cache,
}
if args.batch_size is not None:
    config["ag.max_batch_size"] = args.batch_size
finetune_overrides = finetune_config_overrides(args)
if finetune_overrides and not args.model.endswith("-ft"):
    raise ValueError(
        f"--finetune_* flags were passed but --model {args.model!r} isn't "
        "a '-ft' variant, so fine-tuning is disabled and the flags would "
        "silently have no effect; pass the matching '-ft' model instead."
    )
config.update(finetune_overrides)

if finetune_overrides:
    # tabarena's cache key is positional, not content-based; key by
    # hyperparameters so different LRs don't silently collide.
    variant = "_".join(
        f"{key}={value}" for key, value in sorted(config.items())
    )
    result_dir = result_dir / variant
result_dir.mkdir(parents=True, exist_ok=True)

generator = ConfigGenerator(
    search_space={},
    model_cls=model_config.model_cls,
    manual_configs=[config],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    outer_experiments=True,
).build_experiments()
context = TabArenaContext()
jobs = context.build_jobs(
    experiments=experiments,
    subset=args.subset,
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
