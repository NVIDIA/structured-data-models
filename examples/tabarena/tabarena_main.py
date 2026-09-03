# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM tabular model on TabArena."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator

from models import MODEL_CONFIGS, SDMSystem

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
args = parser.parse_args()

model_config = MODEL_CONFIGS[args.model]
result_dir = Path(__file__).parent.parent / "tabarena_out" / model_config.name
result_dir.mkdir(parents=True, exist_ok=True)

generator = SystemConfigGenerator(
    model_cls=SDMSystem,
    name=model_config.system_name,
    manual_configs=[{"model": args.model}],
)
experiments = TabArenaV0pt1ExperimentBundle(
    models=[(generator, 0)],
    system_experiments=True,
).build_experiments()

context = TabArenaContext()
context.build_and_run_jobs(
    experiments,
    expname=result_dir,
    register=False,
    build_kwargs=(
        {"dataset_names": [args.dataset]} if args.dataset is not None else None
    ),
)
