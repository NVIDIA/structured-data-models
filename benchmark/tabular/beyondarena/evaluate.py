# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM tabular model results on BeyondArena."""

import argparse
from pathlib import Path

from tabarena.caching import CacheConfig
from tabarena.contexts import BeyondArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata

from benchmark.tabular.model import MODEL_CONFIGS

benchmark_dir = Path(__file__).parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    help="Model of the named runs (default: every model's default run).",
)
parser.add_argument(
    "--name",
    action="append",
    help="Named run of ``--model`` to evaluate; repeat for one row per run, "
    "or join runs with ',' to score them as one method.",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=benchmark_dir / "beyondarena_out",
    help="Directory holding one result directory per run.",
)
parser.add_argument(
    "--cache_root",
    type=Path,
    help="Parent directory of every BeyondArena cache.",
)
args = parser.parse_args()

result_root = args.output_root
output_root = benchmark_dir / "beyondarena_evals"

if args.name:
    if args.model is None:
        parser.error("--name requires --model")
    model_config = MODEL_CONFIGS[args.model]
    runs = [
        (
            name.replace(",", "+"),
            model_config.beyondarena_method_name,
            [result_root / part / "outer_model" for part in name.split(",")],
        )
        for name in args.name
    ]
else:
    runs = [
        (
            model_config.name,
            model_config.beyondarena_method_name,
            [result_root / model_config.name / "outer_model"],
        )
        for model_config in MODEL_CONFIGS.values()
    ]
runs = [
    (label, method, result_dirs)
    for label, method, result_dirs in runs
    if all(next(d.rglob("results.pkl"), None) is not None for d in result_dirs)
]
if not runs:
    raise FileNotFoundError(
        f"No BeyondArena results found under {result_root}"
    )

base_context = BeyondArenaContext(
    cache_config=(
        CacheConfig.from_root(args.cache_root)
        if args.cache_root is not None
        else None
    ),
)
methods = []
for label, method, result_dirs in runs:
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    method_metadata = MethodMetadata.baseline(
        method=method,
        compute="gpu",
        artifact_dir=output_dir / "artifacts",
    )
    processed = EndToEnd.from_path_raw(
        path_raw=result_dirs,
        method_metadata=method_metadata,
        task_metadata=base_context.task_metadata_collection,
        backend="native",
    )
    prefix = f"[{label}] "
    results = processed.get_results(new_result_prefix=prefix)
    results.to_csv(
        output_dir / "method_results_per_split.csv",
        index=False,
    )
    methods.extend(processed.to_method_metadata_lst(new_result_prefix=prefix))

context = BeyondArenaContext.from_new_methods(methods)
leaderboard = context.compare(
    output_dir=output_root,
    only_valid_tasks=[method.method for method in methods],
)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
