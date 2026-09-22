# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM tabular model results on TabArena."""

import argparse
import gzip
import pickle
from pathlib import Path

from tabarena.caching import CacheConfig
from tabarena.contexts import TabArenaContext
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
    "join runs with ',' to score them as one method, and append '=LABEL' to "
    "name that method.",
)
parser.add_argument(
    "--validation",
    choices=("outer", "official"),
    default="outer",
    help="Which runs to score: single fits on all rows, or the arena's "
    "bagged protocol.",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=benchmark_dir / "tabarena_out",
    help="Directory holding one result directory per run.",
)
parser.add_argument(
    "--cache_root",
    type=Path,
    help="Parent directory of every TabArena cache.",
)
parser.add_argument(
    "--subset",
    action="append",
    help="Score only these tasks; repeat to combine filters.",
)
parser.add_argument(
    "--backend",
    choices=("native", "ray"),
    default="native",
    help="Process the raw results in-process or in parallel with ray.",
)
args = parser.parse_args()

result_root = args.output_root
output_root = benchmark_dir / "evals"


def framework(result_dir: Path) -> str | None:
    """The method name of the run.

    The recorded framework, or its family for a bagged run, whose configs
    are named ``<family>_c1_..._BAG_L1``.
    """
    path = next(result_dir.rglob("results.pkl"), None)
    if path is None:
        return None
    with gzip.open(path, "rb") as f:
        name = pickle.load(f)["framework"]
    return name.split("_c1_")[0] if args.validation == "official" else name


if args.name:
    if args.model is None:
        parser.error("--name requires --model")
    model_config = MODEL_CONFIGS[args.model]
    runs = []
    for name in args.name:
        name, _, label = name.partition("=")
        runs.append(
            (
                label or name.replace(",", "+"),
                label
                or framework(
                    result_root
                    / name.split(",")[0]
                    / f"{args.validation}_model"
                ),
                [
                    result_root / part / f"{args.validation}_model"
                    for part in name.split(",")
                ],
            )
        )
else:
    runs = [
        (
            model_config.name,
            framework(
                result_root / model_config.name / f"{args.validation}_model"
            ),
            [result_root / model_config.name / f"{args.validation}_model"],
        )
        for model_config in MODEL_CONFIGS.values()
    ]
runs = [
    (label, method, result_dirs)
    for label, method, result_dirs in runs
    if all(next(d.rglob("results.pkl"), None) is not None for d in result_dirs)
]
if not runs:
    raise FileNotFoundError(f"No TabArena results found under {result_root}")

base_context = TabArenaContext(
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

    if args.validation == "official":
        # A bagged run is a config method, like the hosted models.
        method_metadata = MethodMetadata.config(
            method=method,
            display_name=label,
            compute="gpu",
            is_bag=True,
            can_hpo=False,
            artifact_dir=output_dir / "artifacts",
        )
    else:
        method_metadata = MethodMetadata.baseline(
            method=method,
            compute="gpu",
            artifact_dir=output_dir / "artifacts",
        )
    processed = EndToEnd.from_path_raw(
        path_raw=result_dirs,
        method_metadata=method_metadata,
        task_metadata=base_context.task_metadata_collection,
        backend=args.backend,
        name=method
        if method == label and args.validation == "outer"
        else None,
    )
    prefix = None if method == label else f"[{label}] "
    results = processed.get_results(new_result_prefix=prefix)
    results.to_csv(
        output_dir / "method_results_per_split.csv",
        index=False,
    )
    methods.extend(processed.to_method_metadata_lst(new_result_prefix=prefix))

context = TabArenaContext.from_new_methods(methods)
leaderboard = context.compare(
    output_dir=output_root,
    subset=args.subset,
    only_valid_tasks=methods,
)
website = context.leaderboard_to_website_format(leaderboard)
print(website.to_string(index=False))
