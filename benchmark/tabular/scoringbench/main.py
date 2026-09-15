# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import sys
from functools import partial
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).parent.parent
MODEL_NAMES = ("tabiclv2", "kumo-tabular")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoringbench-path",
        type=Path,
        required=True,
        help="Path to the pinned ScoringBench checkout.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="kumo-tabular",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dataset")
    selection.add_argument("--dataset-index", "--dataset_index", type=int)
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-size", "--sample_size", type=int)
    parser.add_argument("--n-repeats-cv", "--n_repeats_cv", type=int)
    parser.add_argument("--batch-size", "--batch_size", type=int)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=BENCHMARK_DIR / "scoringbench_out" / "univariate",
    )
    return parser


def _add_source_path(path: Path) -> None:
    path = path.resolve()
    if not (path / "scoringbench" / "univariate").is_dir():
        raise FileNotFoundError(f"ScoringBench checkout not found at {path}")
    sys.path.insert(0, str(path))


def _select_datasets(
    datasets: list[dict[str, Any]],
    *,
    name: str | None,
    index: int | None,
) -> list[dict[str, Any]]:
    if name is not None:
        selected = [dataset for dataset in datasets if dataset["name"] == name]
        if not selected:
            raise ValueError(f"Unknown ScoringBench dataset: {name!r}")
        return selected
    if index is None:
        return datasets
    if index < 0 or index >= len(datasets):
        raise IndexError(
            f"Dataset index {index} is outside [0, {len(datasets)})"
        )
    return [datasets[index]]


def main() -> None:
    args = _parser().parse_args()
    _add_source_path(args.scoringbench_path)

    config_module = importlib.import_module("scoringbench.univariate.config")
    datasets_module = importlib.import_module(
        "scoringbench.univariate.datasets"
    )
    runner_module = importlib.import_module("scoringbench.univariate.runner")
    models_module = importlib.import_module(
        "benchmark.tabular.scoringbench.models"
    )

    datasets = datasets_module.get_DATASETS_CONFIG()
    if args.dataset is not None:
        datasets = _select_datasets(datasets, name=args.dataset, index=None)
    datasets = datasets_module.validate_datasets(datasets)
    datasets = _select_datasets(
        datasets,
        name=None,
        index=args.dataset_index,
    )

    wrapper = models_module.WRAPPERS[args.model]
    model_config = models_module.MODEL_CONFIGS[args.model]
    seed = config_module.SEED if args.seed is None else args.seed
    sample_size = (
        config_module.SAMPLE_SIZE
        if args.sample_size is None
        else args.sample_size
    )
    n_repeats_cv = (
        config_module.N_REPEATS_CV
        if args.n_repeats_cv is None
        else args.n_repeats_cv
    )
    n_folds = 2 if args.lite else config_module.N_FOLDS
    factory = partial(
        wrapper,
        seed=seed,
        batch_size=args.batch_size,
    )
    runner_module.run_benchmark(
        datasets_config=datasets,
        model_factories={model_config.method: factory},
        output_dir=args.output_dir,
        n_folds=n_folds,
        n_repeats_cv=n_repeats_cv,
        seed=seed,
        sample_size=sample_size,
    )


if __name__ == "__main__":
    main()
