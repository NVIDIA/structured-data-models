# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit coverage and report native-normalized timings for measured methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.utils.utils import get_pred_from_proba
from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
from tabarena.models import MethodMetadata
from tabarena.utils.pickle_utils import load_pickle

MODELS = ("kumo", "kumo-batched", "causilo", "tabpfn-fast")
LABELS = {
    "kumo": "KumoTabular-Small sequential no-refit",
    "kumo-batched": "KumoTabular-Small batched auto no-refit",
    "causilo": "Causilo no-refit",
    "tabpfn-fast": "TabPFN-3.5-Fast no-refit",
}


def compare_predictions(
    records: dict, paths: dict, keys: set, output: Path
) -> None:
    """Compare preserved native test and OOF arrays."""
    rows = []
    for dataset, repeat, fold in sorted(keys):
        key = (dataset, repeat, fold)
        base = load_pickle(paths[("kumo", *key)])
        candidate = load_pickle(paths[("kumo-batched", *key)])
        assert base["task_metadata"] == candidate["task_metadata"]
        assert base["problem_type"] == candidate["problem_type"]
        a_sim, b_sim = (
            base["simulation_artifacts"],
            candidate["simulation_artifacts"],
        )
        for field in (
            "y_test_idx",
            "y_val_idx",
            "y_test",
            "y_val",
            "ordered_class_labels",
        ):
            assert (field in a_sim) == (field in b_sim), field
            if field in a_sim:
                assert np.array_equal(
                    np.asarray(a_sim[field]), np.asarray(b_sim[field])
                ), field
        row = {"dataset": dataset, "repeat": repeat, "fold": fold}
        for split, label in (("test", "test"), ("val", "oof")):
            a_dict = a_sim[f"pred_proba_dict_{split}"]
            b_dict = b_sim[f"pred_proba_dict_{split}"]
            assert len(a_dict) == 1
            assert len(b_dict) == 1
            a = np.asarray(a_dict[base["framework"]])
            b = np.asarray(b_dict[candidate["framework"]])
            assert a.shape == b.shape
            assert np.isfinite(a).all()
            assert np.isfinite(b).all()
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            row.update(
                {
                    f"{label}_shape": list(a.shape),
                    f"{label}_baseline_dtype": str(a.dtype),
                    f"{label}_candidate_dtype": str(b.dtype),
                    f"{label}_byte_equal": a.dtype == b.dtype
                    and a.tobytes() == b.tobytes(),
                    f"{label}_max_abs": float(delta.max(initial=0)),
                    f"{label}_mean_abs": float(delta.mean()),
                    f"{label}_changed_values": int(np.count_nonzero(a != b)),
                }
            )
            if base["problem_type"] != "regression":
                a_labels = get_pred_from_proba(
                    a, problem_type=base["problem_type"]
                )
                b_labels = get_pred_from_proba(
                    b, problem_type=base["problem_type"]
                )
                row[f"{label}_changed_labels"] = int(
                    np.count_nonzero(a_labels != b_labels)
                )
        for field in ("metric_error", "metric_error_val"):
            row[f"{field}_candidate_minus_baseline"] = (
                records[("kumo-batched", *key)]["metrics"][field]
                - records[("kumo", *key)]["metrics"][field]
            )
        rows.append(row)
    (output / "paired_predictions.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--models",
        choices=MODELS,
        nargs="+",
        default=["kumo-batched", "causilo", "tabpfn-fast"],
    )
    parser.add_argument("--compare-predictions", action="store_true")
    parser.add_argument(
        "--baseline-source-commit",
        help="Original baseline commit required for prediction comparison",
    )
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Only common completed splits; never full-suite scoring",
    )
    parser.add_argument(
        "--official-plots",
        action="store_true",
        help="Full coverage only; native scoring and Pareto plots",
    )
    args = parser.parse_args()
    models = tuple(args.models)
    if len(set(models)) != len(models):
        raise ValueError("Select distinct model identities")

    context = TabArenaContext(
        methods=[],
        backend="native",
        fillna_method=None,
        calibration_method=None,
    )
    expected = {
        (r["dataset"], r["repeat"], r["fold"])
        for r in json.loads(args.grid.read_text())
    }
    native_grid = {
        (task.tabarena_task_name, split.repeat, split.fold)
        for task in context.task_metadata_collection
        for split in task.splits_metadata.values()
    }
    if expected != native_grid:
        raise ValueError(
            "Supplied grid differs from pinned native task metadata"
        )
    if len(expected) != 816 or len({key[0] for key in expected}) != 51:
        raise ValueError(
            "Expected the pinned official 51-dataset / 816-split grid"
        )
    records = {}
    result_paths = {}
    protocols = set()
    raw_files = {model: [] for model in models}
    identities = {model: set() for model in models}
    for path in sorted(
        path
        for root in args.run_root
        for path in root.rglob("benchmark-result.json")
    ):
        record = json.loads(path.read_text())
        model = record["model"]
        if model not in models or record["status"] != "completed":
            continue
        key = (record["dataset"], record["repeat"], record["fold"])
        identity = (model, *key)
        if identity in records:
            raise ValueError(
                f"Duplicate successful attempt: {identity}; "
                "select one before evaluation"
            )
        if key not in expected:
            raise ValueError(f"Unexpected split: {identity}")
        protocol = record["protocol"]
        protocols.add(json.dumps(protocol, sort_keys=True))
        if protocol["outer_experiments"] or protocol["refit_folds"]:
            raise ValueError(f"Wrong requested protocol: {identity}")
        validation = record["validation_protocol"]
        if validation["key"] != "8x1" or validation["num_child_models"] != 8:
            raise ValueError(f"Wrong fitted ensemble: {identity}")
        raw = path.parent / record["result_file"]
        if (
            hashlib.sha256(raw.read_bytes()).hexdigest()
            != record["result_sha256"]
        ):
            raise ValueError(f"Raw result changed: {raw}")
        settings = path.parent / record["fitted_settings_file"]
        if (
            hashlib.sha256(settings.read_bytes()).hexdigest()
            != record["fitted_settings_sha256"]
        ):
            raise ValueError(f"Fitted settings changed: {settings}")
        raw_result = load_pickle(raw)
        for name in (
            "time_train_s",
            "time_infer_s",
            "metric_error",
            "metric_error_val",
        ):
            value = float(record["metrics"][name])
            if not math.isfinite(value) or (
                name.startswith("time_") and value < 0
            ):
                raise ValueError(f"Invalid result value {name}: {identity}")
            raw_value = raw_result[name]
            if type(raw_value)(value) != raw_value:
                raise ValueError(
                    f"Summary differs from raw numeric value: "
                    f"{identity} {name}"
                )
            record["metrics"][name] = float(raw_value)
        identities[model].add(json.dumps(record["identity"], sort_keys=True))
        records[identity] = record
        result_paths[identity] = raw
        raw_files[model].append(raw)
    if len(protocols) != 1:
        raise ValueError(
            "Selected methods must share resource/protocol settings"
        )
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError(
            "Exactly one source/harness freeze is required per model identity"
        )
    model_identities = {
        model: json.loads(next(iter(values)))
        for model, values in identities.items()
    }
    if (
        len(
            {
                (
                    value["tabarena"]["commit"],
                    value["tabarena"]["manifest_sha256"],
                )
                for value in model_identities.values()
            }
        )
        != 1
    ):
        raise ValueError("Models must use the same pinned TabArena source")
    available = {
        model: {key[1:] for key in records if key[0] == model}
        for model in models
    }
    common = set.intersection(*(available[model] for model in models))
    complete = all(available[model] == expected for model in models)
    args.output.mkdir(parents=True, exist_ok=True)
    coverage = {
        "complete": complete,
        "expected_splits_per_method": 816,
        "expected_datasets": 51,
        "completed_splits": {model: len(available[model]) for model in models},
        "missing_splits": {
            model: sorted(expected - available[model]) for model in models
        },
        "common_splits": len(common),
        "common_datasets": len({key[0] for key in common}),
        "scope": "full measured cohort"
        if complete
        else "partial common-split diagnostic; not full-suite rating",
        "identities_by_model": model_identities,
        "grid_sha256": hashlib.sha256(args.grid.read_bytes()).hexdigest(),
    }
    (args.output / "coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n"
    )
    if not complete and not args.partial:
        raise ValueError(
            "Incomplete grid; coverage saved. Use --partial "
            "for a labeled common-split diagnostic"
        )
    if not common:
        raise ValueError("No split completed by all selected methods")
    if args.official_plots and not complete:
        raise ValueError(
            "Full-suite scoring/plots require every measured split; "
            "no imputation"
        )
    denominators = {}
    for task in context.task_metadata_collection:
        splits = list(task.splits_metadata.values())
        denominators[task.tabarena_task_name] = (
            sum(s.num_instances_train for s in splits) / len(splits),
            sum(s.num_instances_test for s in splits) / len(splits),
        )
    rows = []
    for model in models:
        for dataset, repeat, fold in sorted(common):
            record = records[(model, dataset, repeat, fold)]
            train_rows, test_rows = denominators[dataset]
            metrics = record["metrics"]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "repeat": repeat,
                    "fold": fold,
                    "mean_outer_train_rows": train_rows,
                    "mean_outer_test_rows": test_rows,
                    "time_train_s": metrics["time_train_s"],
                    "time_infer_s": metrics["time_infer_s"],
                    "time_train_s_per_1K": metrics["time_train_s"]
                    * 1000
                    / train_rows,
                    "time_infer_s_per_1K": metrics["time_infer_s"]
                    * 1000
                    / test_rows,
                    "metric_error": metrics["metric_error"],
                    "metric_error_val": metrics["metric_error_val"],
                    "metric": metrics["metric"],
                }
            )
    split_frame = pd.DataFrame(rows)
    if (split_frame.groupby("dataset")["metric"].nunique() != 1).any():
        raise ValueError("Metrics differ within a dataset")
    split_frame.to_csv(
        args.output / "common_split_normalized.csv", index=False
    )
    measures = [
        "time_train_s",
        "time_infer_s",
        "time_train_s_per_1K",
        "time_infer_s_per_1K",
        "metric_error",
        "metric_error_val",
    ]
    dataset_frame = split_frame.groupby(["model", "dataset"], as_index=False)[
        measures
    ].mean()
    dataset_frame.to_csv(args.output / "common_dataset_means.csv", index=False)
    aggregate = dataset_frame.groupby("model")[
        ["time_train_s_per_1K", "time_infer_s_per_1K"]
    ].median()
    aggregate.columns = ["median_train_s_per_1K", "median_infer_s_per_1K"]
    aggregate.to_csv(args.output / "timing_medians.csv")
    comparisons = {}
    for field, split in (
        ("metric_error", "test"),
        ("metric_error_val", "oof"),
    ):
        quality = dataset_frame.pivot(
            index="dataset", columns="model", values=field
        )
        comparisons[split] = {}
        for other in models[1:]:
            delta = quality[models[0]] - quality[other]
            comparisons[split][other] = {
                "datasets": len(delta),
                "first_model": models[0],
                "first_model_wins": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "first_model_losses": int((delta > 0).sum()),
                "scope": "Lower dataset-mean native error wins; no public Elo",
            }
    (args.output / "pairwise_quality.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    if args.compare_predictions:
        if not {"kumo", "kumo-batched"}.issubset(models):
            raise ValueError(
                "Prediction comparison needs kumo and kumo-batched"
            )
        if not args.baseline_source_commit:
            raise ValueError("Pass the original baseline source commit")
        if (
            model_identities["kumo"]["sdm"].get("commit")
            != args.baseline_source_commit
        ):
            raise ValueError(
                "The supplied baseline is not the requested source"
            )
        if model_identities["kumo"]["sdm"].get("dirty_tracked_source"):
            raise ValueError("Original baseline tracked source must be clean")
        compare_predictions(records, result_paths, common, args.output)
    if args.official_plots:
        processed_methods = []
        frames = []
        for model in models:
            metadata = MethodMetadata.config(
                method=LABELS[model],
                display_name=LABELS[model],
                compute="gpu",
                is_bag=True,
                can_hpo=False,
                artifact_dir=args.output / "artifacts" / model,
            )
            processed = EndToEnd.from_raw(
                results_lst=[load_pickle(path) for path in raw_files[model]],
                method_metadata=metadata,
                task_metadata=context.task_metadata_collection,
                backend="native",
            )
            result_frame = processed.get_results(
                new_result_prefix=f"[{model}] "
            )
            result_methods = processed.to_method_metadata_lst(
                new_result_prefix=f"[{model}] "
            )
            if len(result_frame) != 816 or len(result_methods) != 1:
                raise ValueError(
                    f"Expected one default and 816 processed rows for {model}"
                )
            frames.append(result_frame)
            processed_methods.extend(result_methods)
        frame = pd.concat(frames, ignore_index=True)
        if len(processed_methods) != len(models) or frame[
            "method"
        ].nunique() != len(models):
            raise ValueError(
                "Native scoring must contain exactly "
                "the selected measured defaults"
            )
        frame.to_csv(args.output / "native_scoring_input.csv", index=False)
        pool = TabArenaContext(
            methods=processed_methods,
            backend="native",
            fillna_method=None,
            calibration_method=None,
        )
        leaderboard = pool.compare(
            output_dir=args.output / "native-comparison",
            ta_results=frame,
            fillna=None,
            calibration_method=None,
            plot=True,
        )
        pool.leaderboard_to_website_format(leaderboard).to_csv(
            args.output / "measured_leaderboard.csv", index=False
        )
        (args.output / "scoring_scope.json").write_text(
            json.dumps(
                {
                    "pool": [LABELS[model] for model in models],
                    "published_reference_methods_in_pool": False,
                    "imputation": False,
                    "calibration_method": None,
                    "rating_scope": (
                        "Selected measured methods only; "
                        "not comparable to public leaderboard Elo"
                    ),
                    "protocol": (
                        "8x1; outer_experiments=False; refit_folds=False"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "coverage": coverage,
                "timings": aggregate.reset_index().to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
