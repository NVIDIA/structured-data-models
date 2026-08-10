# ruff: noqa: T201, BLE001
"""Profile the full TabICLv2 + SentenceTransformer pipeline across STRABLE.

Profiles fit + predict for 20 STRABLE datasets selected for variety in
row count, text column count, and text length. Runs each dataset twice:
once without text processing (baseline) and once with SentenceTransformer.
Writes per-dataset timing and dataset characteristics to a CSV.

Usage:
    python profile_sentence_transformer.py
    python profile_sentence_transformer.py --output results.csv
"""

import argparse
import csv
import json
import time
import traceback

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp

DATASETS = [
    "tobacco-problem",
    "hospitals",
    "schools",
    "drug-shortages",
    "osha-accidents",
    "financial-product-complaint",
    "wine-dataset",
    "drug-enforcement",
    "museums",
    "chocolate-bar-ratings",
    "clear-corpus",
    "mercari",
    "covid-clinical-trials",
    "yelp_business",
    "kickstarter-projects",
    "sf-building-permits",
    "lending-club-loan",
    "food-enforcement",
    "medicines",
    "michelin-ratings",
]

TASK_OVERRIDES = {
    "tobacco-problem": "classification",
}

CSV_FIELDS = [
    "dataset",
    "num_rows",
    "num_text_cols",
    "avg_text_length",
    "max_text_length",
    "task",
    "mode",
    "fit_time_s",
    "predict_time_s",
    "total_time_s",
    "metric_name",
    "metric_value",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="profile_results.csv",
    help="Output CSV path (default: profile_results.csv).",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for dataset_name in DATASETS:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    try:
        config_path = hf_hub_download(
            repo_id="inria-soda/STRABLE-benchmark",
            filename=f"{dataset_name}/config.json",
            repo_type="dataset",
        )
        with open(config_path) as f:
            config = json.load(f)
        target_name = config["target_name"]
        task = TASK_OVERRIDES.get(
            dataset_name,
            "classification"
            if "classification" in config.get("task", "")
            else "regression",
        )

        data_path = hf_hub_download(
            repo_id="inria-soda/STRABLE-benchmark",
            filename=f"{dataset_name}/data.parquet",
            repo_type="dataset",
        )
        arrow_table = pq.read_table(data_path)
    except Exception as e:
        print(f"  Skipping (download error): {e}")
        continue

    stypes = sdm.infer_stypes(arrow_table, with_text=True)
    text_cols = [
        col for col, stype in stypes.items() if stype == sdm.Stype.text
    ]

    if not text_cols:
        print("  Skipping (no text columns)")
        continue

    text_lengths = []
    for col in text_cols:
        arr = arrow_table.column(col).combine_chunks()
        text_lengths.append(pc.drop_null(pc.utf8_length(arr)))
    all_lengths = pa.concat_arrays(text_lengths)
    avg_len = pc.mean(all_lengths).as_py() or 0
    max_len = pc.max(all_lengths).as_py() or 0

    base_row = {
        "dataset": dataset_name,
        "num_rows": len(arrow_table),
        "num_text_cols": len(text_cols),
        "avg_text_length": round(avg_len),
        "max_text_length": max_len,
        "task": task,
    }

    print(f"  Rows: {base_row['num_rows']}")
    print(f"  Text columns: {base_row['num_text_cols']} {text_cols}")
    print(f"  Avg text length: {base_row['avg_text_length']} chars")

    stypes_no_text = sdm.infer_stypes(arrow_table, with_text=False)

    for mode in ("none", "embed"):
        row = {**base_row, "mode": mode}
        print(f"\n  --- mode: {mode} ---")

        try:
            use_text = mode == "embed"
            table = sdm.TableTensor.from_arrow(
                table=arrow_table,
                stypes=stypes if use_text else stypes_no_text,
                device=device,
            )

            generator = torch.Generator(device=device).manual_seed(42)
            num_rows = len(table)
            perm = torch.randperm(num_rows, generator=generator, device=device)
            context_size = int(0.8 * num_rows)
            context = table[perm[:context_size]]
            query = table[perm[context_size:]]
            if task == "regression":
                ground_truth = query[:, target_name].numerical.squeeze(-1)
            else:
                ground_truth = query[:, target_name].categorical.code.squeeze(
                    -1
                )

            model = sdm.models.TabICLv2(device=device)
            recipe = model.default_recipe()
            if use_text:
                text_processor = sp.Sequential(
                    sp.SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2",
                    ),
                    sp.PCA(num_components=64),
                )
                recipe.prepend_features(sp.StypeDispatch(text=text_processor))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.amp.autocast(
                device.type, torch.float16, enabled=table.is_cuda
            ):
                model.fit(
                    x=context.drop_columns(target_name),
                    y=context[:, target_name],
                    recipe=recipe,
                    generator=generator,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fit = time.perf_counter() - t0

            t0 = time.perf_counter()
            with torch.amp.autocast(
                device.type, torch.float16, enabled=table.is_cuda
            ):
                prediction = model.predict(
                    query.drop_columns(target_name),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_predict = time.perf_counter() - t0

            if task == "regression":
                prediction = prediction.numerical.mean(dim=-1)
                metric_name = "rmse"
                metric_value = (
                    (prediction - ground_truth).pow(2).mean().sqrt().item()
                )
            else:
                prediction = prediction.categorical.code.squeeze(-1)
                metric_name = "accuracy"
                metric_value = (
                    (prediction == ground_truth).float().mean().item()
                )

            row.update(
                {
                    "fit_time_s": round(t_fit, 3),
                    "predict_time_s": round(t_predict, 3),
                    "total_time_s": round(t_fit + t_predict, 3),
                    "metric_name": metric_name,
                    "metric_value": round(metric_value, 4),
                }
            )

            print(f"    Fit:     {row['fit_time_s']:.3f}s")
            print(f"    Predict: {row['predict_time_s']:.3f}s")
            print(f"    Total:   {row['total_time_s']:.3f}s")
            print(f"    {metric_name}: {metric_value:.4f}")

        except Exception:
            traceback.print_exc()
            row.update(
                {
                    "fit_time_s": None,
                    "predict_time_s": None,
                    "total_time_s": None,
                    "metric_name": None,
                    "metric_value": None,
                }
            )

        results.append(row)

with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to {args.output}")
