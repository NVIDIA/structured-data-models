"""kNN context selection for NemotronRelational (KumoRFM).

Per-cluster kNN context selection on relational data. Groups test rows
by feature similarity, selects kNN-informed context per cluster, runs
the relational sampler and model per cluster.

Compares:
  1. Full context  — random subsample of training rows (standard)
  2. Clustered kNN — per-cluster kNN-selected context rows
"""

# ruff: noqa
import argparse
from typing import cast

import numpy as np
import pandas as pd
import relbench
import torch
import torchmetrics
from sklearn.cluster import KMeans

import sdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-amazon")
parser.add_argument("--task", default="user-churn")
parser.add_argument("--k", type=int, default=100)
parser.add_argument("--num-clusters", type=int, default=20)
parser.add_argument("--num-neighbors", type=int, nargs="+", default=[16, 16])
parser.add_argument("--num-estimators", type=int, default=1)
args = parser.parse_args()

args.context_size = 10_000
args.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

# --- Load relational data (same as rel_bench.py) ----------------------------

task = relbench.tasks.get_task(args.dataset, args.task, download=True)
binary = task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION
multiclass = task.task_type == relbench.base.TaskType.MULTICLASS_CLASSIFICATION
classification = binary or multiclass

db = task.dataset.get_db(upto_test_timestamp=False)
tables = {}
for name, table in db.table_dict.items():
    tables[name] = sdm.TableTensor.from_pandas(
        df=table.df,
        stypes=sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                cast(str, table.pkey_col): "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text="drop",
            unsupported="drop",
        ),
    )

data = sdm.RelationalData(
    tables=tables,
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": cast(str, db.table_dict[right_table].pkey_col),
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)
time_columns = {
    name: table.time_col
    for name, table in db.table_dict.items()
    if table.time_col is not None
}
sampler = data.sampler(
    temporal=(
        sdm.TemporalSamplingConfig(
            time_columns=time_columns,
            strategy="last",
        )
        if time_columns
        else None
    ),
)

dfs = [
    task.get_table(split, mask_input_cols=False).df
    for split in ["train", "val", "test"]
]
task_df = pd.concat(dfs, ignore_index=True)
if binary:
    labels = sorted(
        pd.concat(dfs[:2], ignore_index=True)[task.target_col]
        .dropna()
        .unique()
    )
    task_df[task.target_col] = task_df[task.target_col] == labels[-1]
task_table = sdm.TableTensor.from_pandas(
    df=task_df,
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: "categorical" if classification else "numerical",
    },
)
context_all, query_all = task_table.split(
    [len(dfs[0]) + len(dfs[1]), len(dfs[2])]
)

sampler_kwargs = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

model = sdm.models.NemotronRelational(device=device)

n_context = len(context_all)
n_query = len(query_all)
print(f"Dataset: {args.dataset}/{args.task}")
print(
    f"Context: {n_context} rows, Query: {n_query} rows, k={args.k}, clusters={args.num_clusters}"
)
print()

# --- 1. Full context baseline (random subsample) ----------------------------

context_rand = context_all[torch.randperm(n_context)[: args.context_size]]
context_rand_sampled, related_rand = sampler(
    context_rand, **sampler_kwargs
).to(device)

with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
    model.fit(
        x=context_rand_sampled.drop_columns(task.target_col),
        y=context_rand_sampled[task.target_col],
        related_tables=related_rand,
        num_estimators=args.num_estimators,
    )

full_preds = []
full_targets = []
for batch in query_all.split(1000):
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
        out = model.predict(
            *sampler(batch.drop_columns(task.target_col), **sampler_kwargs).to(
                device
            )
        )
    if binary:
        pred, target = sdm.evaluation.to_binary_class(
            out, batch[task.target_col].to(device), positive_class=True
        )
    else:
        pred = out.numerical
        target = batch[task.target_col].to(device).numerical.squeeze(-1)
    full_preds.append(pred.cpu())
    full_targets.append(target.cpu())

full_pred = torch.cat(full_preds)
full_target = torch.cat(full_targets)
if binary:
    full_score = torchmetrics.classification.BinaryAUROC()(
        full_pred, full_target
    )
else:
    full_score = torchmetrics.regression.MeanAbsoluteError()(
        full_pred, full_target
    )
metric_name = "AUROC" if binary else "MAE"
print(
    f"Full context  ({len(context_rand)} rows): {metric_name}={full_score:.4f}"
)
model.clear()

# --- kNN index on task-table features ---------------------------------------

context_feat = context_all.drop_columns(task.target_col).numerical
query_feat = query_all.drop_columns(task.target_col).numerical

if context_feat.size(-1) == 0 or query_feat.size(-1) == 0:
    print(
        "No numerical features in task table for kNN — skipping kNN evaluation"
    )
    exit()

mean = context_feat.mean(dim=0)
std = context_feat.std(dim=0).clamp(min=1e-8)
context_norm = (context_feat - mean) / std
query_norm = (query_feat - mean) / std

knn_chunk = 5000
knn_indices_parts = []
context_norm_cpu = context_norm.cpu()
query_norm_cpu = query_norm.cpu()
for i in range(0, n_query, knn_chunk):
    chunk_dists = torch.cdist(
        query_norm_cpu[i : i + knn_chunk], context_norm_cpu
    )
    knn_indices_parts.append(
        chunk_dists.topk(args.k, dim=-1, largest=False).indices
    )
    print(f"  kNN chunk {min(i + knn_chunk, n_query)}/{n_query}", flush=True)
knn_indices = torch.cat(knn_indices_parts, dim=0)
del knn_indices_parts, context_norm_cpu, query_norm_cpu

# --- Cluster test rows ------------------------------------------------------

num_clusters = min(args.num_clusters, n_query)
kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed, n_init=10)
cluster_labels = kmeans.fit_predict(query_norm.cpu().numpy())

# --- 2. Clustered kNN context -----------------------------------------------

knn_preds = []
knn_targets = []

for c in range(num_clusters):
    mask = cluster_labels == c
    query_idx = torch.where(torch.tensor(mask))[0]
    if len(query_idx) == 0:
        continue

    ctx_indices = knn_indices[query_idx].unique()
    context_knn = context_all[ctx_indices]
    if len(context_knn) > args.context_size:
        context_knn = context_knn[
            torch.randperm(len(context_knn))[: args.context_size]
        ]

    context_sampled, related_knn = sampler(context_knn, **sampler_kwargs).to(
        device
    )

    gen = torch.Generator(device=device).manual_seed(args.seed)
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
        model.fit(
            x=context_sampled.drop_columns(task.target_col),
            y=context_sampled[task.target_col],
            related_tables=related_knn,
            num_estimators=args.num_estimators,
            generator=gen,
        )

    cluster_query = query_all[query_idx]
    for batch in cluster_query.split(1000):
        with torch.amp.autocast(device.type, torch.bfloat16, enabled=True):
            out = model.predict(
                *sampler(
                    batch.drop_columns(task.target_col), **sampler_kwargs
                ).to(device)
            )
        if binary:
            pred, target = sdm.evaluation.to_binary_class(
                out, batch[task.target_col].to(device), positive_class=True
            )
        else:
            pred = out.numerical
            target = batch[task.target_col].to(device).numerical.squeeze(-1)
        knn_preds.append(pred.cpu())
        knn_targets.append(target.cpu())

    model.clear()
    print(
        f"  Cluster {c + 1}/{num_clusters}: {len(query_idx)} queries, {len(ctx_indices)} ctx rows",
        flush=True,
    )

knn_pred = torch.cat(knn_preds)
knn_target = torch.cat(knn_targets)
if binary:
    knn_score = torchmetrics.classification.BinaryAUROC()(knn_pred, knn_target)
else:
    knn_score = torchmetrics.regression.MeanAbsoluteError()(
        knn_pred, knn_target
    )
print(
    f"Clustered kNN ({num_clusters} clusters, k={args.k}): {metric_name}={knn_score:.4f}"
)
