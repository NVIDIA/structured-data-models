"""Diagnostic for clustered kNN context selection.

Prints a per-cluster table to pinpoint whether prediction failures come from
column orientation flips, bad predictions, context size, or class imbalance.
"""

# ruff: noqa
import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

import sdm
from sdm.processing.execution import RecipeExecution

from recipe import knn_recipe

DATASETS = {
    "eeg-eye-state": {
        "openml_name": "eeg-eye-state",
        "target": "Class",
        "n_train": 10000,
    },
    "diabetes": {"openml_name": "diabetes", "target": "class", "n_train": 500},
    "churn": {"openml_name": "churn", "target": "class", "n_train": 3500},
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=list(DATASETS), default="eeg-eye-state"
)
parser.add_argument("--k", type=int, default=100)
parser.add_argument("--num-estimators", type=int, default=8)
args = parser.parse_args()

args.num_clusters = 20
args.seed = 0
args.n_train = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

ds = DATASETS[args.dataset]
target_name: str = str(ds["target"])
n_train_default = int(ds["n_train"])
args.n_train = args.n_train if args.n_train is not None else n_train_default

data = fetch_openml(ds["openml_name"], version=1, as_frame=True, parser="auto")
df = data.data.assign(**{target_name: data.target}).sample(
    frac=1, random_state=args.seed
)
stypes = sdm.infer_stypes(df, overrides={target_name: "categorical"})

print(
    f"Dataset: {args.dataset} ({len(df)} rows, n_train={args.n_train}, "
    f"n_test={len(df) - args.n_train}, k={args.k}, clusters={args.num_clusters})"
)
print()

train = sdm.TableTensor.from_pandas(
    df=df.iloc[: args.n_train], stypes=stypes, device=device
)
test = sdm.TableTensor.from_pandas(
    df=df.iloc[args.n_train :], stypes=stypes, device=device
)
n_train = train.size(0)
n_test = test.size(0)

y_true = test[:, target_name].categorical.squeeze(-1)

# --- kNN index ---------------------------------------------------------------

model = sdm.models.TabICLv2(device=device)
recipe = knn_recipe()

knn_execution = RecipeExecution(recipe)
knn_contexts = knn_execution.fit_transform(
    x=train.drop_columns(target_name),
    y=train[:, target_name],
    related_tables=None,
    num_members=1,
)
knn_queries = knn_execution.transform(
    x=test.drop_columns(target_name),
    related_tables=None,
)

train_feat = knn_contexts[0].x.numerical
test_feat = knn_queries[0].x.numerical

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

dists = torch.cdist(test_norm, train_norm)
knn_indices = dists.topk(args.k, dim=-1, largest=False).indices

# --- Cluster -----------------------------------------------------------------

num_clusters = min(args.num_clusters, n_test)
kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed, n_init=10)
cluster_labels = kmeans.fit_predict(test_norm.cpu().numpy())

train_labels = train[:, target_name].categorical.squeeze(-1)

# --- Per-cluster diagnostic --------------------------------------------------

autocast = torch.amp.autocast(
    device.type, torch.bfloat16, enabled=train.is_cuda
)

print(
    f"{'clust':>5} {'queries':>7} {'ctx_sz':>6} {'ctx_%pos':>8} "
    f"{'raw_auc':>7} {'adj_auc':>7} {'col1=pos':>8} {'mean_conf':>9} {'num_cls':>7}"
)
print("-" * 80)

all_probs = torch.zeros(n_test, 2, device=device)

with autocast:
    for c in range(num_clusters):
        mask = cluster_labels == c
        query_idx = torch.where(torch.tensor(mask, device=device))[0]
        if len(query_idx) == 0:
            continue

        ctx_idx = knn_indices[query_idx].unique()
        context = train[ctx_idx]

        ctx_labels = train_labels[ctx_idx].float()
        ctx_pct_pos = ctx_labels.mean().item()

        gen = torch.Generator(device=device).manual_seed(args.seed)
        model.fit(
            x=context.drop_columns(target_name),
            y=context[:, target_name],
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=gen,
        )
        pred = model.predict(test[query_idx].drop_columns(target_name))
        model.clear()

        probs = pred.numerical
        num_out_classes = probs.size(-1)
        if probs.size(-1) < 2:
            probs = torch.nn.functional.pad(probs, (0, 2 - probs.size(-1)))

        y_cluster = y_true[query_idx].cpu().numpy()
        scores = probs[:, -1].float().cpu().numpy()

        if len(np.unique(y_cluster)) < 2:
            raw_auc_val = float("nan")
        else:
            raw_auc_val = roc_auc_score(y_cluster, scores)

        adj_auc_val = (
            max(raw_auc_val, 1 - raw_auc_val)
            if not np.isnan(raw_auc_val)
            else float("nan")
        )
        col1_is_pos = raw_auc_val > 0.5 if not np.isnan(raw_auc_val) else None

        confidence = probs.max(dim=-1).values.float().mean().item()

        all_probs[query_idx] = probs.to(all_probs.dtype)

        print(
            f"{c:>5} {len(query_idx):>7} {ctx_idx.size(0):>6} {ctx_pct_pos:>8.3f} "
            f"{raw_auc_val:>7.3f} {adj_auc_val:>7.3f} {str(col1_is_pos):>8} {confidence:>9.3f} {num_out_classes:>7}"
        )

print("-" * 80)

# --- Aggregate ---------------------------------------------------------------

all_scores = all_probs[:, -1].float().cpu().numpy()
y_all = y_true.cpu().numpy()
raw_total = roc_auc_score(y_all, all_scores)
adj_total = max(raw_total, 1 - raw_total)

# Count orientation
col1_pos_count = 0
col1_neg_count = 0
for c in range(num_clusters):
    mask = cluster_labels == c
    query_idx = torch.where(torch.tensor(mask, device=device))[0]
    if len(query_idx) == 0:
        continue
    y_c = y_true[query_idx].cpu().numpy()
    s_c = all_probs[query_idx, -1].float().cpu().numpy()
    if len(np.unique(y_c)) < 2:
        continue
    if roc_auc_score(y_c, s_c) > 0.5:
        col1_pos_count += 1
    else:
        col1_neg_count += 1

print(f"Aggregate raw AUC: {raw_total:.3f}, adjusted: {adj_total:.3f}")
print(
    f"Column orientation: {col1_pos_count} clusters col1=positive, {col1_neg_count} clusters col1=negative"
)
print(f"Total queries: {n_test}, total clusters: {num_clusters}")
