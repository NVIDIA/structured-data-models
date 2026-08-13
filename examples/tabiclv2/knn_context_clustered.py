"""Clustered kNN context selection for TabICLv2.

Groups test rows by feature similarity (k-means), then uses the union of
each cluster's kNN sets as shared context via fit/predict caching. This
avoids the per-query forward pass bottleneck while preserving approximate
locality.

Strategies compared:
  1. Full context   — all training rows
  2. Random context — k random training rows
  3. Clustered kNN  — k-means clusters of test rows, union of kNN sets as
                      context per cluster, fit/predict caching
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
    "customer-satisfaction": {
        "openml_name": "customer_satisfaction_in_airline",
        "target": "satisfaction",
        "n_train": 100000,
    },
    "sdss17": {
        "openml_name": "SDSS17",
        "target": "ObjectType",
        "n_train": 52000,
    },
    "diabetes130": {
        "openml_name": "Diabetes130US",
        "target": "readmitted",
        "n_train": 47679,
    },
    "coupon": {
        "openml_name": "in_vehicle_coupon_recommendation",
        "target": "AcceptCoupon",
        "n_train": 8456,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=list(DATASETS), default="eeg-eye-state"
)
parser.add_argument("--k", type=int, default=100)
parser.add_argument("--num-estimators", type=int, default=8)
args = parser.parse_args()

args.num_clusters = 20
args.num_random_draws = 5
args.seed = 0
args.n_train = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

ds = DATASETS[args.dataset]
target_name: str = str(ds["target"])
args.n_train = args.n_train if args.n_train is not None else int(ds["n_train"])

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

y_true = test[:, target_name].categorical.squeeze(-1)  # [N_test]
y_true_np = y_true.cpu().numpy()


n_classes_true = len(np.unique(y_true_np))


def auc(probs: torch.Tensor) -> float:
    scores = probs.float().cpu().numpy()
    if n_classes_true == 2 and scores.shape[-1] == 2:
        raw = roc_auc_score(y_true_np, scores[:, -1])
        return max(raw, 1 - raw)
    if scores.shape[-1] < n_classes_true:
        pad_width = n_classes_true - scores.shape[-1]
        scores = np.pad(scores, ((0, 0), (0, pad_width)))
    return roc_auc_score(
        y_true_np, scores, multi_class="ovr", average="weighted"
    )


# --- kNN index (on preprocessed features) ------------------------------------

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

train_feat = knn_contexts[0].x.numerical  # [N_train, C]
test_feat = knn_queries[0].x.numerical  # [N_test, C]

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

dists = torch.cdist(test_norm, train_norm)  # [N_test, N_train]
knn_indices = dists.topk(args.k, dim=-1, largest=False).indices  # [N_test, k]

# --- Cluster test rows -------------------------------------------------------

num_clusters = min(args.num_clusters, n_test)
kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed, n_init=10)
cluster_labels = kmeans.fit_predict(test_norm.cpu().numpy())

# --- Evaluation ---------------------------------------------------------------
autocast = torch.amp.autocast(
    device.type, torch.bfloat16, enabled=train.is_cuda
)

# 1. Full context
with autocast:
    gen = torch.Generator(device=device).manual_seed(args.seed)
    pred_full = model(
        x_context=train.drop_columns(target_name),
        y_context=train[:, target_name],
        x_query=test.drop_columns(target_name),
        recipe=recipe,
        num_estimators=args.num_estimators,
        generator=gen,
    )
print(
    f"Full context    ({n_train} rows):           {auc(pred_full.numerical):.3f}"
)

# 2. Random context (averaged over several draws)
random_aucs = []
for i in range(args.num_random_draws):
    gen = torch.Generator(device=device).manual_seed(args.seed + i)
    indices = torch.randperm(n_train, device=device, generator=gen)[: args.k]
    context = train[indices]
    with autocast:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        pred_rand = model(
            x_context=context.drop_columns(target_name),
            y_context=context[:, target_name],
            x_query=test.drop_columns(target_name),
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=gen,
        )
    random_aucs.append(auc(pred_rand.numerical))
mean_auc = sum(random_aucs) / len(random_aucs)
print(
    f"Random context  ({args.k} rows, avg {args.num_random_draws} draws): {mean_auc:.3f}"
)

# 3. Clustered kNN context (fit/predict per cluster)
num_classes = int(train[:, target_name].categorical.max().item()) + 1
clustered_probs = torch.zeros(n_test, num_classes, device=device)

with autocast:
    for c in range(num_clusters):
        mask = cluster_labels == c
        query_indices = torch.where(torch.tensor(mask, device=device))[0]
        if len(query_indices) == 0:
            continue

        ctx_indices = knn_indices[query_indices].unique()
        context = train[ctx_indices]

        gen = torch.Generator(device=device).manual_seed(args.seed)
        model.fit(
            x=context.drop_columns(target_name),
            y=context[:, target_name],
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=gen,
        )
        pred = model.predict(test[query_indices].drop_columns(target_name))
        model.clear()

        probs = pred.numerical
        if probs.size(-1) < num_classes:
            probs = torch.nn.functional.pad(
                probs, (0, num_classes - probs.size(-1))
            )
        clustered_probs[query_indices] = probs.to(clustered_probs.dtype)

        print(
            f"  Cluster {c + 1}/{num_clusters}: "
            f"{len(query_indices)} queries, {ctx_indices.size(0)} ctx rows",
            flush=True,
        )

clustered_auc = auc(clustered_probs)
print(
    f"Clustered kNN   ({num_clusters} clusters, k={args.k}): {clustered_auc:.3f}"
)
