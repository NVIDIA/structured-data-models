"""Batched kNN context selection for TabICLv2.

Bypasses the ICLModel wrapper and calls _TabICLv2 directly with a batch
dimension, so all per-query contexts are processed in one forward pass.
No Python loop over test rows.

Strategies compared:
  1. Full context  — all training rows (via ICLModel, single forward pass)
  2. Random context — k random training rows (via ICLModel)
  3. kNN context   — k nearest neighbors per query, batched [B, k+1, C]
"""

# ruff: noqa
import argparse

import numpy as np
import torch
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
parser.add_argument(
    "--k", type=int, default=100, help="kNN neighbors per query"
)
parser.add_argument("--num-estimators", type=int, default=4)
parser.add_argument("--num-random-draws", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-train", type=int, default=None)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=256,
    help="max queries per forward pass (memory control)",
)
args = parser.parse_args()

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

train = sdm.TableTensor.from_pandas(
    df=df.iloc[: args.n_train], stypes=stypes, device=device
)
test = sdm.TableTensor.from_pandas(
    df=df.iloc[args.n_train :], stypes=stypes, device=device
)
n_train = train.size(0)
n_test = test.size(0)

y_true = test[:, target_name].categorical.squeeze(-1)
y_true_np = y_true.cpu().numpy()
n_classes_true = len(np.unique(y_true_np))

print(
    f"Dataset: {args.dataset} ({len(df)} rows, n_train={n_train}, "
    f"n_test={n_test}, k={args.k}, chunk_size={args.chunk_size})"
)
print()


def auc(scores: np.ndarray) -> float:
    if n_classes_true == 2 and scores.shape[-1] == 2:
        raw = roc_auc_score(y_true_np, scores[:, -1])
        return max(raw, 1 - raw)
    if scores.shape[-1] < n_classes_true:
        pad_width = n_classes_true - scores.shape[-1]
        scores = np.pad(scores, ((0, 0), (0, pad_width)))
    return roc_auc_score(
        y_true_np, scores, multi_class="ovr", average="weighted"
    )


# --- Preprocess features with recipe -----------------------------------------

model = sdm.models.TabICLv2(device=device)
recipe = knn_recipe()

execution = RecipeExecution(recipe)
gen = torch.Generator(device=device).manual_seed(args.seed)
contexts = execution.fit_transform(
    x=train.drop_columns(target_name),
    y=train[:, target_name],
    related_tables=None,
    num_members=1,
    generator=gen,
)
queries = execution.transform(
    x=test.drop_columns(target_name),
    related_tables=None,
)

train_feat = contexts[0].x.numerical  # [N_train, C]
test_feat = queries[0].x.numerical  # [N_test, C]
train_y = contexts[0].y.categorical.code.squeeze(-1)  # [N_train]
num_classes = int(train_y.max().item()) + 1

# --- kNN index ---------------------------------------------------------------

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

dists = torch.cdist(test_norm, train_norm)  # [N_test, N_train]
knn_indices = dists.topk(args.k, dim=-1, largest=False).indices  # [N_test, k]

# --- Evaluation ---------------------------------------------------------------
autocast = torch.amp.autocast(
    device.type, torch.bfloat16, enabled=train.is_cuda
)

# 1. Full context (via ICLModel wrapper, standard path)
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
full_scores = pred_full.numerical.float().cpu().numpy()
print(f"Full context  ({n_train} rows):           {auc(full_scores):.3f}")

# 2. Random context (via ICLModel wrapper)
random_aucs = []
for i in range(args.num_random_draws):
    rand_gen = torch.Generator(device=device).manual_seed(args.seed + i)
    indices = torch.randperm(n_train, device=device, generator=rand_gen)[
        : args.k
    ]
    context = train[indices]
    with autocast:
        model_gen = torch.Generator(device=device).manual_seed(args.seed)
        pred_rand = model(
            x_context=context.drop_columns(target_name),
            y_context=context[:, target_name],
            x_query=test.drop_columns(target_name),
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=model_gen,
        )
    random_aucs.append(auc(pred_rand.numerical.float().cpu().numpy()))
mean_auc = sum(random_aucs) / len(random_aucs)
print(
    f"Random context ({args.k} rows, avg {args.num_random_draws} draws): {mean_auc:.3f}"
)

# 3. Batched kNN context (direct _TabICLv2 call)
# Access the inner classification model
inner_model = model.cls_model

# Build batched inputs: for each test row, gather its k neighbors + the query
# x_batched: [B, k+1, C] where first k rows are context, last row is query
# y_batched: [B, k] labels for context rows
knn_train_feat = train_feat[knn_indices]  # [N_test, k, C]
query_feat = test_feat.unsqueeze(-2)  # [N_test, 1, C]
x_batched = torch.cat([knn_train_feat, query_feat], dim=-2)  # [N_test, k+1, C]
y_batched = train_y[knn_indices]  # [N_test, k]

# Process in chunks to control memory
all_logits = []
n_chunks = (n_test + args.chunk_size - 1) // args.chunk_size
with autocast, torch.inference_mode():
    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, n_test)
        logits = inner_model(
            x_batched[start:end],
            y_batched[start:end],
            num_classes=num_classes,
        )  # [chunk, 1, num_classes]
        all_logits.append(logits)
        if (chunk_idx + 1) % 50 == 0 or chunk_idx == n_chunks - 1:
            print(f"  kNN chunk {chunk_idx + 1}/{n_chunks}", flush=True)

knn_logits = torch.cat(all_logits, dim=0).squeeze(-2)  # [N_test, 10]
knn_logits = knn_logits[..., :num_classes]  # [N_test, num_classes]
knn_probs = torch.softmax(knn_logits.float() / 0.9, dim=-1)
knn_scores = knn_probs.cpu().numpy()
print(
    f"kNN context    ({args.k} rows per query, batched): {auc(knn_scores):.3f}"
)
