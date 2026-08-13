"""kNN context selection for TabICLv2.

Reproduces the core idea from LoCalPFN (Thomas et al., NeurIPS 2024) with
TabICLv2: instead of using the full training set as in-context examples,
retrieve the k nearest neighbors for each query row and use only those as
context. Compares four strategies:

  1. Full context  — all training rows (TabICLv2 default)
  2. Random context — k random training rows (controls for context size)
  3. kNN context    — k nearest training rows per query row
"""

# ruff: noqa
import argparse

import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

import sdm
from sdm.processing.execution import RecipeExecution

from recipe import knn_recipe

parser = argparse.ArgumentParser()
parser.add_argument(
    "--k", type=int, default=100, help="context size for kNN / random"
)
parser.add_argument("--num-estimators", type=int, default=4)
parser.add_argument("--num-random-draws", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-train", type=int, default=10000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

target_name = "Class"
data = fetch_openml("eeg-eye-state", version=1, as_frame=True, parser="auto")
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

y_true = test[:, target_name].categorical.squeeze(-1)  # [N_test]
y_true_np = y_true.cpu().numpy()


def auc(probs: torch.Tensor) -> float:
    score = probs[:, -1].float().cpu().numpy()
    raw = roc_auc_score(y_true_np, score)
    return max(raw, 1 - raw)


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
    f"Full context  ({n_train} rows):           {auc(pred_full.numerical):.3f}"
)

# 2. Random context (averaged over several draws)
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
    random_aucs.append(auc(pred_rand.numerical))
mean_auc = sum(random_aucs) / len(random_aucs)
print(
    f"Random context ({args.k} rows, avg {args.num_random_draws} draws): {mean_auc:.3f}"
)

# 3. kNN context (per-query)
num_classes = int(train[:, target_name].categorical.max().item()) + 1
knn_probs = []
with autocast:
    for i in range(n_test):
        context = train[knn_indices[i]]
        gen = torch.Generator(device=device).manual_seed(args.seed)
        pred = model(
            x_context=context.drop_columns(target_name),
            y_context=context[:, target_name],
            x_query=test[i : i + 1].drop_columns(target_name),
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=gen,
        )
        probs = pred.numerical  # [1, num_classes_seen]
        if probs.size(-1) < num_classes:
            probs = torch.nn.functional.pad(
                probs, (0, num_classes - probs.size(-1))
            )
        knn_probs.append(probs)
        if (i + 1) % 50 == 0 or i == n_test - 1:
            print(f"  kNN {i + 1}/{n_test}", flush=True)

knn_all_probs = torch.cat(knn_probs, dim=0)  # [N_test, num_classes]
knn_auc = auc(knn_all_probs)
print(f"kNN context    ({args.k} rows per query):   {knn_auc:.3f}")
