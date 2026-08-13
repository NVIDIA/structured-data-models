"""Fine-tune TabICLv2 with kNN-retrieved local context.

Reproduces the full LoCalPFN approach: for each training row, retrieve its
k nearest neighbors (excluding itself) as context, predict the row, and
backprop through the model. After fine-tuning, evaluate with the batched
kNN inference from knn_context_batched.py.

Steps:
  1. Preprocess features with recipe
  2. Precompute kNN indices for all training rows (leave-one-out)
  3. Fine-tune _TabICLv2 on (neighbors → predict row) pairs
  4. Evaluate: full context, random context, kNN batched
"""

# ruff: noqa
import argparse
import numpy as np
import torch
import torch.nn.functional as F
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
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="training batch size (number of query rows per step)",
)
parser.add_argument(
    "--chunk-size", type=int, default=256, help="eval chunk size"
)
parser.add_argument(
    "--num-estimators", type=int, default=4, help="for full/random baselines"
)
parser.add_argument("--num-random-draws", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-train", type=int, default=None)
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

train_table = sdm.TableTensor.from_pandas(
    df=df.iloc[: args.n_train], stypes=stypes, device=device
)
test_table = sdm.TableTensor.from_pandas(
    df=df.iloc[args.n_train :], stypes=stypes, device=device
)
n_train = train_table.size(0)
n_test = test_table.size(0)

y_true = test_table[:, target_name].categorical.squeeze(-1)
y_true_np = y_true.cpu().numpy()
n_classes_true = len(np.unique(y_true_np))

print(
    f"Dataset: {args.dataset} ({len(df)} rows, n_train={n_train}, "
    f"n_test={n_test}, k={args.k}, lr={args.lr}, epochs={args.epochs})"
)
print()

# --- Preprocess features with recipe -----------------------------------------

model = sdm.models.TabICLv2(device=device)
recipe = knn_recipe()

execution = RecipeExecution(recipe)
gen = torch.Generator(device=device).manual_seed(args.seed)
contexts = execution.fit_transform(
    x=train_table.drop_columns(target_name),
    y=train_table[:, target_name],
    related_tables=None,
    num_members=args.num_estimators,
    generator=gen,
)
queries = execution.transform(
    x=test_table.drop_columns(target_name),
    related_tables=None,
)

# Per-member preprocessed features and labels
train_feats = [ctx.x.numerical for ctx in contexts]  # list of [N_train, C]
test_feats = [qry.x.numerical for qry in queries]  # list of [N_test, C]
train_ys = [
    ctx.y.categorical.code.squeeze(-1).long() for ctx in contexts
]  # list of [N_train]
num_classes = int(train_ys[0].max().item()) + 1
n_members = args.num_estimators

# Use member 0 as reference for kNN and shapes
train_feat = train_feats[0]
test_feat = test_feats[0]
train_y = train_ys[0]

print(
    f"Features: {train_feat.size(-1)} cols, {num_classes} classes, {n_members} members"
)

# --- kNN indices (precomputed, leave-one-out for train) ----------------------

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

knn_chunk = 5000

# Train kNN: each row's k nearest neighbors, excluding itself (chunked)
train_knn_parts = []
train_norm_cpu = train_norm.cpu()
for i in range(0, n_train, knn_chunk):
    chunk_dists = torch.cdist(
        train_norm_cpu[i : i + knn_chunk], train_norm_cpu
    )
    # Exclude self: set diagonal block to inf
    for j in range(chunk_dists.size(0)):
        chunk_dists[j, i + j] = float("inf")
    train_knn_parts.append(
        chunk_dists.topk(args.k, dim=-1, largest=False).indices
    )
    print(
        f"  train kNN chunk {min(i + knn_chunk, n_train)}/{n_train}",
        flush=True,
    )
train_knn = torch.cat(train_knn_parts, dim=0).to(device)  # [N_train, k]
del train_knn_parts

# Test kNN (chunked)
test_knn_parts = []
test_norm_cpu = test_norm.cpu()
for i in range(0, n_test, knn_chunk):
    chunk_dists = torch.cdist(test_norm_cpu[i : i + knn_chunk], train_norm_cpu)
    test_knn_parts.append(
        chunk_dists.topk(args.k, dim=-1, largest=False).indices
    )
    print(
        f"  test kNN chunk {min(i + knn_chunk, n_test)}/{n_test}", flush=True
    )
test_knn = torch.cat(test_knn_parts, dim=0).to(device)  # [N_test, k]
del test_knn_parts, train_norm_cpu, test_norm_cpu

print(
    f"kNN indices computed (train: {train_knn.shape}, test: {test_knn.shape})"
)
print()

# --- Eval helper --------------------------------------------------------------


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


def eval_knn_batched(inner_model: torch.nn.Module) -> float:
    """Evaluate kNN context with batched forward pass, averaged over members."""
    member_probs = []
    for m in range(n_members):
        knn_train_f = train_feats[m][test_knn]  # [N_test, k, C]
        query_f = test_feats[m].unsqueeze(-2)  # [N_test, 1, C]
        x_batched = torch.cat([knn_train_f, query_f], dim=-2)
        y_batched = train_ys[m][test_knn]  # [N_test, k]

        all_logits = []
        n_chunks = (n_test + args.chunk_size - 1) // args.chunk_size
        with torch.amp.autocast(
            device.type, torch.bfloat16, enabled=train_feat.is_cuda
        ):
            with torch.inference_mode():
                for chunk_idx in range(n_chunks):
                    start = chunk_idx * args.chunk_size
                    end = min(start + args.chunk_size, n_test)
                    logits = inner_model(
                        x_batched[start:end],
                        y_batched[start:end],
                        num_classes=num_classes,
                    )
                    all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0).squeeze(-2)[..., :num_classes]
        member_probs.append(torch.softmax(logits.float() / 0.9, dim=-1))
        print(f"  eval member {m + 1}/{n_members}", flush=True)

    avg_probs = torch.stack(member_probs, dim=0).mean(
        dim=0
    )  # [N_test, num_classes]
    return auc(avg_probs.cpu().numpy())


# --- Pre-finetune baselines ---------------------------------------------------

autocast = torch.amp.autocast(
    device.type, torch.bfloat16, enabled=train_feat.is_cuda
)

# Full context baseline
with autocast:
    gen = torch.Generator(device=device).manual_seed(args.seed)
    pred_full = model(
        x_context=train_table.drop_columns(target_name),
        y_context=train_table[:, target_name],
        x_query=test_table.drop_columns(target_name),
        recipe=recipe,
        num_estimators=args.num_estimators,
        generator=gen,
    )
full_auc = auc(pred_full.numerical.float().cpu().numpy())
print(f"Pre-finetune  full context ({n_train} rows): {full_auc:.3f}")

# Random baseline
random_aucs = []
for i in range(args.num_random_draws):
    rand_gen = torch.Generator(device=device).manual_seed(args.seed + i)
    indices = torch.randperm(n_train, device=device, generator=rand_gen)[
        : args.k
    ]
    context = train_table[indices]
    with autocast:
        model_gen = torch.Generator(device=device).manual_seed(args.seed)
        pred_rand = model(
            x_context=context.drop_columns(target_name),
            y_context=context[:, target_name],
            x_query=test_table.drop_columns(target_name),
            recipe=recipe,
            num_estimators=args.num_estimators,
            generator=model_gen,
        )
    random_aucs.append(auc(pred_rand.numerical.float().cpu().numpy()))
rand_auc = sum(random_aucs) / len(random_aucs)
print(f"Pre-finetune  random ({args.k} rows):        {rand_auc:.3f}")

# kNN batched baseline (before fine-tuning)
knn_auc_before = eval_knn_batched(model.cls_model)
print(f"Pre-finetune  kNN batched ({args.k} rows):   {knn_auc_before:.3f}")
print()

# --- Fine-tuning --------------------------------------------------------------

inner_model = model.cls_model
inner_model.train()
optimizer = torch.optim.Adam(inner_model.parameters(), lr=args.lr)

n_steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size

for epoch in range(args.epochs):
    perm = torch.randperm(n_train, device=device)
    epoch_loss = 0.0
    n_batches = 0

    for step in range(n_steps_per_epoch):
        start = step * args.batch_size
        end = min(start + args.batch_size, n_train)
        batch_idx = perm[start:end]
        bs = batch_idx.size(0)

        # Pick a random member for this step
        m = step % n_members

        # Build [bs, k+1, C]: k neighbors + query row
        neighbor_feat = train_feats[m][train_knn[batch_idx]]  # [bs, k, C]
        query_feat_b = train_feats[m][batch_idx].unsqueeze(-2)  # [bs, 1, C]
        x = torch.cat([neighbor_feat, query_feat_b], dim=-2)  # [bs, k+1, C]

        # Labels: [bs, k] for context, [bs] for query target
        y_context = train_ys[m][train_knn[batch_idx]]  # [bs, k]
        y_target = train_ys[m][batch_idx]  # [bs]

        with torch.amp.autocast(
            device.type, torch.bfloat16, enabled=train_feat.is_cuda
        ):
            logits = inner_model(
                x, y_context, num_classes=num_classes
            )  # [bs, 1, 10]
            logits = logits.squeeze(-2)[..., :num_classes]  # [bs, num_classes]
            loss = F.cross_entropy(logits, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
        if n_batches % 50 == 0:
            print(
                f"  step {n_batches}/{n_steps_per_epoch}, loss={loss.item():.4f}",
                flush=True,
            )

    avg_loss = epoch_loss / n_batches

    # Eval after each epoch
    inner_model.eval()
    knn_auc_epoch = eval_knn_batched(inner_model)
    inner_model.train()

    print(
        f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, kNN AUC={knn_auc_epoch:.3f}"
    )

# --- Post-finetune eval -------------------------------------------------------

inner_model.eval()
print()
knn_auc_after = eval_knn_batched(inner_model)
print(f"Post-finetune kNN batched ({args.k} rows):   {knn_auc_after:.3f}")
print(f"Improvement:  {knn_auc_after - knn_auc_before:+.3f}")
