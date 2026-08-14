"""Fine-tune TabICLv2 with 2-hop kNN context.

Same as knn_finetune.py but context is the union of 1-hop and 2-hop
neighbors in the kNN graph, subsampled to k rows. 2-hop neighbors are
rows that are neighbors of neighbors — structurally related but not
necessarily close in direct feature distance.
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
parser.add_argument("--dataset", choices=list(DATASETS), default="coupon")
parser.add_argument("--k", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--num-estimators", type=int, default=8)
args = parser.parse_args()

args.epochs = 5
args.batch_size = 64
args.chunk_size = 256
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

train_feats = [ctx.x.numerical for ctx in contexts]
test_feats = [qry.x.numerical for qry in queries]
train_ys = [ctx.y.categorical.code.squeeze(-1).long() for ctx in contexts]
num_classes = int(train_ys[0].max().item()) + 1
n_members = args.num_estimators

train_feat = train_feats[0]
test_feat = test_feats[0]
train_y = train_ys[0]

print(
    f"Features: {train_feat.size(-1)} cols, {num_classes} classes, {n_members} members"
)

# --- 1-hop kNN indices -------------------------------------------------------

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

knn_chunk = 5000

train_knn_parts = []
train_norm_cpu = train_norm.cpu()
for i in range(0, n_train, knn_chunk):
    chunk_dists = torch.cdist(
        train_norm_cpu[i : i + knn_chunk], train_norm_cpu
    )
    for j in range(chunk_dists.size(0)):
        chunk_dists[j, i + j] = float("inf")
    train_knn_parts.append(
        chunk_dists.topk(args.k, dim=-1, largest=False).indices
    )
    print(
        f"  train kNN chunk {min(i + knn_chunk, n_train)}/{n_train}",
        flush=True,
    )
train_knn_1hop = torch.cat(train_knn_parts, dim=0)  # [N_train, k] on CPU
del train_knn_parts

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
test_knn_1hop = torch.cat(test_knn_parts, dim=0)  # [N_test, k] on CPU
del test_knn_parts, train_norm_cpu, test_norm_cpu

print(
    f"1-hop kNN computed (train: {train_knn_1hop.shape}, test: {test_knn_1hop.shape})"
)

# --- 2-hop: union of 1-hop and neighbors-of-neighbors, subsampled to k -------


def build_2hop_indices(
    knn_1hop: torch.Tensor, n_rows: int, k: int, seed: int
) -> torch.Tensor:
    """Build 2-hop context indices: union of 1-hop + 2-hop, subsampled to k."""
    result = torch.zeros(n_rows, k, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for i in range(n_rows):
        hop1 = knn_1hop[i]  # [k]
        hop2 = knn_1hop[hop1].reshape(-1)  # [k*k]
        union = torch.cat([hop1, hop2]).unique()
        # Exclude self for training rows
        union = union[union != i]
        if union.size(0) >= k:
            perm = torch.randperm(union.size(0), generator=gen)[:k]
            result[i] = union[perm]
        else:
            # Pad with random duplicates if union is too small
            pad = union[
                torch.randint(
                    union.size(0), (k - union.size(0),), generator=gen
                )
            ]
            result[i] = torch.cat([union, pad])
        if (i + 1) % 2000 == 0 or i == n_rows - 1:
            print(f"  2-hop {i + 1}/{n_rows}", flush=True)
    return result


print("Building 2-hop train indices...")
train_knn_2hop = build_2hop_indices(
    train_knn_1hop, n_train, args.k, args.seed
).to(device)
print("Building 2-hop test indices...")
test_knn_2hop = build_2hop_indices(
    test_knn_1hop, n_test, args.k, args.seed + 1
).to(device)

# Also move 1-hop to device for baseline comparison
train_knn_1hop = train_knn_1hop.to(device)
test_knn_1hop = test_knn_1hop.to(device)

print()

# --- Eval helpers -------------------------------------------------------------


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


def eval_knn_batched(
    inner_model: torch.nn.Module, test_knn: torch.Tensor
) -> float:
    """Evaluate kNN context with batched forward pass, averaged over members."""
    member_probs = []
    for m in range(n_members):
        knn_train_f = train_feats[m][test_knn]
        query_f = test_feats[m].unsqueeze(-2)
        x_batched = torch.cat([knn_train_f, query_f], dim=-2)
        y_batched = train_ys[m][test_knn]

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

    avg_probs = torch.stack(member_probs, dim=0).mean(dim=0)
    return auc(avg_probs.cpu().numpy())


# --- Pre-finetune baselines ---------------------------------------------------

autocast = torch.amp.autocast(
    device.type, torch.bfloat16, enabled=train_feat.is_cuda
)

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
print(f"Pre-finetune  full context ({n_train} rows):  {full_auc:.3f}")

print("  1-hop eval...")
auc_1hop_before = eval_knn_batched(model.cls_model, test_knn_1hop)
print(f"Pre-finetune  1-hop kNN ({args.k} rows):      {auc_1hop_before:.3f}")

print("  2-hop eval...")
auc_2hop_before = eval_knn_batched(model.cls_model, test_knn_2hop)
print(f"Pre-finetune  2-hop kNN ({args.k} rows):      {auc_2hop_before:.3f}")
print()

# --- Fine-tuning with 2-hop context ------------------------------------------

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

        m = step % n_members

        neighbor_feat = train_feats[m][train_knn_2hop[batch_idx]]
        query_feat_b = train_feats[m][batch_idx].unsqueeze(-2)
        x = torch.cat([neighbor_feat, query_feat_b], dim=-2)

        y_context = train_ys[m][train_knn_2hop[batch_idx]]
        y_target = train_ys[m][batch_idx]

        with torch.amp.autocast(
            device.type, torch.bfloat16, enabled=train_feat.is_cuda
        ):
            logits = inner_model(x, y_context, num_classes=num_classes)
            logits = logits.squeeze(-2)[..., :num_classes]
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

    inner_model.eval()
    auc_2hop_epoch = eval_knn_batched(inner_model, test_knn_2hop)
    inner_model.train()

    print(
        f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, 2-hop AUC={auc_2hop_epoch:.3f}"
    )

# --- Post-finetune eval -------------------------------------------------------

inner_model.eval()
print()
print("Post-finetune evaluation:")

auc_1hop_after = eval_knn_batched(inner_model, test_knn_1hop)
print(
    f"  1-hop kNN ({args.k} rows):  {auc_1hop_after:.3f}  (was {auc_1hop_before:.3f}, delta {auc_1hop_after - auc_1hop_before:+.3f})"
)

auc_2hop_after = eval_knn_batched(inner_model, test_knn_2hop)
print(
    f"  2-hop kNN ({args.k} rows):  {auc_2hop_after:.3f}  (was {auc_2hop_before:.3f}, delta {auc_2hop_after - auc_2hop_before:+.3f})"
)
