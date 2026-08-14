"""Fine-tune TabICLv2 with 2-hop kNN context via DiskGraph.

Same as knn_finetune.py but uses DiskGraph's sampling engine for multi-hop
neighborhood retrieval instead of a Python loop. The kNN graph is ingested
into DiskGraph's on-disk index, and 2-hop neighborhoods are sampled natively
in Rust.

Requires: diskgraph Python package (build from /home/jana/work/kumo-diskgraph)
"""

# ruff: noqa
import argparse
import json
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
parser.add_argument(
    "--hop1-fanout",
    type=int,
    default=10,
    help="neighbors sampled per node at hop 1",
)
parser.add_argument(
    "--hop2-fanout",
    type=int,
    default=10,
    help="neighbors sampled per node at hop 2",
)
args = parser.parse_args()

args.epochs = 5
args.batch_size = 64
args.chunk_size = 256
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
    f"n_test={n_test}, k={args.k}, lr={args.lr})"
)
print(f"Hop fanouts: hop1={args.hop1_fanout}, hop2={args.hop2_fanout}")
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

# --- Build 1-hop kNN graph ---------------------------------------------------

mean = train_feat.mean(dim=0)
std = train_feat.std(dim=0).clamp(min=1e-8)
train_norm = (train_feat - mean) / std
test_norm = (test_feat - mean) / std

knn_chunk = 5000

# Train kNN (1-hop, for graph construction)
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
train_knn_1hop = torch.cat(train_knn_parts, dim=0)  # [N_train, k] CPU
del train_knn_parts

# Test kNN (1-hop, for baseline comparison)
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
test_knn_1hop = torch.cat(test_knn_parts, dim=0).to(device)  # [N_test, k]
del test_knn_parts, train_norm_cpu, test_norm_cpu

print(f"1-hop kNN computed")

# --- Ingest kNN graph into DiskGraph -----------------------------------------

import diskgraph

tmpdir = tempfile.mkdtemp(prefix="diskgraph_knn_")
parquet_dir = os.path.join(tmpdir, "parquet")
index_dir = os.path.join(tmpdir, "index")
os.makedirs(parquet_dir)

# Write nodes parquet
node_ids = pa.array(list(range(n_train)), type=pa.int64())
nodes_table = pa.table({"node_id": node_ids})
pq.write_table(nodes_table, os.path.join(parquet_dir, "nodes.parquet"))

# Write edges parquet (flatten kNN adjacency)
src_ids = np.repeat(np.arange(n_train), args.k)
dst_ids = train_knn_1hop.numpy().reshape(-1)
edges_table = pa.table(
    {
        "src": pa.array(src_ids, type=pa.int64()),
        "dst": pa.array(dst_ids, type=pa.int64()),
    }
)
pq.write_table(edges_table, os.path.join(parquet_dir, "edges.parquet"))

print(f"Parquet written: {n_train} nodes, {len(src_ids)} edges")

# Build config
config = {
    "dataset_name": "knn_graph",
    "num_shards": 1,
    "tables": [
        {
            "name": "nodes",
            "columns": [
                {"name": "node_id", "ingested_dtype": "int64"},
            ],
            "primary_key": "node_id",
            "source": {
                "type": "parquet",
                "paths": [os.path.join(parquet_dir, "nodes.parquet")],
            },
        },
        {
            "name": "edges",
            "columns": [
                {"name": "src", "ingested_dtype": "int64"},
                {"name": "dst", "ingested_dtype": "int64"},
            ],
            "primary_key": None,
            "fkeys": [
                {"src_column": "src", "dst_table": "nodes", "reverse": {}},
                {"src_column": "dst", "dst_table": "nodes", "reverse": {}},
            ],
            "source": {
                "type": "parquet",
                "paths": [os.path.join(parquet_dir, "edges.parquet")],
            },
        },
    ],
}

print("Ingesting into DiskGraph...", flush=True)
stats = diskgraph.ingest(json.dumps(config), index_dir)
print(f"Ingest complete: {index_dir}")

# --- 2-hop sampling via DiskGraph --------------------------------------------

engine = diskgraph.SamplerEngine(index_dir, random_seed=args.seed)

# Determine relation name from the ingested graph
type_mapping = json.loads(engine.get_type_mapping())
print(f"Type mapping: {json.dumps(type_mapping, indent=2)}")

# Build sampling plan: 2 hops
# The relation name depends on how DiskGraph names the fkey edges
# For embedded fkeys on "edges" table: edges_src_to_nodes_node_id and edges_dst_to_nodes_node_id
plan = {
    "layers": [
        {
            "fanouts": {},
            "default_fanout": {
                "fanout": args.hop1_fanout,
                "mode": "uniform",
                "absent_key_policy": "uniform_fallback",
            },
        },
        {
            "fanouts": {},
            "default_fanout": {
                "fanout": args.hop2_fanout,
                "mode": "uniform",
                "absent_key_policy": "uniform_fallback",
            },
        },
    ]
}


def sample_2hop(engine, query_ids, plan, k, seed=0):
    """Sample 2-hop neighborhoods and return [len(query_ids), k] index tensor."""
    iterator = engine.sample(
        {"nodes": query_ids},
        json.dumps(plan),
        fused=False,
    )

    # Collect all global IDs from node events
    all_global_ids = set()
    for event in iterator:
        if event["type"] == "node":
            batch = event["batch"]
            gids = batch.column("id_b").to_pylist()
            all_global_ids.update(gids)

    # Remove query IDs from neighbor sets
    query_set = set(query_ids)
    neighbor_ids = list(all_global_ids - query_set)

    return neighbor_ids


def build_2hop_indices(engine, all_ids, plan, k, batch_size=500, seed=0):
    """Build 2-hop context indices for all rows, subsampled to k."""
    gen = torch.Generator().manual_seed(seed)
    result = torch.zeros(len(all_ids), k, dtype=torch.long)

    for start in range(0, len(all_ids), batch_size):
        end = min(start + batch_size, len(all_ids))
        batch_ids = all_ids[start:end]

        # Sample each query individually to get per-query neighbors
        for i, qid in enumerate(batch_ids):
            neighbors = sample_2hop(engine, [qid], plan, k, seed)
            neighbors_t = torch.tensor(neighbors, dtype=torch.long)

            if neighbors_t.size(0) >= k:
                perm = torch.randperm(neighbors_t.size(0), generator=gen)[:k]
                result[start + i] = neighbors_t[perm]
            elif neighbors_t.size(0) > 0:
                pad = neighbors_t[
                    torch.randint(
                        neighbors_t.size(0),
                        (k - neighbors_t.size(0),),
                        generator=gen,
                    )
                ]
                result[start + i] = torch.cat([neighbors_t, pad])

        if (end) % 2000 == 0 or end == len(all_ids):
            print(f"  2-hop sampling {end}/{len(all_ids)}", flush=True)

    return result


print("Building 2-hop train indices via DiskGraph...")
train_ids = list(range(n_train))
train_knn_2hop = build_2hop_indices(
    engine, train_ids, plan, args.k, seed=args.seed
).to(device)

print("Building 2-hop test indices via DiskGraph...")
# For test, we sample starting from training nodes closest to test rows
# Use 1-hop test kNN as seed nodes, then expand via DiskGraph
test_knn_2hop_parts = []
test_knn_1hop_cpu = test_knn_1hop.cpu()
gen = torch.Generator().manual_seed(args.seed + 1)
for i in range(n_test):
    # Start from this test row's 1-hop neighbors in the train graph
    seed_nodes = test_knn_1hop_cpu[i, : args.hop1_fanout].tolist()
    neighbors = sample_2hop(engine, seed_nodes, plan, args.k, args.seed)
    neighbors_t = torch.tensor(neighbors, dtype=torch.long)

    row = torch.zeros(args.k, dtype=torch.long)
    if neighbors_t.size(0) >= args.k:
        perm = torch.randperm(neighbors_t.size(0), generator=gen)[: args.k]
        row = neighbors_t[perm]
    elif neighbors_t.size(0) > 0:
        pad = neighbors_t[
            torch.randint(
                neighbors_t.size(0),
                (args.k - neighbors_t.size(0),),
                generator=gen,
            )
        ]
        row = torch.cat([neighbors_t, pad])
    test_knn_2hop_parts.append(row)

    if (i + 1) % 1000 == 0 or i == n_test - 1:
        print(f"  test 2-hop {i + 1}/{n_test}", flush=True)

test_knn_2hop = torch.stack(test_knn_2hop_parts).to(device)  # [N_test, k]
train_knn_1hop = train_knn_1hop.to(device)

print(
    f"2-hop indices: train={train_knn_2hop.shape}, test={test_knn_2hop.shape}"
)
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


def eval_knn_batched(inner_model, test_knn):
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
print(
    f"Pre-finetune  full context ({n_train} rows):  {auc(pred_full.numerical.float().cpu().numpy()):.3f}"
)

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

# Cleanup
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
