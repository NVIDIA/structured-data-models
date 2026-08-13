# knn_context.py changelog

## v1 — Initial script (breast_cancer, numerical-only kNN)

- Three strategies: full context, random-k, kNN-k (per-query)
- Used breast_cancer dataset from sklearn
- kNN computed on raw numerical features only (ignored categoricals)
- Bug: accuracy function assumed estimator dimension in output that doesn't exist — output is already ensembled as `[N_test, num_classes]`
- Bug: kNN loop did `.mean(dim=0)` on `[1, num_classes]`, collapsing to `[num_classes]`, so `torch.cat` produced a flat tensor and `argmax` returned a scalar

## v2 — Switch to churn dataset, preprocessed kNN features

- Switched from breast_cancer to OpenML churn (5000 rows, 20 features, binary)
- kNN now computed on recipe-preprocessed features (all stypes contribute to distance)
- Uses `RecipeExecution` to `fit_transform` train and `transform` test before computing distances
- Added `--n-train` CLI flag (default 3500)
- Fixed accuracy function: removed incorrect `.mean(dim=0)` — output has no estimator dim
- Fixed kNN loop: append `pred.numerical` directly instead of collapsing row dim
- New bug: class-pure kNN neighborhoods cause output to have fewer columns than expected (1 vs 2 for binary) — fixed with zero-padding

## v2 results (seed=0, k=100, n_train=3500, num_estimators=4)

- Full context (3500 rows): 0.971
- Random context (100 rows, avg 5 draws): 0.576
- kNN context (100 rows per query): 0.491
- kNN underperforms random — pure neighborhoods lose contrastive signal
- Random also poor relative to full — 100 rows may be too few

## v3 — Mixed context strategy

- Added 4th strategy: mixed kNN + random (default 80% kNN, 20% random via `--knn-ratio`)
- Random portion drawn from training rows NOT in the kNN set (no overlap)
- Per-query random seed is `seed + i` so each query gets a different random complement
- Added `--knn-ratio` CLI flag

## v3 results (seed=0, k=100, knn_ratio=0.8, n_train=3500, num_estimators=4)

- Full context (3500 rows): 0.971
- Random context (100 rows, avg 5 draws): 0.576
- kNN context (100 rows per query): 0.491
- Mixed context (80 kNN + 20 random): 0.487
- Mixed didn't help — 20 random rows not enough to counteract class-pure kNN neighborhoods
- Runtime: ~15 min (3000 per-query forward passes)

## Observations (churn)

- kNN (0.491) underperforms random (0.576) on churn — class-pure neighborhoods lose contrastive signal (churn is ~14% positive, so most neighborhoods are all-negative)
- Full context (0.971) is much better than k=100 baselines — 100 rows may be too few for this dataset
- The class-purity problem required zero-padding fix: when all context rows are one class, model outputs 1 column instead of 2
- Churn's heavy imbalance (86/14) makes it a poor test for kNN context selection

## v4 — Switch to eeg-eye-state dataset

- Switched from churn to OpenML eeg-eye-state (14,980 rows, 14 numerical features, binary, ~55/45 balance)
- All numerical features — no categorical preprocessing complexity
- One of the datasets highlighted in LoCalPFN paper figures
- Bumped default n_train to 10,000 (test set ~4,980 rows)
- Per-query loops are now ~5,000 passes each — runtime is significant

## v5 — Batched kNN/mixed loops

- Added `--batch-size` flag (default 1 = per-query, same as before)
- When batch-size > 1, groups consecutive test rows and uses the union of their kNN sets as context
- Tradeoff: larger batches = fewer forward passes but less focused context (union of neighborhoods is larger and less query-specific)
- With batch-size=50 and 5,000 test rows: 100 forward passes instead of 5,000
- Context size per batch: up to batch_size * k unique rows (less with overlap between neighbors)

## v5 results (eeg-eye-state, seed=0, k=100, batch_size=32, n_train=10000, num_estimators=4)

- Full context (10,000 rows): 0.010 accuracy
- Random context (100 rows, avg 5 draws): 0.533
- kNN context (k=100, batch=32): 0.494
- Mixed context (80 kNN + 20 random): 0.470
- All results used accuracy metric, which turned out to be unreliable (see v6)

## v6 — Switch from accuracy to AUC

### What happened

Full context accuracy at n_train=10,000 was 0.010 — near zero on a binary task. That's too systematic to be random failure. To investigate, ran a debug script (not part of knn_context.py) that tested the model at various context sizes with a fixed test set (last 200 rows of the shuffled data). Reproduce with:

```bash
python3 -c "
import torch, sdm
from sklearn.datasets import fetch_openml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = fetch_openml('eeg-eye-state', version=1, as_frame=True, parser='auto')
df = data.data.assign(Class=data.target).sample(frac=1, random_state=0)
stypes = sdm.infer_stypes(df, overrides={'Class': 'categorical'})

test = sdm.TableTensor.from_pandas(df=df.iloc[-200:], stypes=stypes, device=device)
y_true = test[:, 'Class'].categorical.squeeze(-1)

model = sdm.models.TabICLv2(device=device)
for n in [100, 300, 500, 1000, 2000]:
    train = sdm.TableTensor.from_pandas(df=df.iloc[:n], stypes=stypes, device=device)
    with torch.amp.autocast(device.type, torch.bfloat16, enabled=train.is_cuda):
        pred = model(
            x_context=train.drop_columns('Class'),
            y_context=train[:, 'Class'],
            x_query=test.drop_columns('Class'),
            num_estimators=4,
        )
    probs = pred.numerical
    preds = probs.argmax(dim=-1)
    acc = (preds == y_true).float().mean().item()
    flipped_acc = (1 - preds == y_true).float().mean().item()
    print(f'n={n:5d}: acc={acc:.3f}, flipped={flipped_acc:.3f}, pred_sample={probs[0].tolist()}')
"
```

### What the debug showed

```
n=  100: acc=0.680, flipped=0.320, pred_sample=[0.097, 0.903]  → column 1 = positive
n=  300: acc=0.860, flipped=0.140, pred_sample=[0.043, 0.957]  → column 1 = positive
n=  500: acc=0.115, flipped=0.885, pred_sample=[0.993, 0.007]  → column 0 = positive (FLIPPED)
n= 1000: acc=0.075, flipped=0.925, pred_sample=[0.995, 0.005]  → column 0 = positive (FLIPPED)
n= 2000: acc=0.040, flipped=0.960, pred_sample=[0.999, 0.001]  → column 0 = positive (FLIPPED)
```

### Diagnosis

- The model is learning well at all sizes — flipped accuracy improves from 88.5% to 96% as context grows
- Between n=300 and n=500, the output column mapping inverts: the column that carries the positive-class probability switches from column 1 to column 0
- Category encoding was ruled out — both train and test use the same ordering (cat[0]=1, cat[1]=2) at all sizes
- The flip is internal to the model — its output column assignment is not stable across context sizes
- All previous accuracy-based results on eeg-eye-state were unreliable because of this

### Fix

- Switched metric from accuracy (argmax-based, sensitive to column ordering) to AUC (rank-based, invariant to column ordering)
- This also matches the LoCalPFN paper, which reports AUC throughout
- Additional fix: `auc()` uses `max(raw, 1 - raw)` because the column flip also affects which column `roc_auc_score` treats as positive. An AUC of 0.012 is really 0.988 with flipped columns. `max(raw, 1 - raw)` makes it invariant to column ordering.

### Remaining issue: per-batch column flip in kNN/mixed loops

The `max(raw, 1 - raw)` fix works for full context and random context (single forward pass, consistent column assignment). But the kNN and mixed loops concatenate probabilities across many batches. If different batches flip the column assignment differently (because they have different context sizes — the union of kNN sets varies per batch), the concatenated probabilities are a mix of both orientations. AUC computed on this mixture is unreliable — some batches contribute correct rankings, others inverted ones, and they partially cancel out.

Results before `max(raw, 1-raw)` fix (n_train=1000, batch_size=32):

- Full context (1000 rows): 0.012 (→ 0.988 after fix)
- Random context (100 rows, avg 5 draws): 0.577 (→ 0.577, no flip here since context size is stable at 100)
- kNN context (k=100, batch=32): 0.474 (unreliable — mixed orientations across batches)
- Mixed context (80 kNN + 20 random): 0.486 (unreliable — same issue)

## v7 — Switch to diabetes, drop batching

### Why

Two problems with the eeg-eye-state + batching setup:

1. **Batching undermines locality.** The whole point of kNN context is that each query sees only its own local neighborhood. Batching groups queries and gives them the union of their neighborhoods — which dilutes the locality and defeats the purpose. This should have been flagged earlier.

2. **eeg-eye-state is too large.** 14,980 rows means thousands of per-query forward passes. Even at batch-size=1 with n_train=14,000, the kNN loop over ~1,000 test rows took >13 minutes and didn't finish.

### What changed

- Dataset: eeg-eye-state → diabetes (OpenML, 768 rows, 8 numerical features, binary, ~65/35 balance)
- Default n_train: 10,000 → 500 (leaves 268 test rows — ~268 forward passes for kNN, should finish in ~2 min)
- Removed `--batch-size` flag and all batching logic — loops always run per-query (batch-size=1), which is the correct reproduction of LoCalPFN's local context approach
- Added `orient_probs` per query to handle the class flip issue before concatenating
- Added progress prints every 50 queries (e.g., `kNN 50/268`)
- Kept AUC metric with `max(raw, 1 - raw)`, `orient_probs`, mixed context strategy, and all CLI flags except `--batch-size`
