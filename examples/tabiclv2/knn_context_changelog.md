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
