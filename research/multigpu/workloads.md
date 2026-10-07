# Real workloads and model checkpoints

Prepared for SDM commit `842c408fe` on 2026-10-08. Artifact root is `/Users/ardrianw/repositories/.kumo-multigpu-20261008`; portable relative directories are `data/` and `models/`.

| Model family | Dataset/task | TRAIN | VAL | Features / structure | Planned contexts | Primary quality |
|---|---|---:|---:|---|---|---|
| KumoTabular | Covertype | 464,809 | 116,203 | 54 numeric inputs; seven classes, labels 1–7 | 1,024 / 4,096 / 16,384 / 32,768 / 65,536 | Accuracy, multiclass log loss, macro F1 |
| KumoTabular | California housing | 16,512 | 4,128 | Eight numeric inputs; target units $100,000 | 1,024 / 4,096 / 16,384 | MAE, RMSE, R² |
| KumoRelational | rel-hm / user-churn | 3,832,692 | 76,556 | Native three-table relational database; exact `[16,16]` temporal neighborhood sampling | 1,024 / 4,096 / 16,384 / 32,768 / 65,536 | AUROC, average precision, log loss |
| KumoRelational | rel-f1 / driver-position | 7,453 | 499 | Native nine-table relational database; exact `[16,16]` temporal neighborhood sampling | 1,024 / 4,096 | MAE, RMSE, R² |

These are intended experiment sizes, not a claim that every model/algorithm/GPU combination fits. Record OOMs at the requested size rather than silently changing workload. Use 256 validation queries for smoke runs, then 2,048 / 4,096 / 16,384 where available. For rel-f1, use all 499 VAL rows after the smoke run. Increasing context uses a nested prefix of the same fixed TRAIN order. Increasing query count uses a nested prefix of the same VAL order. All algorithm and GPU-count arms must use identical rows, model checkpoint, precision, preprocessing and ensemble member plan.

## Tabular interface

Each `data/{covertype,california_housing}` directory contains `x_train.npy`, `y_train.npy`, `x_val.npy`, `y_val.npy`, `train_ids.npy`, `val_ids.npy`, `manifest.json`, and source description. Load with `np.load(..., allow_pickle=False)`; features are float32. Covertype labels are int64, California labels float32. Select rows by taking the first N stored array rows, since the arrays already reflect the shared split/permutation. The IDs are source-row identities for auditing, not a second index operation into the prepared arrays.

The underlying datasets do not ship an official held-out TEST split. We define a reproducible 80/20 TRAIN/VAL split using `train_test_split(random_state=20261008)` and class stratification for Covertype. TRAIN ordering uses `np.random.default_rng(20261008)` and VAL ordering uses seed `20261009`. This is an experimental validation split, not a canonical leaderboard split. There is no fitted preprocessing in preparation; fit the SDM recipe exclusively on the selected TRAIN context. VAL labels are used only after predictions to calculate quality. Covertype outputs must be aligned to labels 1–7 before log loss or accuracy computation.

Reproduce with `python research/multigpu/prepare_workloads.py --root DATA_ROOT --dataset covertype` or `--dataset california_housing`.

## Native relational interface

Each `data/{rel-hm,rel-f1}` directory contains original `db/*.parquet`, `tasks/TASK/{train,val}.parquet`, `TASK-train-ids.npy`, `TASK-val-ids.npy`, and `manifest.json`. Parquet preserves RelBench primary keys, foreign keys and time-column metadata. Unlike the tabular arrays, the native task parquet remains in its original order: select rows with the first N saved split IDs. TRAIN IDs are a permutation from seed `20261008`; VAL IDs are original ordered row indices.

Read only these copied files. No TEST files were copied, read or used. Only TRAIN labels may enter context or fitted preprocessing. Drop the VAL target before sampling and prediction; use the target only in scoring. Configure the native SDM sampler with the task timestamp, exact two-hop `[16,16]` neighbors and temporal `last` strategy. Preserve all task/entity identity mappings. Do not construct target lags from VAL labels. The stock `examples/relational/rel_bench.py` evaluates TEST and constructs lag features from all splits, so it is an API reference rather than an appropriate benchmark entrypoint for this experiment.

Rel-hm uses the existing August TRAIN/VAL-only cache at `.benchmark-artifacts/rel-hm-user-churn-authority.pvq66o/raw-cache/rel-hm`. Rel-f1 uses the existing local RelBench cache at `~/Library/Caches/relbench/rel-f1`, copying only DB, TRAIN and VAL. Full provenance, file sizes, row counts, schema and SHA256 are in each manifest. Temporal filtering remains a sampler responsibility; future DB records must not be admitted by a query's task timestamp.

## Exact checkpoints

| Family | SDM requested tag | Fresh remote commit |
|---|---|---|
| nvidia/Kumo-Tabular | v1.0.1 | `3c3e10bbdb590ace29e7026847f92db3c603096d` |
| nvidia/Kumo-Relational | v1.0.1 | `2bd603d3d8f25f67a7aaa8579908595e20567e22` |

`prepare_models.py` downloads exact commits and creates a portable Hugging Face cache. Set `HF_HUB_CACHE=/absolute/artifact/root/models/hub` and `HF_HUB_OFFLINE=1` before importing SDM. This lets unmodified public model constructors find checkpoints by their existing `v1.0.1` revision. Copies include checkpoint hashes, licenses, model cards and available third-party notices; no authentication tokens are packaged.

The preexisting local KumoTabular `v1.0.1` reference pointed to `f794cb1e62482d62e8879de6c5285cb5f5e63442`, which returned HTTP 404 when queried by commit on 2026-10-08. Fresh tag resolution returned the commit above. We deliberately use the fresh exact revision for every arm. Do not mix old KumoRelational v2.1 checkpoint snapshots with current SDM architectures.

## Sources and licensing

- Covertype: [UCI dataset](https://archive.ics.uci.edu/dataset/31/covertype), Jock Blackard, CC BY 4.0. Downloaded through scikit-learn's source-checksummed dataset loader.
- California housing: [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), the 1990 California census housing dataset distributed through StatLib. The loader metadata does not specify a separate dataset distribution license; do not equate the scikit-learn code license with a dataset license.
- RelBench: [project](https://relbench.stanford.edu/) and local cached parquet. RelBench code is MIT; source data remains subject to its underlying dataset terms (H&M competition data and Formula 1 source data respectively).
- Kumo weights: [Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) and [Kumo-Relational](https://huggingface.co/nvidia/Kumo-Relational), OpenMDW 1.1 according to the current model cards and bundled licenses. Preserve upstream third-party notices.

This preparation establishes comparable data and checkpoint inputs. Performance and quality claims require completed runner measurements and independent comparison of saved predictions.

## Preparation validation

All eight exact checkpoints loaded successfully through the unmodified SDM public constructors on CPU at `842c408fe`, with `HF_HUB_OFFLINE=1` and the portable cache. This confirms checkpoint key/shape compatibility; it is not an inference or GPU performance test.

| Model | Classification parameters | Regression parameters |
|---|---:|---:|
| KumoTabular small | 27,458,266 | 28,466,231 |
| KumoTabular medium | 61,485,274 | 62,492,087 |
| KumoTabular large | 213,668,250 | 215,683,191 |
| KumoRelational | 29,915,010 | 30,908,383 |

Independent workload inspection confirmed zero overlapping tabular TRAIN/VAL source IDs, no duplicate TRAIN IDs, exact relational VAL ordering, and finite tabular features. All seven Covertype classes occur in the first 1,024 context rows, with counts `[367,506,59,11,22,22,37]` for classes 1–7. Covertype's raw total is 581,012 rows and California's is 20,640. Data generation scripts passed Ruff and formatting checks.
