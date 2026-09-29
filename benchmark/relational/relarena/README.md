# Relational Models on RelArena-α

This benchmark evaluates `structured-data-models` on the [RelArena-α](https://github.com/PriorLabs/relarena) benchmark.

## Setup

Run the commands below from the repository root:

```bash
pip install relarena==0.1.0
```

## Run

- **`KumoRelational`:**

  ```bash
  python -m benchmark.relational.relarena.kumo_relational
  ```

Pass a dataset name or task name to run only that [RelBenchV1](https://star-project.stanford.edu/relbench) dataset/task:

```bash
python -m benchmark.relational.relarena.kumo_relational \
  --datasets rel-f1 \
  --tasks driver-top3
```

## Results

- **`KumoRelational`** uses a maximum of 20K context examples up to two hops, excluding text columns.
  It achieves an Elo of 1832.2 (rank 3/12) on the models-only leaderboard, and an Elo of 1827.8 (rank 4/14) on the combined models-and-systems leaderboard (as of 2026-09-29).

| Classification Task (AUROC ⬆️) | `kumo-relational` | Regression Task (MAE ⬇️)    | `kumo-relational` |
| :----------------------------- | ----------------: | :-------------------------- | ----------------: |
| `rel-amazon/item-churn`        |            0.8252 | `rel-amazon/item-ltv`       |           44.6104 |
| `rel-amazon/user-churn`        |            0.6958 | `rel-amazon/user-ltv`       |           14.1884 |
| `rel-avito/user-clicks`        |            0.6800 | `rel-avito/ad-ctr`          |            0.0313 |
| `rel-avito/user-visits`        |            0.6701 | `rel-event/user-attendance` |            0.2412 |
| `rel-event/user-ignore`        |            0.8593 | `rel-f1/driver-position`    |            3.8294 |
| `rel-event/user-repeat`        |            0.7806 | `rel-hm/item-sales`         |            0.0400 |
| `rel-f1/driver-dnf`            |            0.7331 | `rel-stack/post-votes`      |            0.0638 |
| `rel-f1/driver-top3`           |            0.8258 | `rel-trial/site-success`    |            0.3989 |
| `rel-hm/user-churn`            |            0.6979 | `rel-trial/study-adverse`   |           41.1757 |
| `rel-stack/user-badge`         |            0.8647 |                             |                   |
| `rel-stack/user-engagement`    |            0.9007 |                             |                   |
| `rel-trial/study-outcome`      |            0.7095 |                             |                   |

## Runtime

Per-task runtimes aggregated over the 21 entity-level [RelBenchV1](https://star-project.stanford.edu/relbench) tasks from the [RelArena-α report](https://arxiv.org/abs/2608.16319), excluding warm-up caching:

| Model                          | #Configs | Cache |    Mean |     Min |     p25 |     p50 |     p75 |      Max |
| ------------------------------ | -------: | :---: | ------: | ------: | ------: | ------: | ------: | -------: |
| `kumo-relational`              |        3 |  ❌   |   1 min | 0.1 min | 0.5 min | 0.6 min |   1 min |    3 min |
| `rdblearn`                     |        6 |  ✅   |  11 min | 0.5 min |   2 min |   7 min |  11 min |   50 min |
| `tabpfn-rel-local-2026-09-28`  |        1 |  ✅   |  11 min | 0.4 min | 1.0 min |   5 min |  20 min |   49 min |
| `tabpfn-rel-local`             |        3 |  ✅   |  12 min | 0.6 min | 0.8 min |   4 min |   6 min |   85 min |
| `lightgbm`                     |       30 |  ❌   |  14 min | 0.1 min | 0.2 min |   2 min |  28 min |   53 min |
| `tabpfn-rel-client-2026-09-28` |        1 |  ✅   |  16 min | 0.8 min |   2 min |  13 min |  26 min |   78 min |
| `graphsage`                    |        4 |  ❌   |  47 min |   2 min |   9 min |  35 min |  83 min |  141 min |
| `relgnn-es`                    |       10 |  ✅   |  73 min |   1 min |   6 min |  28 min |  99 min |  336 min |
| `tabpfn-rel-client-2026-08-15` |        3 |  ✅   |  76 min |  18 min |  72 min |  88 min |  94 min |  108 min |
| `rt-plurel`                    |        – |  ✅   | 351 min | 109 min | 129 min | 231 min | 516 min |  979 min |
| `relgt`                        |        9 |  ✅   | 511 min |  40 min | 175 min | 273 min | 466 min | 2442 min |
