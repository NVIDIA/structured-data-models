# Kumo Relational on RelBench

This example evaluates `KumoRelational` on [RelBench](https://relbench.stanford.edu).

## Run

```bash
python main.py --dataset=rel-f1 --task=driver-dnf
```

## Options

- `--num_estimators`: Number of estimators (default: `1`).
- `--context_size`: Maximum number of context rows (default: `10_000`).
- `--context_mode`: Context selection strategy, either `last` or `rnd` (default: `last`).
- `--num_neighbors`: The number of neighbors to sample for each hop (default: `[16, 16]`).
- `--num_lags`: The number of lag targets to use as additional input (default: `20`).
- `--text_dim`: Text embedding dimension after PCA for the entity table (default: `0`).

## Results

Optimal hyperparameters found via the [RelArena-α](../../../benchmark/relational/relarena) evaluation protocol, using 8 estimators and 20 lag targets:

| Dataset      | Task              | Val AUROC | Test AUROC |  Size | Mode   |  Neighbors | Text Dim | Text Model  |
| :----------- | :---------------- | --------: | ---------: | ----: | :----- | ---------: | -------: | :---------- |
| `rel-amazon` | `user-churn`      |     69.35 |      69.47 | `20K` | `last` |   `[8, 8]` |      `0` |             |
| `rel-amazon` | `item-churn`      |     82.15 |      82.54 | `20K` | `last` | `[32, 32]` |      `0` |             |
| `rel-avito`  | `user-visits`     |     69.56 |      67.11 | `20K` | `last` | `[32, 32]` |      `0` |             |
| `rel-avito`  | `user-clicks`     |     63.69 |      68.24 | `20K` | `last` | `[32, 32]` |      `0` |             |
| `rel-event`  | `user-repeat`     |     73.30 |      80.90 |  `1K` | `rnd`  |   `[4, 4]` |      `0` |             |
| `rel-event`  | `user-ignore`     |     84.67 |      83.80 |  `2K` | `last` | `[16, 16]` |      `0` |             |
| `rel-f1`     | `driver-dnf`      |     79.37 |      84.18 |  `2K` | `last` |   `[4, 4]` |      `0` |             |
| `rel-f1`     | `driver-top3`     |     88.16 |      90.77 |  `1K` | `rnd`  |   `[4, 4]` |      `0` |             |
| `rel-hm`     | `user-churn`      |     70.36 |      69.92 | `20K` | `last` |   `[1, 1]` |      `0` |             |
| `rel-stack`  | `user-engagement` |     90.01 |      90.19 | `20K` | `rnd`  |   `[8, 8]` |     `32` | `mini`[^1]  |
| `rel-stack`  | `user-badge`      |     89.32 |      88.09 | `20K` | `rnd`  |   `[8, 8]` |     `64` | `mpnet`[^2] |
| `rel-trial`  | `study-outcome`   |     66.36 |      73.93 | `20K` | `last` |   `[8, 8]` |     `32` | `mini`[^1]  |

| Dataset      | Task              | Val nMAE | Test nMAE |  Size | Mode   |  Neighbors | Text Dim | Text Model |
| :----------- | :---------------- | -------: | --------: | ----: | :----- | ---------: | -------: | :--------- |
| `rel-amazon` | `user-ltv`        |    21.60 |     25.01 | `20K` | `rnd`  | `[64, 64]` |      `0` |            |
| `rel-amazon` | `item-ltv`        |     7.38 |      7.56 | `20K` | `rnd`  |       `[]` |      `0` |            |
| `rel-avito`  | `ad-ctr`          |    32.40 |     32.95 |  `2K` | `last` | `[64, 64]` |      `0` |            |
| `rel-event`  | `user-attendance` |    31.25 |     31.70 | `20K` | `rnd`  | `[64, 64]` |      `0` |            |
| `rel-f1`     | `driver-position` |    44.63 |     43.55 | `20K` | `last` |   `[8, 8]` |      `0` |            |
| `rel-hm`     | `item-sales`      |     9.81 |      8.00 | `20K` | `rnd`  |       `[]` |      `0` |            |
| `rel-stack`  | `post-votes`      |    11.36 |     12.49 | `20K` | `rnd`  |       `[]` |      `0` |            |
| `rel-trial`  | `study-adverse`   |    10.21 |      9.42 | `20K` | `last` |       `[]` |     `32` | `mini`[^1] |
| `rel-trial`  | `site-success`    |    87.90 |     84.31 | `10K` | `last` |   `[8, 8]` |     `32` | `mini`[^1] |

[^1]: `sentence-transformers/all-MiniLM-L6-v2`

[^2]: `sentence-transformers/all-mpnet-base-v2`
