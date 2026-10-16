# Kumo Relational on RelBench

This example evaluates `KumoRelational` on [RelBench](https://relbench.stanford.edu).

## Run

```bash
python main.py --dataset=rel-f1 --task=driver-dnf
```

## Options

- `--num_estimators`: Number of estimators (default: `1`).
- `--context_size`: Maximum number of context rows (default: `10_000`).
- `--context_mode`: Context selection strategy, either `latest` or `random` (default: `latest`).
- `--num_neighbors`: The number of neighbors to sample for each hop (default: `[16, 16]`).
- `--text`: Include text embeddings from the task entity table.
- `--text_dim`: Final text embedding dimension after PCA (default: `64`).

## Results

Optimal hyperparameters found via the [RelArena-α](../../../benchmark/relational/relarena) evaluation protocol, using 8 estimators:

| Dataset      | Task              | Val AUROC | Test AUROC |  Size | Mode     |  Neighbors | Text Model  | Text Dim |
| :----------- | :---------------- | --------: | ---------: | ----: | :------- | ---------: | :---------- | -------: |
| `rel-amazon` | `user-churn`      |    0.6935 |     0.6947 | `20K` | `latest` |   `[8, 8]` |             |          |
| `rel-amazon` | `item-churn`      |    0.8215 |     0.8254 | `20K` | `latest` | `[32, 32]` |             |          |
| `rel-avito`  | `user-visits`     |    0.6956 |     0.6711 | `20K` | `latest` | `[32, 32]` |             |          |
| `rel-avito`  | `user-clicks`     |    0.6369 |     0.6824 | `20K` | `latest` | `[32, 32]` |             |          |
| `rel-event`  | `user-repeat`     |    0.7330 |     0.8090 |  `1K` | `random` |   `[4, 4]` |             |          |
| `rel-event`  | `user-ignore`     |    0.8467 |     0.8380 |  `2K` | `latest` | `[16, 16]` |             |          |
| `rel-f1`     | `driver-dnf`      |    0.7937 |     0.8418 |  `2K` | `latest` |   `[4, 4]` |             |          |
| `rel-f1`     | `driver-top3`     |    0.8816 |     0.9077 |  `1K` | `random` |   `[4, 4]` |             |          |
| `rel-hm`     | `user-churn`      |    0.7036 |     0.6992 | `20K` | `latest` |   `[1, 1]` |             |          |
| `rel-stack`  | `user-engagement` |    0.9001 |     0.9019 | `20K` | `random` |   `[8, 8]` | `mini`[^1]  |     `32` |
| `rel-stack`  | `user-badge`      |    0.8932 |     0.8809 | `20K` | `random` |   `[8, 8]` | `mpnet`[^2] |     `64` |
| `rel-trial`  | `study-outcome`   |    0.6636 |     0.7393 | `20K` | `latest` |   `[8, 8]` | `mini`[^1]  |     `32` |

| Dataset      | Task              | Val nMAE | Test nMAE |  Size | Mode     |  Neighbors | Text Model | Text Dim |
| :----------- | :---------------- | -------: | --------: | ----: | :------- | ---------: | :--------- | -------: |
| `rel-amazon` | `user-ltv`        |    21.60 |     25.01 | `20K` | `random` | `[64, 64]` |            |          |
| `rel-amazon` | `item-ltv`        |     7.38 |      7.56 | `20K` | `random` |       `[]` |            |          |
| `rel-avito`  | `ad-ctr`          |    32.40 |     32.95 |  `2K` | `latest` | `[64, 64]` |            |          |
| `rel-event`  | `user-attendance` |    31.25 |     31.70 | `20K` | `random` | `[64, 64]` |            |          |
| `rel-f1`     | `driver-position` |    44.63 |     43.55 | `20K` | `latest` |   `[8, 8]` |            |          |
| `rel-hm`     | `item-sales`      |     9.81 |      8.00 | `20K` | `random` |       `[]` |            |          |
| `rel-stack`  | `post-votes`      |    11.36 |     12.49 | `20K` | `random` |       `[]` |            |          |
| `rel-trial`  | `study-adverse`   |    10.21 |      9.42 | `20K` | `latest` |       `[]` | `mini`[^1] |     `32` |
| `rel-trial`  | `site-success`    |    87.90 |     84.31 | `10K` | `latest` |   `[8, 8]` | `mini`[^1] |     `32` |

[^1]: `sentence-transformers/all-MiniLM-L6-v2`

[^2]: `sentence-transformers/all-mpnet-base-v2`
