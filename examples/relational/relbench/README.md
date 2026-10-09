# KumoRelational on RelBench

This example evaluates `KumoRelational` on the
[RelBench](https://relbench.stanford.edu/) benchmark.

## Run

```bash
python rel_bench.py --dataset=rel-amazon --task=user-churn
python rel_bench.py --dataset=rel-stack --task=user-engagement --text
```

Use `--text` to encode text features from the task entity table. The example
uses a maximum context size of 10K rows by default; pass `--context_size` to
change it. Contexts use the latest rows by default; pass
`--context_mode=random` to sample from all available rows.

## Results

Classification results use ROC-AUC (higher is better).

`mini` is `sentence-transformers/all-MiniLM-L6-v2`; `mpnet` is
`sentence-transformers/all-mpnet-base-v2`.

| Dataset      | Task              |    Val |   Test | Text model | Text dim | Estimators | Context size | Context mode | Neighbors |
| :----------- | :---------------- | -----: | -----: | :--------- | -------: | ---------: | -----------: | :----------- | :-------- |
| `rel-amazon` | `user-churn`      | 0.6935 | 0.6947 |            |          |          8 |          20K | latest       | [8, 8]    |
| `rel-amazon` | `item-churn`      | 0.8215 | 0.8254 |            |          |          8 |          20K | latest       | [32, 32]  |
| `rel-avito`  | `user-visits`     | 0.6956 | 0.6711 |            |          |          8 |          20K | latest       | [32, 32]  |
| `rel-avito`  | `user-clicks`     | 0.6369 | 0.6824 |            |          |          8 |          20K | latest       | [32, 32]  |
| `rel-event`  | `user-repeat`     | 0.7330 | 0.8090 |            |          |          8 |           1K | random       | [4, 4]    |
| `rel-event`  | `user-ignore`     | 0.8467 | 0.8380 |            |          |          8 |           2K | latest       | [16, 16]  |
| `rel-f1`     | `driver-dnf`      | 0.7937 | 0.8418 |            |          |          8 |           2K | latest       | [4, 4]    |
| `rel-f1`     | `driver-top3`     | 0.8816 | 0.9077 |            |          |          8 |           1K | random       | [4, 4]    |
| `rel-hm`     | `user-churn`      | 0.7036 | 0.6992 |            |          |          8 |          20K | latest       | [1, 1]    |
| `rel-stack`  | `user-engagement` | 0.9001 | 0.9019 | mini       |       32 |          8 |          20K | random       | [8, 8]    |
| `rel-stack`  | `user-badge`      | 0.8932 | 0.8809 | mpnet      |       64 |          8 |          20K | random       | [8, 8]    |
| `rel-trial`  | `study-outcome`   | 0.6636 | 0.7393 | mini       |       32 |          8 |          20K | latest       | [8, 8]    |

Regression results use normalized mean absolute error (nMAE; lower is better).

| Dataset      | Task              |    Val |   Test | Text model | Text dim | Estimators | Context size | Context mode | Neighbors |
| :----------- | :---------------- | -----: | -----: | :--------- | -------: | ---------: | -----------: | :----------- | :-------- |
| `rel-amazon` | `user-ltv`        | 0.2160 | 0.2501 |            |          |          8 |          20K | random       | [64, 64]  |
| `rel-amazon` | `item-ltv`        | 0.0738 | 0.0756 |            |          |          8 |          20K | random       | [0, 0]    |
| `rel-avito`  | `ad-ctr`          | 0.3240 | 0.3295 |            |          |          8 |           2K | latest       | [64, 64]  |
| `rel-event`  | `user-attendance` | 0.3125 | 0.3170 |            |          |          8 |          20K | random       | [64, 64]  |
| `rel-f1`     | `driver-position` | 0.4463 | 0.4355 |            |          |          8 |          20K | latest       | [8, 8]    |
| `rel-hm`     | `item-sales`      | 0.0981 | 0.0800 |            |          |          8 |          20K | random       | [0, 0]    |
| `rel-stack`  | `post-votes`      | 0.1136 | 0.1249 |            |          |          8 |          20K | random       | [0, 0]    |
| `rel-trial`  | `study-adverse`   | 0.1021 | 0.0942 | mini       |       32 |          8 |          20K | latest       | [0, 0]    |
| `rel-trial`  | `site-success`    | 0.8790 | 0.8431 | mini       |       32 |          8 |          10K | latest       | [8, 8]    |
