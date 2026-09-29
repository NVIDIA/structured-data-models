# Relational Models on RelArena

This benchmark evaluates `structured-data-models` on the [RelArena](https://github.com/PriorLabs/relarena) benchmark.

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

## Runtime

Per-task runtimes aggregated over the 21 entity-level [RelBenchV1](https://star-project.stanford.edu/relbench) tasks from the [RelArena-α report](https://arxiv.org/abs/2608.16319), excluding warm-up caching.

| Model | #Trials | Cache | Mean | Min | p25 | p50 | p75 | Max |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `kumo-relational` | 3 | ✗ | 1.03 min | 0.13 min | 0.45 min | 0.63 min | 1.24 min | 3.14 min |
| `rdblearn` | 6 | ✓ | 11 min | 0.5 min | 2 min | 7 min | 11 min | 50 min |
| `tabpfn-rel-local` | 3 | ✓ | 12 min | 0.6 min | 0.8 min | 4 min | 6 min | 85 min |
| `lightgbm` | 30 | ✗ | 14 min | 0.1 min | 0.2 min | 2 min | 28 min | 53 min |
| `graphsage` | 4 | ✗ | 47 min | 2 min | 9 min | 35 min | 83 min | 141 min |
| `relgnn-es` | 10 | ✓ | 73 min | 1 min | 6 min | 28 min | 99 min | 336 min |
| `tabpfn-rel-client` | 3 | ✓ | 76 min | 18 min | 72 min | 88 min | 94 min | 108 min |
| `rt-plurel` | – | ✓ | 351 min | 109 min | 129 min | 231 min | 516 min | 979 min |
| `relgt` | 9 | ✓ | 511 min | 40 min | 175 min | 273 min | 466 min | 2442 min |
