# Relational Benchmarks

The [`relbench/`](relbench/) package benchmarks `sdm.models.KumoRelational` on RelBench tasks.

Run the general benchmark from the repository root with a dataset and task:

```bash
python -m benchmark.relational.relbench.main --dataset rel-f1 --task driver-position
```

Run all SALT autocomplete tasks with:

```bash
python -m benchmark.relational.relbench.salt
```
