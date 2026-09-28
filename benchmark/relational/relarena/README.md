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

Pass a dataset name or task name to run only that RelBench dataset/task:

```bash
python -m benchmark.relational.relarena.kumo_relational \
  --datasets rel-f1 \
  --tasks driver-top3
```
