# Tabular Foundation Models on TALENT

This benchmark evaluates `structured-data-models` on the corrected 300-dataset [TALENT](https://github.com/LAMDA-Tabular/TALENT) benchmark.

## Setup

Run the commands below from the repository root:

```bash
pip install . \
  "TALENT @ git+https://github.com/LAMDA-Tabular/TALENT.git@08301d6"
```

Download and extract the datasets from the [official TALENT dataset page](https://huggingface.co/datasets/LAMDA-Tabular/TALENT).

## Run

- **`TabICLv2`:**

  ```bash
  python -m benchmark.tabular.talent.main --model tabiclv2 --dataset-path /path/to/talent/data
  ```

- **`KumoTabular`:**

  ```bash
  python -m benchmark.tabular.talent.main --model kumo-tabular --dataset-path /path/to/talent/data
  ```

- **`KumoTabular` (small):**

  ```bash
  python -m benchmark.tabular.talent.main --model kumo-small --dataset-path /path/to/talent/data
  ```

- **`TabFM`:**

  ```bash
  python -m benchmark.tabular.talent.main --model tabfm --dataset-path /path/to/talent/data
  ```

> [!NOTE]
> Weights of `TabFM` are distributed under the [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TabFM` benchmark, which will download its weights noninteractively.

Pass a dataset name to run only that TALENT dataset:

```bash
python -m benchmark.tabular.talent.main \
  --model kumo-tabular \
  --dataset-path /path/to/talent/data \
  --dataset Bank_Customer_Churn_Dataset
```

## Fine-tuning

Add `--finetune` to full fine-tune every parameter of the model on each dataset's training split before evaluating (see `benchmark/tabular/finetune.py`). `--finetune` alone is enough to get benchmark-quality fine-tuning; tune it further with `--finetune-epochs`, `--finetune-iters-per-epoch`, `--finetune-lr`, `--finetune-train-size`, `--finetune-context-frac`, and `--finetune-val-frac`:

```bash
python -m benchmark.tabular.talent.main \
  --model kumo-small \
  --dataset-path /path/to/talent/data \
  --finetune \
  --finetune-epochs 75 \
  --finetune-lr 1e-6
```

## Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.talent.evaluate
```

Install `matplotlib` and `scikit-posthocs`, then add `--plot-cd` to generate critical-difference diagrams as PNG files.
The evaluator downloads TALENT's published result tables by default; pass their local directory with `--official-results` to run offline.
