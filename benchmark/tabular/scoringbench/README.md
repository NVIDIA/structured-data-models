# Tabular Probabilistic Regression on ScoringBench

This benchmark evaluates `structured-data-models` on the univariate [ScoringBench](https://scoringbench.com) benchmark.

ScoringBench owns the evaluation protocol, including five-fold cross-validation and fold-local feature imputation. Numerical columns are imputed with the training median and categorical columns with the training mode before they reach the model.

## Setup

Run the commands below from the repository root:

```bash
pip install .

git clone https://github.com/jonaslandsgesell/ScoringBench.git
git -C ScoringBench checkout cc0f4bbcafb1df80797324c00956950bf1ac1d66
git -C ScoringBench submodule update --init
pip install -r ScoringBench/requirements.txt
```

## Run

- **TabICLv2:**

  ```bash
  python -m benchmark.tabular.scoringbench.main \
    --scoringbench-path /path/to/ScoringBench \
    --model tabiclv2
  ```

- **KumoTabular:**

  ```bash
  python -m benchmark.tabular.scoringbench.main \
    --scoringbench-path /path/to/ScoringBench \
    --model kumo-tabular
  ```

- **KumoTabular (small):**

  ```bash
  python -m benchmark.tabular.scoringbench.main \
    --scoringbench-path /path/to/ScoringBench \
    --model kumo-small
  ```

Pass `--dataset cpu_act` to run one dataset, `--dataset-index 0` to select by validated index, or `--lite` to use two folds.
Results are written under `benchmark/tabular/scoringbench_out/univariate/raw/` and completed folds are reused automatically.

## Fine-tuning

Add `--finetune` to full fine-tune every parameter of the model on each dataset's training split before evaluating (see `benchmark/tabular/finetune.py`). `--finetune` alone is enough to get benchmark-quality fine-tuning; tune it further with `--finetune-epochs`, `--finetune-iters-per-epoch`, `--finetune-lr`, `--finetune-train-size`, `--finetune-context-frac`, and `--finetune-val-frac`:

```bash
python -m benchmark.tabular.scoringbench.main \
  --scoringbench-path /path/to/ScoringBench \
  --model kumo-small \
  --finetune \
  --finetune-epochs 75 \
  --finetune-lr 1e-6
```

## Evaluate

Aggregate the SDM runs and compare them with the official results:

```bash
python -m benchmark.tabular.scoringbench.evaluate \
  --scoringbench-path /path/to/ScoringBench
```

Combined results and leaderboard artifacts are written under `benchmark/tabular/evals/scoringbench/`.
