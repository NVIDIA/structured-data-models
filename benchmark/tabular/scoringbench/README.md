# Tabular Probabilistic Regression on ScoringBench

This benchmark evaluates the regression distributions produced by `structured-data-models` on the univariate [ScoringBench](https://github.com/jonaslandsgesell/ScoringBench) benchmark.

KumoTabular and TabICLv2 expose regression quantiles and therefore support ScoringBench's distributional metrics. TabFM is not included because its current regression head produces a point prediction rather than a predictive distribution.

## Setup

Clone the tested ScoringBench revision and install its benchmark dependencies separately from SDM's core dependencies:

```bash
git clone --recurse-submodules https://github.com/jonaslandsgesell/ScoringBench.git
git -C ScoringBench checkout cc0f4bbcafb1df80797324c00956950bf1ac1d66

pip install structured-data-models
pip install -r ScoringBench/requirements.txt
```

ScoringBench is not currently an installable Python project, so its checkout is supplied to the commands with `--scoringbench-path`.

## Protocol

The integration delegates dataset loading, cross-validation, subsampling, feature imputation, metrics, result persistence, and ranking to the pinned ScoringBench source. Its default univariate protocol uses five folds, one repeat, and at most 3,000 rows per dataset.

ScoringBench imputes missing features within each fold before invoking a model: numerical columns use the training median and categorical columns use the training mode. Consequently, SDM does not receive native missing feature values in this benchmark.

SDM produces quantiles at fixed thousandth probability levels. The adapter linearly interpolates them at ScoringBench's exact 200 levels from 0.005 through 0.995 and passes those predictions to its standard quantile-to-distribution conversion.

## Run

Run KumoTabular on every validated dataset:

```bash
python -m benchmark.tabular.scoringbench.main \
  --scoringbench-path /path/to/ScoringBench \
  --model kumo-tabular
```

Run TabICLv2:

```bash
python -m benchmark.tabular.scoringbench.main \
  --scoringbench-path /path/to/ScoringBench \
  --model tabiclv2
```

Run a named dataset or its validated zero-based index:

```bash
python -m benchmark.tabular.scoringbench.main \
  --scoringbench-path /path/to/ScoringBench \
  --model kumo-tabular \
  --dataset cpu_act

python -m benchmark.tabular.scoringbench.main \
  --scoringbench-path /path/to/ScoringBench \
  --model kumo-tabular \
  --dataset-index 0
```

Use `--lite` for two folds. `--sample-size`, `--n-repeats-cv`, `--seed`, `--batch-size`, and `--output-dir` override the corresponding execution settings.

Completed model/dataset/fold results are reused automatically. By default, raw Parquet results are written under `benchmark/tabular/scoringbench_out/univariate/raw/`.

## Evaluate

Initialize ScoringBench's official result submodule if it was not cloned recursively:

```bash
git -C /path/to/ScoringBench submodule update --init output
```

Then aggregate the SDM runs and compare them with the official results using ScoringBench's ranking implementation:

```bash
python -m benchmark.tabular.scoringbench.evaluate \
  --scoringbench-path /path/to/ScoringBench
```

The evaluator leaves the ScoringBench checkout unchanged. It writes combined results and generated leaderboard artifacts under `benchmark/tabular/evals/scoringbench/`.
