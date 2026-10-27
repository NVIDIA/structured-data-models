# Time-Series Forecasting Benchmarks

This directory contains benchmarks of `structured-data-models` on [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval), [fev-bench](https://github.com/autogluon/fev), and [TIME](https://github.com/zqiao11/TIME).

All benchmarks share one model adapter (see `benchmark/forecasting/model.py`) that maps each forecasting task onto the `ICLModel` interface: time steps become rows, past targets become `y_context`, all covariates and timestamps of the context become `x_context`, and the forecast horizon becomes `x_query` with one row of known covariates and timestamps per future time step. Columns keep their semantic types, so each model decides which ones it consumes. Series in a batch are left-padded with missing values, and multivariate series are flattened for models without multi-target support.

______________________________________________________________________

> [!NOTE]
> Weights of `TimesFM3` are distributed under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TimesFM3` benchmark, which will download its weights noninteractively.

## GIFT-Eval

### Setup

Run the commands below from the repository root:

```bash
pip install . gluonts datasets python-dotenv
pip install --no-deps "git+https://github.com/SalesforceAIResearch/gift-eval.git@577a65cfa3acb812405c00d989230bf59f7f8a3d"

hf download Salesforce/GiftEval --repo-type dataset --local-dir /path/to/gift_eval
export GIFT_EVAL=/path/to/gift_eval
```

GIFT-Eval is not released on PyPI and is therefore pinned to a commit. It is installed without dependencies since it pins outdated `numpy` and `datasets` versions.

### Run

```bash
python -m benchmark.forecasting.gift_eval.main --model timesfm3
```

Pass `--dataset` to run only selected GIFT-Eval datasets, *e.g.*, `--dataset hospital --dataset ett1/H`.
Use `--batch_size` to set the number of series per model call and `--max_context_size` to truncate long contexts.
Results are written to `benchmark/forecasting/gift_eval_out/<model>/`, and already completed configurations are skipped.

### Evaluate

```bash
python -m benchmark.forecasting.gift_eval.evaluate
```

This compares all available SDM results against published GIFT-Eval results on the same configurations, normalized by a seasonal naive baseline and aggregated by geometric mean.

`TimesFM3` follows the published `TimesFM-3` evaluation with symmetric averaging over two estimators, contexts truncated to 15,360 time steps, and at most 32 variates per forward pass. Its results can still differ from the published ones:

- Published results batch series of different context lengths together, which shifts their positional encodings on datasets with varying context lengths.
- Published results clip forecasts of non-negative series at zero.

______________________________________________________________________

## fev-bench

### Setup

Run the command below from the repository root:

```bash
pip install . fev==0.10.0
```

Tasks are downloaded from Hugging Face on first use.

### Run

```bash
python -m benchmark.forecasting.fev_bench.main --model timesfm3
```

Pass `--task` to run only selected fev-bench tasks, *e.g.*, `--task hospital --task proenfo_gfc12`.
Results are written to `benchmark/forecasting/fev_bench_out/<model>/`, and already completed tasks are skipped.

### Evaluate

```bash
python -m benchmark.forecasting.fev_bench.evaluate
```

This compares all available SDM results against published fev-bench results on the same tasks with `fev.leaderboard`, using a seasonal naive baseline.

On fev-bench, `TimesFM3` also imputes missing values with the last observed value, converts dynamic covariates to numbers, and ignores static covariates. In addition to the differences listed for GIFT-Eval, published results hash string covariates with a randomly seeded instead of a fixed hash function.

______________________________________________________________________

## TIME

### Setup

Run the commands below from the repository root:

```bash
pip install . gluonts datasets python-dotenv scipy pyyaml

git clone https://github.com/zqiao11/TIME.git
git -C TIME checkout 3ca5c41d71c76f3c70632c0121142d325350ee32
pip install --no-deps -e TIME

hf download Real-TSF/TIME --repo-type dataset --local-dir /path/to/time
export TIME_DATASET=/path/to/time
```

TIME is not released on PyPI and is therefore pinned to a commit. It is installed without dependencies since it pins outdated `numpy`, `gluonts` and `datasets` versions.

### Run

```bash
python -m benchmark.forecasting.time_bench.main --model timesfm3
```

Pass `--dataset` to run only selected TIME datasets, *e.g.*, `--dataset Job_Claims/W --dataset CPHL/15T`.
Results are written to `benchmark/forecasting/time_bench_out/<model>/`, and already completed configurations are skipped.

### Evaluate

```bash
python -m benchmark.forecasting.time_bench.evaluate --time-path /path/to/TIME
```

This compares all available SDM results against published TIME results on the same configurations with TIME's official leaderboard script, normalized by a seasonal naive baseline.

The differences to published `TimesFM-3` results listed for GIFT-Eval apply to TIME as well.
