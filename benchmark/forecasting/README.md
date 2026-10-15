# Time-Series Forecasting Benchmarks

This directory contains benchmarks of `structured-data-models` on [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval).

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

`TimesFM3` follows the published `TimesFM-3` evaluation with symmetric averaging over two estimators and contexts truncated to 15,360 time steps. Its results can still differ from the published ones:

- Published results batch series of different context lengths together, which shifts their positional encodings on datasets with varying context lengths.
- Published results clip forecasts of non-negative series at zero.
- Published results forecast at most 32 variates jointly, subsampling covariates of `bizitobs_application` and `bizitobs_service`.
