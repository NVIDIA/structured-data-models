# Time Series Forecasting Foundation Models on GIFT-Eval

This benchmark evaluates `structured-data-models` on the [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) benchmark.

## Setup

Run the commands below from the repository root:

```bash
pip install . pandas gluonts datasets python-dotenv
pip install --no-deps "git+https://github.com/SalesforceAIResearch/gift-eval.git@577a65cfa3acb812405c00d989230bf59f7f8a3d"
```

## Run

- **`TimesFM3`**:

  ```bash
  python -m benchmark.timeseries.gift_eval.main --model timesfm3
  ```

  `TimesFM3` follows official evaluation with symmetric averaging over two estimators, contexts truncated to 15,360 time steps and up to 32 variates per forward pass.

> [!NOTE]
> Weights of `TimesFM3` are distributed under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TimesFM3` benchmark, which will download its weights noninteractively.

Results will be written to `benchmark/timeseries/gift_eval_out/<model>/`, and already completed configurations are skipped.

Pass `--dataset` to run only selected GIFT-Eval datasets, *e.g.*, `--dataset hospital --dataset ett1/H`.

## Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.timeseries.gift_eval.evaluate
```
