# Time Series Forecasting Foundation Models on TIME

This benchmark evaluates `structured-data-models` on the [TIME](https://github.com/zqiao11/TIME) benchmark.

## Setup

Run the commands below from the repository root:

```bash
pip install . gluonts datasets python-dotenv scipy pyyaml
pip install --no-deps "git+https://github.com/zqiao11/TIME.git@3ca5c41d71c76f3c70632c0121142d325350ee32"
```

## Run

- **`TimesFM3`**:

  ```bash
  python -m benchmark.timeseries.time_bench.main --model timesfm3
  ```

  `TimesFM3` follows official evaluation with symmetric averaging over two estimators, contexts truncated to 15,360 time steps and up to 32 variates per forward pass.
  Unlike the official `float32` evaluation, we use `bfloat16` mixed precision for improved efficiency with negligible differences in aggregate benchmark accuracy.

> [!NOTE]
> Weights of `TimesFM3` are distributed under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TimesFM3` benchmark, which will download its weights noninteractively.

Results are written to `benchmark/timeseries/time_bench_out/<model>/`, and already completed configurations are skipped.

Pass `--dataset` to run only selected TIME datasets, *e.g.*, `--dataset Job_Claims/W --dataset CPHL/15T`.

## Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.timeseries.time_bench.evaluate
```
