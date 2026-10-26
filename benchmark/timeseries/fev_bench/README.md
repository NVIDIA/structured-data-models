# Time Series Forecasting Foundation Models on fev-bench

This benchmark evaluates `structured-data-models` on the [fev-bench](https://github.com/autogluon/fev) benchmark.

## Setup

Run the command below from the repository root:

```bash
pip install . fev==0.10.0
```

## Run

- **`TimesFM3`**:

  ```bash
  python -m benchmark.timeseries.fev_bench.main --model timesfm3
  ```

  `TimesFM3` follows official evaluation with symmetric averaging over two estimators, contexts truncated to 15,360 time steps and up to 32 variates per forward pass.

> [!NOTE]
> Weights of `TimesFM3` are distributed under the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TimesFM3` benchmark, which will download its weights noninteractively.

Results will be written to `benchmark/timeseries/fev_bench_out/<model>/`, and already completed configurations are skipped.

Pass `--task` to run only selected fev-bench tasks, *e.g.*, `--task hospital --task proenfo_gfc12`.
Pass `--mini` to run the 20-task fev-bench-mini subset:

## Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.timeseries.fev_bench.evaluate
```
