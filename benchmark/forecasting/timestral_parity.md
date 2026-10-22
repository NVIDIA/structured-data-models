# Timestral 5M parity

Run from the repository root with the upstream reference installed:

```bash
pip install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
python -m benchmark.forecasting.timestral_parity --device cpu --repeats 20
python -m benchmark.forecasting.timestral_parity --device cuda --repeats 20
```

The script verifies the installed source revision and uses `mistralai/Timestral-5M` revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`. Downloading weights is subject to their [Modified MIT license](../../sdm/models/timestral/WEIGHTS_LICENSE).

It compares upstream `TimestralForecaster.forecast` with SDM's public `Timestral.forward`, using default sign averaging, nonnegative clipping, and normalization refinement. It also compares SDM's fitted `predict` with its one-shot output. Deterministic inputs cover a short patch, univariate and multivariate targets, independent batches, missing and entirely missing targets, constant and positive targets, numerical and binary covariates, categorical past and known-future covariates, unseen and missing categories, missing target history with categorical covariates, a 1,030-step forecast, and truncation of a 32,775-step context.

Recorded on 2026-10-18 with PyTorch `2.14.0+cu130`, a Cortex-X925 CPU using two PyTorch threads, and an NVIDIA GB10 GPU:

| Comparison                    | Result                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| CPU, horizons up to 65 steps  | Maximum absolute error `3.25e-5`                                |
| CPU, 1,030-step horizon       | Maximum absolute error `0.0801`; maximum scaled error `1.82e-5` |
| CPU, all 15 cases             | Pass `atol=rtol=2e-5`                                           |
| CUDA, horizons up to 65 steps | Maximum absolute error `1.96e-5`                                |
| CUDA, 1,030-step horizon      | Maximum absolute error `0.0547`; maximum scaled error `1.66e-5` |
| CUDA, all 15 cases            | Pass `atol=rtol=1e-4`                                           |

Scaled error is `abs(actual - reference) / (1 + abs(reference))`. The long-horizon example produces large tail quantiles; its absolute error should be interpreted alongside its scaled error. Timestral uses SDM's shared `TransformerBlock`, including `Attention`, `RotaryEmbedding` with explicit positions, and input-dependent sigmoid output gating. Time and variate attention are separate residual blocks, with one normalized `SwiGLU` MLP after the final attention step of each upstream layer. A single full mask replaces upstream's split context/forecast attention calls. Checkpoint loading permutes query/key channels to match SDM's partial-rotation layout while leaving values unchanged. This preserves the attention computation mathematically; reordered floating-point reductions introduce small differences on CPU and CUDA.

Only 5M was checked against pretrained weights. The 28M, 123M, and 209M variants use pinned configurations and checkpoint loading but have not been parity-tested. Parity is scoped to float32; lower precision is unvalidated.

The integration needs no upstream package at runtime. Optional pretrained pytest comparisons skip when upstream or cached 5M weights are unavailable. The ordinary public-contract and neural-block tests use random weights and run offline. SDM represents categorical values with category indices; for multivariate targets those indices become the raw numerical covariates, matching upstream when supplied the same codes.

The default recipe uses `AlignCategories(min_frequency=0, unknown="append")` to align independently tensorized vocabularies. Context codes and unused dictionary entries are retained; new query categories are appended to the returned vocabulary without changing fitted state. Numeric vocabularies use tensor lookups; string vocabularies use SDM's batched relational join, with cuDF on CUDA when installed and the existing CPU fallback otherwise. Forecast tensor execution and target encoding stay on the input device.

## Runtime versus upstream

The [runtime benchmark](timestral_runtime.py) checks prediction parity before timing each workload. Reproduce it with:

```bash
python -m benchmark.forecasting.timestral_runtime --device cpu --output benchmark/forecasting/timestral_runtime_cpu.json
python -m benchmark.forecasting.timestral_runtime --device cuda --output benchmark/forecasting/timestral_runtime_cuda.json
```

Measured before moving category alignment into the recipe, on the same CPU/GPU and PyTorch version described above, with two CPU threads, float32, eager execution, one series per call, and the default forecast options. Values are median milliseconds across 30 calls after five warmups. Each pair alternates execution order; CUDA is synchronized before and after each individual timed call. These are public API latencies, including preprocessing and output construction, with inputs already on the target device. Loading, initial table construction, and warmup are excluded. CPU and CUDA benchmarks ran sequentially.

| Workload               | Context / horizon | CPU SDM | CPU upstream | CUDA SDM | CUDA upstream |
| ---------------------- | ----------------- | ------- | ------------ | -------- | ------------- |
| Univariate             | 128 / 96          | 79.03   | 68.79        | 9.25     | 9.09          |
| Univariate             | 1,024 / 96        | 82.16   | 71.09        | 10.03    | 9.84          |
| Three targets          | 1,024 / 96        | 85.71   | 73.62        | 10.56    | 10.32         |
| Numerical covariates   | 1,024 / 96        | 92.55   | 80.50        | 12.06    | 11.47         |
| Categorical covariates | 1,024 / 96        | 96.02   | 82.22        | 14.92    | 11.90         |
| Long horizon           | 79 / 1,030        | 163.30  | 143.34       | 23.54    | 23.65         |

SDM takes roughly 14–17% longer on CPU in these workloads. On CUDA, most workloads are within 5% of upstream; the categorical case takes roughly 25% longer. The categorical comparison includes SDM's dictionary alignment, while upstream accepts numeric category codes. Neither implementation is compiled or uses lower precision. No attribution of the remaining runtime difference to individual operations has been profiled.

The [CPU report](timestral_runtime_cpu.json) and [CUDA report](timestral_runtime_cuda.json) include all individual samples, quartiles, revisions, and benchmark settings.

## Shared transformer refactor

Remeasured on 2026-10-18 after replacing the private attention and residual blocks with `sdm.nn.TransformerBlock`. The same benchmark settings apply, with 20 timed calls after five warmups (`--repeats 20`). The CUDA baseline was measured immediately before this refactor; CPU and CUDA runs were sequential. All values below are median milliseconds.

| Workload                            | CPU SDM | CPU upstream | CUDA SDM before | CUDA SDM after | CUDA upstream after |
| ----------------------------------- | ------- | ------------ | --------------- | -------------- | ------------------- |
| Univariate (128 / 96)               | 80.70   | 68.87        | 9.91            | 9.43           | 9.12                |
| Univariate (1,024 / 96)             | 84.05   | 71.73        | 10.72           | 10.33          | 9.84                |
| Three targets (1,024 / 96)          | 85.74   | 74.24        | 11.37           | 10.84          | 10.31               |
| Numerical covariates (1,024 / 96)   | 92.79   | 80.73        | 12.88           | 12.11          | 11.41               |
| Categorical covariates (1,024 / 96) | 94.89   | 82.06        | 14.39           | 13.47          | 11.86               |
| Long horizon (79 / 1,030)           | 163.31  | 139.79       | 24.77           | 23.81          | 23.58               |

The full-mask implementation showed no latency regression in these workloads. Before/after timings are separate runs, and upstream CUDA timings also decreased, so these measurements do not isolate a speedup caused by the refactor. The public 5M parity checks still pass on both devices at the tolerances above. Larger batches, longer covariate timelines, and lower precision remain unbenchmarked.
