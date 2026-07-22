# TabICLv2 processing performance report

## Method

Benchmarks were rerun on 22 July 2026 on AWS Linux (`Linux-6.17.0-1017-aws-x86_64`) with Python 3.10.18, PyTorch 2.13.0+cu130, CUDA 13.0, and an NVIDIA L4. `origin/main` was merged into this PR before the rerun; the main commit included in the benchmarked state is `3d65acc556890d74db57eb23cdc6fcb228d0626f`, and the local merged worktree commit used for the rerun is `cc38e6558379da813d7edd1f56f002bb63337b10`.

The headline workload has 50,000 rows and 100 features, split into 40,000 context rows and 10,000 query rows. It contains 90 numerical and 10 categorical columns and exercises the combined-stress path:

- a constant numerical column;
- missing numerical values and missing categorical codes;
- 4,096 observed categorical values plus one query-only unseen value;
- a `1e12` query outlier for fixed `Clip(-100, 100)`;
- a value of `20` for sigma-based soft clipping;
- explicit Power and Quantile Recipe variants.

Each recipe measurement uses one warmup and five measured repetitions. The speed-of-light run uses five warmups and 20 measured repetitions. Dataset creation, fitted-state preparation, correctness checks, and output comparisons are outside timed regions. CUDA measurements synchronize before and after timed regions and record both host wall-clock time and CUDA-event time. Peak memory is incremental CUDA allocation above the prepared-operation baseline. CPU peak memory remains null to avoid perturbing short operations.

All 194 rerun SDM large-stress results have `correctness_status="pass"`. The pinned TabICLv2 reference artifacts remain historical and were not rerun for this main-state refresh.

## Main-state recipe headline

Times are median / p95 milliseconds. Speedup is SDM CPU median divided by SDM GPU median. Stage measurements are independent and are not additive.

| Recipe   | Task           | Stage                              |             SDM CPU |         SDM GPU | CPU/GPU speedup |
| -------- | -------------- | ---------------------------------- | ------------------: | --------------: | --------------: |
| Power    | Classification | Feature + target preprocessing     | 2315.757 / 2328.383 | 79.754 / 82.238 |          29.04× |
| Power    | Classification | Map model output to original space |       0.070 / 0.115 |   0.070 / 0.083 |           0.99× |
| Power    | Classification | Output transform                   |       0.480 / 0.503 |   0.232 / 0.259 |           2.07× |
| Power    | Classification | Total Recipe overhead              | 2169.644 / 2174.713 | 78.950 / 79.494 |          27.48× |
| Power    | Regression     | Feature + target preprocessing     | 2285.435 / 2290.452 | 73.069 / 73.829 |          31.28× |
| Power    | Regression     | Map model output to original space |     32.674 / 39.385 |   2.169 / 2.266 |          15.07× |
| Power    | Regression     | Output transform                   |     17.784 / 19.941 |   0.496 / 0.594 |          35.85× |
| Power    | Regression     | Total Recipe overhead              | 2260.380 / 2289.452 | 78.044 / 80.600 |          28.96× |
| Quantile | Classification | Feature + target preprocessing     | 1129.658 / 1177.476 | 48.767 / 53.022 |          23.16× |
| Quantile | Classification | Map model output to original space |       0.110 / 0.124 |   0.074 / 0.080 |           1.49× |
| Quantile | Classification | Output transform                   |       0.447 / 0.476 |   0.226 / 0.248 |           1.98× |
| Quantile | Classification | Total Recipe overhead              |   716.368 / 726.307 | 53.656 / 57.099 |          13.35× |
| Quantile | Regression     | Feature + target preprocessing     | 1164.758 / 1254.385 | 46.900 / 48.984 |          24.84× |
| Quantile | Regression     | Map model output to original space |     32.945 / 33.648 |  2.256 / 90.165 |          14.60× |
| Quantile | Regression     | Output transform                   |     17.898 / 18.658 |   0.493 / 0.504 |          36.31× |
| Quantile | Regression     | Total Recipe overhead              |   766.604 / 770.707 | 46.648 / 50.113 |          16.43× |

The main-state result changes the conclusion from the earlier report: Power preprocessing is no longer a multi-second GPU bottleneck. The full Power recipe now lands around 73-80 ms on this L4, and the full Quantile recipe lands around 47-54 ms. The remaining absolute bottlenecks are `Power.fit`, `CategoricalAlign`, and a few search/gather-heavy transform paths, not CPU fallback or transfers inside the timed Processor operations.

The largest observed incremental GPU allocation in the main-state large-stress run was 193,472,000 bytes (184.5 MiB) for `Power.inverse_transform` on the classification processor microbenchmark. Full-recipe peaks were lower: 143,646,720 bytes (137.0 MiB) for Quantile classification preprocessing and 141,116,928 bytes (134.6 MiB) for Power regression total overhead.

## Individual Processor results

Complete individual Processor operation matrix on the combined-stress workload. Current timings are the rerun large-stress GPU medians/p95s. The speed-of-light column is the fastest passing CUDA candidate from `processor_gpu_speed_of_light.json`; when the current implementation is fastest or the only passing candidate, it is listed as `current`.

| Processor / operation                |                         Current GPU large stress |   Speed of light | Candidate                       |
| ------------------------------------ | -----------------------------------------------: | ---------------: | ------------------------------- |
| `StandardScale.fit`                  |     cls: 0.139 / 0.143 ms; reg: 0.136 / 0.156 ms | 0.134 / 0.158 ms | `direct_reduce_state`           |
| `StandardScale.fit_transform`        |     cls: 0.248 / 0.534 ms; reg: 0.276 / 0.518 ms | 0.165 / 0.193 ms | `direct_reduce_affine`          |
| `StandardScale.transform`            |     cls: 0.160 / 0.173 ms; reg: 0.158 / 0.184 ms | 0.189 / 0.196 ms | `direct_affine`                 |
| `StandardScale.inverse_transform`    |     cls: 0.152 / 0.156 ms; reg: 0.147 / 0.438 ms | 0.183 / 0.192 ms | `direct_affine`                 |
| `Clip.transform`                     |     cls: 0.112 / 0.134 ms; reg: 0.104 / 0.138 ms | 0.070 / 0.115 ms | `direct_clamp`                  |
| `Power.fit`                          | cls: 32.207 / 32.461 ms; reg: 32.037 / 32.174 ms | 3.381 / 3.416 ms | `batched_newton_compiled`       |
| `Power.fit_transform`                | cls: 33.101 / 33.253 ms; reg: 32.833 / 32.970 ms | 3.411 / 3.497 ms | `batched_newton_compiled`       |
| `Power.transform`                    |     cls: 1.363 / 1.369 ms; reg: 1.369 / 1.374 ms | 0.376 / 0.394 ms | `vectorized_compiled`           |
| `Power.inverse_transform`            |     cls: 3.146 / 3.159 ms; reg: 3.137 / 3.144 ms | 0.278 / 0.308 ms | `vectorized_compiled`           |
| `Quantile.fit`                       |     cls: 0.602 / 0.657 ms; reg: 0.565 / 0.607 ms | 0.602 / 0.658 ms | `current`                       |
| `Quantile.fit_transform`             |     cls: 3.651 / 3.676 ms; reg: 3.736 / 3.779 ms | 3.709 / 4.560 ms | `current`                       |
| `Quantile.transform`                 |     cls: 4.060 / 4.198 ms; reg: 3.870 / 4.132 ms | 0.831 / 0.852 ms | `batched_searchsorted_compiled` |
| `Quantile.inverse_transform`         |     cls: 2.320 / 2.338 ms; reg: 2.376 / 2.493 ms | 0.221 / 0.227 ms | `batched_searchsorted_compiled` |
| `SigmaClip.fit`                      |     cls: 0.730 / 0.737 ms; reg: 0.717 / 0.722 ms | 0.726 / 0.738 ms | `direct_two_pass_bounds`        |
| `SigmaClip.fit_transform`            |     cls: 1.163 / 1.168 ms; reg: 1.106 / 1.112 ms | 1.254 / 1.281 ms | `direct_two_pass_soft_clip`     |
| `SigmaClip.transform`                |     cls: 0.698 / 0.706 ms; reg: 0.697 / 0.730 ms | 0.709 / 0.723 ms | `direct_soft_clip`              |
| `FeaturePermute.fit`                 |     cls: 0.109 / 0.131 ms; reg: 0.110 / 0.111 ms | 0.092 / 0.115 ms | `direct_shift_permutation`      |
| `FeaturePermute.fit_transform`       |     cls: 0.343 / 0.350 ms; reg: 0.323 / 0.338 ms | 0.074 / 0.101 ms | `direct_index_select`           |
| `FeaturePermute.transform`           |     cls: 0.239 / 0.271 ms; reg: 0.228 / 0.251 ms | 0.079 / 0.099 ms | `direct_index_select`           |
| `FeaturePermute.inverse_transform`   |     cls: 0.295 / 0.323 ms; reg: 0.291 / 0.595 ms | 0.075 / 0.113 ms | `direct_index_select`           |
| `SoftmaxTemperature.transform`       |     cls: 0.126 / 0.138 ms; reg: 0.091 / 0.100 ms | 0.074 / 0.098 ms | `direct_softmax`                |
| `CategoricalAlign.fit`               |     cls: 5.463 / 5.587 ms; reg: 6.230 / 6.773 ms | 0.213 / 0.248 ms | `dense_observed_mask`           |
| `CategoricalAlign.fit_transform`     | cls: 12.756 / 14.315 ms; reg: 14.339 / 15.316 ms | 0.362 / 0.394 ms | `dense_observed_lookup`         |
| `CategoricalAlign.transform`         |     cls: 7.113 / 8.501 ms; reg: 7.095 / 7.565 ms | 0.194 / 0.226 ms | `direct_lookup_gather`          |
| `ClassificationTarget.fit`           |                            cls: 0.176 / 0.202 ms | 0.089 / 0.113 ms | `direct_shift_permutation`      |
| `ClassificationTarget.fit_transform` |                            cls: 0.967 / 1.007 ms | 0.148 / 0.177 ms | `direct_permutation_gather`     |
| `ClassificationTarget.transform`     |                            cls: 0.791 / 0.813 ms | 0.159 / 0.191 ms | `direct_permutation_gather`     |
| `RegressionTarget.fit`               |                            reg: 0.211 / 0.226 ms | 0.117 / 0.147 ms | `direct_reduce_state`           |
| `RegressionTarget.fit_transform`     |                            reg: 0.284 / 0.298 ms | 0.143 / 0.183 ms | `direct_reduce_affine`          |
| `RegressionTarget.transform`         |                            reg: 0.091 / 0.096 ms | 0.076 / 0.090 ms | `direct_affine`                 |
| `RegressionTarget.inverse_transform` |                            reg: 0.090 / 0.094 ms | 0.076 / 0.108 ms | `direct_affine`                 |

Small differences where the large-stress current row is faster than the speed-of-light row are measurement noise across separate runs. The decision signal is the absolute latency and whether a candidate materially changes the implementation.

## Speed-of-light interpretation

The practical lower bound is now much closer to current main for most processors. The old multi-second Power gap has already been closed on `main` by preprocessing improvements. Remaining gaps fall into three categories:

- `Power.fit` can still go from about 32 ms to about 3.4 ms with the compiled analytic Newton prototype, but that changes the fitting algorithm and needs model-quality validation beyond parity checks.
- `Power.transform`, `Power.inverse_transform`, `Quantile.transform`, and `Quantile.inverse_transform` can reach sub-millisecond compiled lower bounds, but eager current main is already low-single-millisecond.
- `CategoricalAlign` still has the largest maintainable absolute cleanup opportunity for integer categorical workloads: dense observed masks and lookup gathers reduce fit/transform from 5-14 ms to 0.2-0.4 ms in the prototype.

For the remaining processors, the absolute runtimes are already sub-millisecond or around 1 ms. Some direct prototypes are faster, but the end-to-end recipe impact is too small to justify risky rewrites.

## CUDA timing, kernel time, host overhead, and throughput

The table below uses median timings from `processor_gpu_speed_of_light.json`. Wall time includes host orchestration plus synchronized CUDA execution. CUDA time is event-measured device time for the already-prepared operation. Throughput is recorded for every row in the JSON/CSV artifacts; selected current rows are shown here.

| Operation                    | Current wall | Current CUDA | Host overhead | Current throughput | Speed of light wall / CUDA | Candidate                       |
| ---------------------------- | -----------: | -----------: | ------------: | -----------------: | -------------------------: | ------------------------------- |
| `Power.fit`                  |    32.272 ms |    32.244 ms |      0.031 ms |    1.12e8 values/s |           3.381 / 3.352 ms | `batched_newton_compiled`       |
| `Power.fit_transform`        |    33.568 ms |    33.525 ms |      0.040 ms |    1.07e8 values/s |           3.411 / 3.380 ms | `batched_newton_compiled`       |
| `Power.transform`            |     1.386 ms |     1.360 ms |      0.025 ms |    3.25e9 values/s |           0.376 / 0.352 ms | `vectorized_compiled`           |
| `Power.inverse_transform`    |     3.217 ms |     3.188 ms |      0.029 ms |    1.40e9 values/s |           0.278 / 0.253 ms | `vectorized_compiled`           |
| `Quantile.fit_transform`     |     3.709 ms |     3.671 ms |      0.037 ms |    9.71e8 values/s |           3.709 / 3.671 ms | `current`                       |
| `Quantile.transform`         |     3.929 ms |     3.893 ms |      0.038 ms |    1.15e9 values/s |           0.831 / 0.806 ms | `batched_searchsorted_compiled` |
| `Quantile.inverse_transform` |     2.566 ms |     2.515 ms |      0.044 ms |    1.75e9 values/s |           0.221 / 0.199 ms | `batched_searchsorted_compiled` |
| `CategoricalAlign.transform` |     7.036 ms |     6.991 ms |      0.039 ms |    7.11e7 values/s |           0.194 / 0.157 ms | `direct_lookup_gather`          |

There is no CPU numerical fallback and no host/device transfer inside the timed Processor operations. Host overhead is now small for the main nonlinear rows; remaining gaps are mostly kernel fusion, batched search/gather layout, and algorithm choice rather than repeated explicit synchronization.

## Transfer and synchronization measurements

Pinned transfers use preallocated source and destination buffers. These are measured separately and are excluded from Processor kernel timing.

| Transfer                                      |       Shape |     Median / p95 |
| --------------------------------------------- | ----------: | ---------------: |
| `Power.fit` H2D input                         | 40,000 × 90 | 1.114 / 1.124 ms |
| `Power.fit` D2H output/state                  |      5 × 90 | 0.067 / 0.098 ms |
| `Quantile.fit` H2D input                      | 40,000 × 90 | 1.109 / 1.117 ms |
| `Quantile.fit` D2H output/state               |  1,000 × 90 | 0.056 / 0.062 ms |
| `Power.fit_transform` H2D input               | 40,000 × 90 | 1.109 / 1.121 ms |
| `Power.fit_transform` D2H output/state        | 40,000 × 90 | 1.130 / 1.141 ms |
| `Power.transform` H2D input                   | 50,000 × 90 | 1.387 / 1.393 ms |
| `Power.transform` D2H output/state            | 50,000 × 90 | 1.401 / 1.407 ms |
| `Power.inverse_transform` H2D input           | 50,000 × 90 | 1.376 / 1.394 ms |
| `Power.inverse_transform` D2H output/state    | 50,000 × 90 | 1.407 / 1.416 ms |
| `Quantile.fit_transform` H2D input            | 40,000 × 90 | 1.109 / 1.123 ms |
| `Quantile.fit_transform` D2H output/state     | 40,000 × 90 | 1.140 / 1.163 ms |
| `Quantile.transform` H2D input                | 50,000 × 90 | 1.378 / 1.388 ms |
| `Quantile.transform` D2H output/state         | 50,000 × 90 | 1.409 / 1.428 ms |
| `Quantile.inverse_transform` H2D input        | 50,000 × 90 | 1.375 / 1.389 ms |
| `Quantile.inverse_transform` D2H output/state | 50,000 × 90 | 1.407 / 1.414 ms |

An idle `torch.cuda.synchronize()` costs 0.00939 ms median and 0.0114 ms p95 on this run. If data is not already resident on device, H2D/D2H copies dominate the sub-millisecond compiled transform lower bounds; for example, `Power.transform` is about 3.16 ms end-to-end after adding measured input and output copies to the 0.376 ms kernel lower bound.

## Production recommendation

| Priority                                    | Change                                                                                                                                    | Expected result                                                                                                                                                            | Complexity / risk                                                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Already achieved on `main`                  | Keep the current finite-input Power fitting path.                                                                                         | `Power.fit` is now about 32 ms and `Power.fit_transform` about 33 ms on the L4, down from multi-second historical measurements and already under the earlier 50 ms target. | No additional PR needed for this benchmark report.                                                                                            |
| Quick win only if this path remains hot     | Keep Power transform/inverse vectorized and compiler-friendly; avoid reintroducing per-column Python scalar loops.                        | Current eager `Power.transform` is about 1.4 ms. Compiled callers can approach 0.376 ms transform and 0.278 ms inverse.                                                    | Low for preserving current structure; a standalone inverse rewrite is not justified because eager vectorized inverse was slower than current. |
| Medium effort                               | Replace integer `CategoricalAlign` fit/transform with dense observed masks and lookup-table gathers for bounded integer category domains. | Fit/fit_transform/transform lower bounds are 0.213 / 0.362 / 0.194 ms versus current 5.8 / 13.1 / 7.0 ms in the speed run.                                                 | Medium; category dtype, metadata, unseen-category semantics, and memory bounds need explicit coverage.                                        |
| Medium effort only for compiled deployments | Make Quantile transform/inverse graph-friendly for `torch.compile` without changing eager semantics.                                      | Current eager transform/inverse are 3.9 / 2.6 ms; compiled lower bounds are 0.831 / 0.221 ms.                                                                              | Medium; useful only if the model execution path actually compiles these processors.                                                           |
| Long term                                   | Validate analytic Newton `Power.fit` as an alternative fitting algorithm.                                                                 | Practical lower bound is 3.381 ms fit and 3.411 ms fit_transform.                                                                                                          | High numerical/model-quality risk because the optimizer changes, even though parity passed within the benchmark tolerance.                    |

The minimal conclusion is: current `main` has already closed the largest Power fit bottleneck for the benchmarked L4. The next low-risk production work is not another broad Power fit rewrite; it is targeted cleanup of categorical remapping if those workloads remain common, and preserving compiler-friendly tensor code so compiled inference paths can use the measured sub-millisecond transform limits.

## Rejected or deprioritized experiments

- `Quantile.fit` and `Quantile.fit_transform` direct preindexed candidates were not selected because the current implementation is the fastest passing candidate in this main-state run.
- Eager batched golden-section `Power.fit` was slower than current main (94.069 ms versus 32.272 ms), so it is not a production recommendation.
- Eager analytic Newton `Power.fit` was also slower than current main (36.235 ms versus 32.272 ms); only the compiled Newton prototype materially improves the fit path.
- Eager vectorized `Power.inverse_transform` was slower than current main (3.695 ms versus 3.217 ms), so inverse should not be changed unless the compiled path is part of the deployment plan.
- Quantile row-major, column-major, float16, bfloat16, and float64 transform variants were slower than the selected compiled candidate or added dtype/layout risk without enough benefit.
- A custom CUDA kernel is not warranted at this point. Standard PyTorch plus compilation already reaches the measured lower bounds for the remaining hot transform/search paths.

## Artifacts

- `tabiclv2_processing_large_stress.json` / `.csv`: 194 rerun SDM CPU/GPU large-stress results for explicit Power and Quantile Recipes and individual Processors.
- `processor_gpu_speed_of_light.json` / `.csv`: current, optimized, dtype, transfer, synchronization, correctness, throughput, CUDA-event, host-overhead, and peak-memory measurements.
- `tabiclv2_reference_large_stress.json` / `.csv`: historical pinned-reference large-stress results from the prior reference run.
- `tabiclv2_processing_full.json` / `.csv`: historical 512-workload Cartesian run from the prior six-factor suite.
- `tabiclv2_processing_large_baseline.json` and `tabiclv2_reference_large_baseline.json`: historical numerical-only, Identity-member baselines.

The current seven-factor Cartesian generator contains 128 characteristic combinations, or 1,024 task/size workloads before Recipe variants. The large-stress run is the reproducible all-factors subset refreshed here.
