# TabICLv2 processing performance report

## Method

Benchmarks ran on 14 July 2026 on an AWS Linux host with four AMD EPYC 7R13
vCPUs, 15 GiB RAM, and an NVIDIA L4 (23,034 MiB). SDM used PyTorch
2.12.1+cu130 with float32 tensors. The pinned reference used NumPy/sklearn
float32 inputs from TabICLv2 commit
`f719c886a586ed4a29236345e319ac1ea596c478`.

The headline combined-stress workload has 50,000 rows and 100 features, split
into 40,000 context and 10,000 query rows. It contains 90 numerical and 10
categorical columns and exercises all requested paths:

- a constant numerical column;
- 1% missing numerical values and missing categorical codes every 97 rows;
- 4,096 observed categorical values plus one query-only unseen value;
- a `1e12` query outlier for fixed `Clip(-100, 100)`;
- a value of `20` for sigma-based soft clipping;
- explicit Power and Quantile Recipe variants.

The reference receives the equivalent pandas DataFrame, so its
`TransformToNumerical` path fits an `OrdinalEncoder` and handles unseen
categories as `-1`. Each measurement uses one warmup and five repetitions.
Dataset construction, fitted-state preparation, assertions, and output
comparisons occur outside timed regions. GPU timings synchronize the device
before and after the operation. GPU peak memory is the largest incremental
CUDA allocation above the prepared-operation baseline. CPU peak memory remains
null to avoid perturbing short operations.

All 210 recorded large-stress results (194 SDM and 16 reference) have
`correctness_status="pass"`.

## Power Recipe headline

Times are median / p95 milliseconds. Cells contain classification / regression.
"Speedup" is SDM CPU median divided by SDM GPU median. Stage distributions are
measured independently and are not additive.

| Stage                              | Pinned TabICLv2 CPU |             SDM CPU |             SDM GPU | CPU/GPU speedup |
| ---------------------------------- | ------------------: | ------------------: | ------------------: | --------------: |
| Feature + target preprocessing     | 4310.751 / 4388.693 | 5500.250 / 5513.649 | 2911.733 / 2867.174 |   1.89× / 1.92× |
| Map model output to original space |      0.008 / 31.059 |      0.017 / 45.349 |       0.029 / 2.071 |  0.61× / 21.90× |
| Output transform                   |      0.059 / 0.0002 |       0.445 / 0.026 |       0.103 / 0.020 |   4.33× / 1.32× |
| Total Recipe overhead              | 4300.828 / 4368.834 | 5372.213 / 5549.592 | 2796.823 / 2732.915 |   1.92× / 2.03× |

The largest observed incremental GPU peak was 180,534,272 bytes (172.2 MiB)
for the regression Quantile total. Power preprocessing peaked at 97.4 MiB;
the Power regression total peaked at 171.8 MiB because it also materializes
the 10,000 × 999 output head.

Classification output mapping is too small for CUDA to amortize dispatch and
synchronization. Regression mapping is large enough to improve from 45.349 ms
on CPU to 2.071 ms on GPU.

## Quantile Recipe comparison

Quantile exposes a materially more GPU-friendly path than Power:

| Task           | Stage         | Pinned TabICLv2 CPU |  SDM CPU | SDM GPU | CPU/GPU speedup |
| -------------- | ------------- | ------------------: | -------: | ------: | --------------: |
| Classification | Preprocessing |            1830.848 | 1324.511 | 252.847 |           5.24× |
| Classification | Total Recipe  |            1789.067 | 1304.179 | 238.477 |           5.47× |
| Regression     | Preprocessing |            1781.132 | 1386.787 | 246.220 |           5.63× |
| Regression     | Total Recipe  |            1837.495 | 1478.236 | 239.726 |           6.17× |

## Individual Processor results

Complete individual Processor operation matrix on the combined-stress workload. Current timings are the original large-stress GPU medians/p95s. Non-target feature Processors show classification and regression because both tasks exercise the same operation. The speed-of-light column is the fastest passing CUDA candidate from `processor_gpu_speed_of_light.json`, measured once on the classification feature workload unless the row is target-specific.

| Processor / operation                |                                 Current GPU large stress |   Speed of light | Candidate                                    |
| ------------------------------------ | -------------------------------------------------------: | ---------------: | -------------------------------------------- |
| `StandardScale.fit`                  |             cls: 0.138 / 0.172 ms; reg: 0.135 / 0.167 ms | 0.129 / 0.150 ms | `direct_reduce_state`                        |
| `StandardScale.fit_transform`        |             cls: 0.240 / 0.518 ms; reg: 0.245 / 0.274 ms | 0.173 / 0.224 ms | `direct_reduce_affine`                       |
| `StandardScale.transform`            |             cls: 0.174 / 0.182 ms; reg: 0.178 / 0.185 ms | 0.180 / 0.193 ms | `direct_affine`                              |
| `StandardScale.inverse_transform`    |             cls: 0.178 / 0.182 ms; reg: 0.184 / 0.190 ms | 0.176 / 0.212 ms | `current`                                    |
| `Clip.transform`                     |             cls: 0.165 / 0.180 ms; reg: 0.163 / 0.176 ms | 0.064 / 0.089 ms | `direct_clamp`                               |
| `Power.fit`                          | cls: 2531.540 / 2588.057 ms; reg: 2490.340 / 2584.606 ms | 2.941 / 3.114 ms | `batched_newton_compiled`                    |
| `Power.fit_transform`                | cls: 2549.359 / 2558.115 ms; reg: 2537.143 / 2599.415 ms | 3.078 / 3.125 ms | `batched_newton_compiled`                    |
| `Power.transform`                    |         cls: 41.254 / 51.740 ms; reg: 36.302 / 40.340 ms | 0.380 / 0.394 ms | `vectorized_compiled`                        |
| `Power.inverse_transform`            |         cls: 43.433 / 45.743 ms; reg: 40.853 / 41.982 ms | 0.326 / 0.356 ms | `vectorized_compiled`                        |
| `Quantile.fit`                       |             cls: 0.590 / 0.680 ms; reg: 0.632 / 0.684 ms | 0.451 / 0.484 ms | `direct_nanquantile_preindexed`              |
| `Quantile.fit_transform`             |       cls: 100.505 / 136.100 ms; reg: 93.801 / 98.213 ms | 5.619 / 5.664 ms | `direct_nanquantile_searchsorted_preindexed` |
| `Quantile.transform`                 |         cls: 90.761 / 99.187 ms; reg: 93.460 / 98.471 ms | 0.884 / 0.915 ms | `batched_searchsorted_compiled`              |
| `Quantile.inverse_transform`         |         cls: 60.811 / 63.698 ms; reg: 73.496 / 83.658 ms | 4.572 / 4.938 ms | `batched_searchsorted_row_major`             |
| `SigmaClip.fit`                      |             cls: 1.707 / 1.731 ms; reg: 1.733 / 1.769 ms | 1.648 / 1.705 ms | `direct_two_pass_bounds`                     |
| `SigmaClip.fit_transform`            |             cls: 2.212 / 2.241 ms; reg: 2.191 / 2.217 ms | 2.196 / 2.299 ms | `current`                                    |
| `SigmaClip.transform`                |             cls: 0.752 / 0.773 ms; reg: 0.743 / 0.755 ms | 0.720 / 0.740 ms | `direct_soft_clip`                           |
| `FeaturePermute.fit`                 |             cls: 0.100 / 0.104 ms; reg: 0.099 / 0.103 ms | 0.113 / 0.146 ms | `direct_shift_permutation`                   |
| `FeaturePermute.fit_transform`       |             cls: 0.290 / 0.335 ms; reg: 0.288 / 0.385 ms | 0.078 / 0.121 ms | `direct_index_select`                        |
| `FeaturePermute.transform`           |             cls: 0.241 / 0.488 ms; reg: 0.242 / 0.259 ms | 0.102 / 0.134 ms | `direct_index_select`                        |
| `FeaturePermute.inverse_transform`   |             cls: 0.285 / 0.307 ms; reg: 0.261 / 0.294 ms | 0.072 / 0.095 ms | `direct_index_select`                        |
| `SoftmaxTemperature.transform`       |             cls: 0.094 / 0.123 ms; reg: 0.089 / 0.101 ms | 0.074 / 0.110 ms | `direct_softmax`                             |
| `CategoricalAlign.fit`               |             cls: 5.455 / 5.576 ms; reg: 5.458 / 5.528 ms | 0.174 / 0.205 ms | `dense_observed_mask`                        |
| `CategoricalAlign.fit_transform`     |         cls: 13.790 / 13.918 ms; reg: 13.682 / 13.942 ms | 0.382 / 0.429 ms | `dense_observed_lookup`                      |
| `CategoricalAlign.transform`         |             cls: 8.334 / 8.523 ms; reg: 8.398 / 8.555 ms | 0.204 / 0.267 ms | `direct_lookup_gather`                       |
| `ClassificationTarget.fit`           |                                    cls: 0.176 / 0.216 ms | 0.098 / 0.128 ms | `direct_shift_permutation`                   |
| `ClassificationTarget.fit_transform` |                                    cls: 0.933 / 0.989 ms | 0.168 / 0.208 ms | `direct_permutation_gather`                  |
| `ClassificationTarget.transform`     |                                    cls: 0.799 / 0.843 ms | 0.159 / 0.196 ms | `direct_permutation_gather`                  |
| `RegressionTarget.fit`               |                                    reg: 0.206 / 0.228 ms | 0.141 / 0.163 ms | `direct_reduce_state`                        |
| `RegressionTarget.fit_transform`     |                                    reg: 0.280 / 0.310 ms | 0.216 / 0.332 ms | `direct_reduce_affine`                       |
| `RegressionTarget.transform`         |                                    reg: 0.091 / 0.093 ms | 0.087 / 0.130 ms | `direct_affine`                              |
| `RegressionTarget.inverse_transform` |                                    reg: 0.097 / 0.131 ms | 0.090 / 0.107 ms | `direct_affine`                              |

The operations with meaningful absolute latency gaps remain the nonlinear numerical paths (`Power` and `Quantile`) plus integer categorical remapping. Simple affine, clamp, sigma-clip, softmax, and target-scale rows are already at sub-millisecond to low-single-millisecond absolute latency.

## Bottleneck and interpretation

`Power.fit` dominates the Power Recipe on both devices. Its feature-wise
Yeo-Johnson lambda search includes Python control flow and scalar reductions,
which limit GPU utilization and introduce synchronization. Tensorized
Quantile, scale, sigma clip, fixed Clip, permutation, output transform, and
large regression inverse mapping all benefit substantially from CUDA.

Consequently, moving the complete Recipe to GPU yields about 2× for Power but
5–6× for Quantile. The smallest localized performance opportunity is to
vectorize or batch Power lambda fitting; Clip is not a meaningful bottleneck.

## GPU speed-of-light follow-up

### Scope and method

The follow-up ran on 17 July 2026 on the same NVIDIA L4 (23,034 MiB), with PyTorch 2.12.1+cu130, CUDA 13.0, and float32 inputs. It now covers the full individual Processor operation matrix from the large-stress benchmark: 31 fit, fit_transform, transform, and inverse_transform rows across numerical, categorical, target, permutation, and output Processors.

Each result has five warmups and 20 measured repetitions. Host wall time and CUDA-event time are recorded around the already-prepared operation, with a device synchronization before and after each sample. Dataset creation, processor preparation, compilation, and correctness checks are outside the timed region. Incremental peak allocation is reset after preparation. Pinned, preallocated H2D and D2H copies are measured separately for the Power and Quantile fit/fit_transform/transform/inverse_transform shapes.

Power candidates must match fitted lambdas within `atol=1e-3` and standardized outputs within `rtol=1e-3, atol=5e-3`. Quantile transform/inverse candidates use `rtol=1e-6, atol=2e-5`, must preserve the exact NaN mask, and must retain the current repeated-value endpoint/midpoint behavior. Integer candidates must match exactly. All selected speed-of-light candidates pass. Four exploratory rows are intentionally rejected: float16, bfloat16, and float64 Quantile transform, plus compiled Quantile inverse.

### Minimum work

`Power.fit` and `Power.fit_transform` must establish safe per-feature lambda bounds, maximize the Yeo-Johnson log likelihood, apply the fitted transform, and compute the fitted mean and scale. The current golden-section algorithm requires about 44 likelihood refinements for these bounds. The maintainable baseline batches that same search over features. The lower-bound prototype instead evaluates analytic first and second derivatives and converges in four fused Newton steps.

`Power.transform` and `Power.inverse_transform` require sign-dependent Yeo-Johnson math plus affine scale/unscale. No feature loop is intrinsically required.

`Quantile.fit_transform` combines `nanquantile` over the selected sample with the same transform work as `Quantile.transform`. `Quantile.transform` requires two monotonic searches per finite value to preserve duplicate midpoints, interpolation, endpoint handling, inverse-normal mapping, and NaN restoration. `Quantile.inverse_transform` requires the inverse-normal CDF to map normal scores back to `[0, 1]`, then one monotonic search/interpolation per value.

For the remaining Processors, the minimum work is small: fixed clamp for `Clip`, reductions plus affine math for `StandardScale`, nanmean/std bounds plus soft clipping for `SigmaClip`, dense observed-category detection plus lookup-gather remapping for integer `CategoricalAlign`, index-select for `FeaturePermute`, softmax over logits, permutation gather for classification targets, and one-column affine math for regression targets.

### Headline result

Times are median / p95. “Maintainable” is the smallest eager PyTorch prototype recommended for production. “Lower bound” is the fastest passing prototype, using `torch.compile` after warmup where compilation materially changes the limit. “Gap closed” measures how much of the absolute distance from current to lower-bound latency the maintainable prototype removes.

| Processor operation          |                Current | Practical lower bound | Maintainable prototype | Current / prototype | Prototype / bound | Gap closed |
| ---------------------------- | ---------------------: | --------------------: | ---------------------: | ------------------: | ----------------: | ---------: |
| `Power.fit`                  | 2622.770 / 2697.540 ms |      2.941 / 3.114 ms |     98.796 / 99.600 ms |              26.55× |            33.60× |     96.34% |
| `Power.fit_transform`        | 2690.451 / 2941.937 ms |      3.078 / 3.125 ms |     99.113 / 99.825 ms |              27.15× |            32.20× |     96.43% |
| `Power.transform`            |     43.403 / 51.359 ms |      0.380 / 0.394 ms |       1.334 / 1.347 ms |              32.53× |             3.51× |     97.78% |
| `Power.inverse_transform`    |     45.090 / 55.336 ms |      0.326 / 0.356 ms |       3.938 / 3.957 ms |              11.45× |            12.09× |     91.93% |
| `Quantile.fit_transform`     |   105.418 / 115.008 ms |      5.619 / 5.664 ms |       5.619 / 5.664 ms |              18.76× |             1.00× |    100.00% |
| `Quantile.transform`         |   104.172 / 113.462 ms |      0.884 / 0.915 ms |       4.280 / 4.450 ms |              24.34× |             4.84× |     96.71% |
| `Quantile.inverse_transform` |    71.974 / 110.100 ms |      4.572 / 4.938 ms |       4.572 / 4.938 ms |              15.74× |             1.00× |    100.00% |

The largest practical gaps are still `Power.fit`, `Power.fit_transform`, `Power.transform`, `Power.inverse_transform`, and `Quantile.transform`. `Quantile.fit_transform` and `Quantile.inverse_transform` also have large absolute gaps, but their maintainable eager lower bounds are the fastest passing candidates measured here.

Peak incremental CUDA allocation for the headline rows is:

| Processor operation          |  Current | Maintainable | Lower bound |
| ---------------------------- | -------: | -----------: | ----------: |
| `Power.fit`                  | 72.9 MiB |    100.4 MiB |    55.3 MiB |
| `Power.fit_transform`        | 72.9 MiB |    100.9 MiB |    55.8 MiB |
| `Power.transform`            | 51.5 MiB |    103.0 MiB |    38.7 MiB |
| `Power.inverse_transform`    | 56.7 MiB |    176.9 MiB |    18.0 MiB |
| `Quantile.fit_transform`     | 59.9 MiB |    183.5 MiB |   183.5 MiB |
| `Quantile.transform`         | 20.0 MiB |     87.1 MiB |    40.4 MiB |
| `Quantile.inverse_transform` | 20.5 MiB |    211.4 MiB |   211.4 MiB |

### Complete individual Processor operation lower bounds

| Processor / operation                |   Current in speed run |   Speed of light | Candidate                                    | Current / bound |
| ------------------------------------ | ---------------------: | ---------------: | -------------------------------------------- | --------------: |
| `StandardScale.fit`                  |       0.245 / 0.300 ms | 0.129 / 0.150 ms | `direct_reduce_state`                        |           1.89× |
| `StandardScale.fit_transform`        |       0.442 / 0.489 ms | 0.173 / 0.224 ms | `direct_reduce_affine`                       |           2.55× |
| `StandardScale.transform`            |       0.185 / 0.227 ms | 0.180 / 0.193 ms | `direct_affine`                              |           1.03× |
| `StandardScale.inverse_transform`    |       0.176 / 0.212 ms | 0.176 / 0.212 ms | `current`                                    |           1.00× |
| `Clip.transform`                     |       0.149 / 0.203 ms | 0.064 / 0.089 ms | `direct_clamp`                               |           2.35× |
| `Power.fit`                          | 2622.770 / 2697.540 ms | 2.941 / 3.114 ms | `batched_newton_compiled`                    |         891.87× |
| `Power.fit_transform`                | 2690.451 / 2941.937 ms | 3.078 / 3.125 ms | `batched_newton_compiled`                    |         874.14× |
| `Power.transform`                    |     43.403 / 51.359 ms | 0.380 / 0.394 ms | `vectorized_compiled`                        |         114.12× |
| `Power.inverse_transform`            |     45.090 / 55.336 ms | 0.326 / 0.356 ms | `vectorized_compiled`                        |         138.41× |
| `Quantile.fit`                       |       0.647 / 0.712 ms | 0.451 / 0.484 ms | `direct_nanquantile_preindexed`              |           1.43× |
| `Quantile.fit_transform`             |   105.418 / 115.008 ms | 5.619 / 5.664 ms | `direct_nanquantile_searchsorted_preindexed` |          18.76× |
| `Quantile.transform`                 |   104.172 / 113.462 ms | 0.884 / 0.915 ms | `batched_searchsorted_compiled`              |         117.80× |
| `Quantile.inverse_transform`         |    71.974 / 110.100 ms | 4.572 / 4.938 ms | `batched_searchsorted_row_major`             |          15.74× |
| `SigmaClip.fit`                      |       1.672 / 1.742 ms | 1.648 / 1.705 ms | `direct_two_pass_bounds`                     |           1.01× |
| `SigmaClip.fit_transform`            |       2.196 / 2.299 ms | 2.196 / 2.299 ms | `current`                                    |           1.00× |
| `SigmaClip.transform`                |       0.723 / 0.731 ms | 0.720 / 0.740 ms | `direct_soft_clip`                           |           1.00× |
| `FeaturePermute.fit`                 |       0.196 / 0.226 ms | 0.113 / 0.146 ms | `direct_shift_permutation`                   |           1.74× |
| `FeaturePermute.fit_transform`       |       0.448 / 0.503 ms | 0.078 / 0.121 ms | `direct_index_select`                        |           5.74× |
| `FeaturePermute.transform`           |       0.237 / 0.296 ms | 0.102 / 0.134 ms | `direct_index_select`                        |           2.33× |
| `FeaturePermute.inverse_transform`   |       0.289 / 0.330 ms | 0.072 / 0.095 ms | `direct_index_select`                        |           4.00× |
| `SoftmaxTemperature.transform`       |       0.123 / 0.152 ms | 0.074 / 0.110 ms | `direct_softmax`                             |           1.66× |
| `CategoricalAlign.fit`               |       5.671 / 7.236 ms | 0.174 / 0.205 ms | `dense_observed_mask`                        |          32.66× |
| `CategoricalAlign.fit_transform`     |     19.094 / 23.695 ms | 0.382 / 0.429 ms | `dense_observed_lookup`                      |          50.04× |
| `CategoricalAlign.transform`         |      9.757 / 15.085 ms | 0.204 / 0.267 ms | `direct_lookup_gather`                       |          47.94× |
| `ClassificationTarget.fit`           |       0.231 / 0.275 ms | 0.098 / 0.128 ms | `direct_shift_permutation`                   |           2.36× |
| `ClassificationTarget.fit_transform` |       1.022 / 1.105 ms | 0.168 / 0.208 ms | `direct_permutation_gather`                  |           6.08× |
| `ClassificationTarget.transform`     |       0.823 / 0.882 ms | 0.159 / 0.196 ms | `direct_permutation_gather`                  |           5.16× |
| `RegressionTarget.fit`               |       0.346 / 0.385 ms | 0.141 / 0.163 ms | `direct_reduce_state`                        |           2.45× |
| `RegressionTarget.fit_transform`     |       0.502 / 0.906 ms | 0.216 / 0.332 ms | `direct_reduce_affine`                       |           2.33× |
| `RegressionTarget.transform`         |       0.185 / 0.206 ms | 0.087 / 0.130 ms | `direct_affine`                              |           2.11× |
| `RegressionTarget.inverse_transform` |       0.150 / 0.184 ms | 0.090 / 0.107 ms | `direct_affine`                              |           1.67× |

These complete lower bounds do not change the production priority. `Power` and `Quantile` dominate absolute runtime. `CategoricalAlign` has a large relative gap and is worth a medium-priority cleanup, but its absolute cost is much smaller than the nonlinear numerical paths. `SigmaClip`, `StandardScale`, `Clip`, `SoftmaxTemperature`, `FeaturePermute.fit`, and regression target scaling are already near their practical limits or too small to justify risky changes.

### Why the current paths are slow

CUDA profiling shows orchestration and synchronization, not device arithmetic, dominate the current nonlinear paths. Profiler instrumentation perturbs wall time, so summed kernel time is diagnostic rather than directly additive to the unprofiled medians.

| Operation                             | GPU kernels | Summed kernel time | Scalar syncs | D2H events |
| ------------------------------------- | ----------: | -----------------: | -----------: | ---------: |
| Current `Power.fit`                   |     232,182 |         447.421 ms |       12,736 |     29,328 |
| Compiled Newton `Power.fit`           |          58 |           2.782 ms |            0 |          0 |
| Current `Power.transform`             |       3,603 |           6.987 ms |           90 |        450 |
| Compiled vector `Power.transform`     |           6 |           0.346 ms |            0 |          0 |
| Current `Quantile.transform`          |       7,888 |          15.703 ms |            0 |        360 |
| Compiled batched `Quantile.transform` |           7 |           0.674 ms |            0 |          0 |

`Power.fit` filters and optimizes one feature at a time. Python converts CUDA scalars for likelihood decisions and fitted lambdas, while boolean indexing per likelihood evaluation triggers dynamic compaction and D2H size reads. The result is thousands of synchronizations, repeated temporary allocations, and over 232,000 small kernels.

`Power.transform` and `Power.inverse_transform` repeat masked Yeo-Johnson math by feature and convert 90 CUDA lambdas to Python floats. `Quantile.transform` and `Quantile.inverse_transform` avoid explicit scalar `.item()` fallbacks, but their per-column finite-value boolean indexing creates dynamic output sizes and fragmented kernels. Batched search removes that compaction.

There is no CPU numerical fallback and no input/output transfer inside the timed Processor operations. The dominant limits are CPU-side orchestration, kernel-launch fragmentation, and synchronization. After fusion, the paths are primarily pointwise/reduction compute (`Power`) or search/memory access (`Quantile`).

### Transfers and synchronization

Pinned transfers use preallocated source and destination buffers:

| Transfer                                      |       Shape |     Median / p95 |
| --------------------------------------------- | ----------: | ---------------: |
| `Power.fit` H2D input                         | 40,000 × 90 | 1.121 / 1.131 ms |
| `Power.fit` D2H output/state                  |      5 × 90 | 0.052 / 0.070 ms |
| `Quantile.fit` H2D input                      | 40,000 × 90 | 1.110 / 1.127 ms |
| `Quantile.fit` D2H output/state               |  1,000 × 90 | 0.060 / 0.085 ms |
| `Power.fit_transform` H2D input               | 40,000 × 90 | 1.122 / 1.133 ms |
| `Power.fit_transform` D2H output/state        | 40,000 × 90 | 1.140 / 1.150 ms |
| `Power.transform` H2D input                   | 50,000 × 90 | 1.386 / 1.404 ms |
| `Power.transform` D2H output/state            | 50,000 × 90 | 1.397 / 1.415 ms |
| `Power.inverse_transform` H2D input           | 50,000 × 90 | 1.386 / 1.393 ms |
| `Power.inverse_transform` D2H output/state    | 50,000 × 90 | 1.410 / 1.417 ms |
| `Quantile.fit_transform` H2D input            | 40,000 × 90 | 1.125 / 1.135 ms |
| `Quantile.fit_transform` D2H output/state     | 40,000 × 90 | 1.133 / 1.144 ms |
| `Quantile.transform` H2D input                | 50,000 × 90 | 1.379 / 1.391 ms |
| `Quantile.transform` D2H output/state         | 50,000 × 90 | 1.389 / 1.394 ms |
| `Quantile.inverse_transform` H2D input        | 50,000 × 90 | 1.377 / 1.381 ms |
| `Quantile.inverse_transform` D2H output/state | 50,000 × 90 | 1.404 / 1.426 ms |

An idle `torch.cuda.synchronize()` costs 0.0103 ms median and 0.0135 ms p95. If data is not resident, transfer dominates the sub-millisecond transform lower bounds; for example `Power.transform` becomes about 3.16 ms end-to-end when adding measured H2D and D2H copies.

### Production recommendation

| Priority      | Change                                                                                                                                                          | Expected result                                                                                                                  | Complexity / risk                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Quick win     | Replace the feature loop in `Power.transform` and `Power.inverse_transform` with broadcasted, NaN-safe Yeo-Johnson expressions.                                 | `Power.transform` 1.334 ms eager; `Power.inverse_transform` 3.938 ms eager. Compiled callers can approach 0.380 ms and 0.326 ms. | Small; low risk. Inverse uses the same fitted buffers and bound repair as the current implementation.             |
| Quick win     | Replace Quantile’s column loop and finite-value compaction with NaN-safe batched `searchsorted` for `transform`, `fit_transform`, and `inverse_transform`.      | `Quantile.transform` 4.280 ms fastest eager, 0.884 ms compiled; `fit_transform` 5.619 ms; inverse 4.572 ms.                      | Small-to-medium; low numerical risk, with temporary-memory tradeoffs for all-column execution.                    |
| Medium effort | Batch the existing golden-section `Power.fit` search over all features, precompute reusable masks/Jacobian terms, and keep it compile-friendly.                 | 98.796 ms eager for fit and 99.113 ms eager for fit_transform; 24–26 ms when compiled after warmup.                              | Medium; reduction order changes fitted lambdas by at most `1e-3` here, so strict downstream parity must be rerun. |
| Medium effort | Replace integer `CategoricalAlign` fit/transform with dense observed-category masks and lookup-table gathers when category dtypes support device-side matching. | 0.174 ms fit, 0.382 ms fit_transform, and 0.204 ms transform lower bounds in this integer workload.                              | Medium; dtype-specific vocabulary behavior and metadata require careful coverage.                                 |
| Long term     | Use the four-step analytic Newton solver only after broader convergence and model-quality validation.                                                           | 2.941 ms fit and 3.078 ms fit_transform practical L4 lower bounds.                                                               | High numerical/algorithmic risk despite low runtime complexity.                                                   |

The minimal production package is therefore: vectorize Power transform/inverse, batch Quantile transform/fit_transform/inverse with `searchsorted`, and batch the existing golden Power fit. This closes most of the absolute latency gap using ordinary PyTorch. Keeping the helpers graph-friendly lets callers that already compile their model approach the measured sub-millisecond transform/inverse bounds. No custom CUDA kernel is warranted at this stage.

### Rejected experiments

- Float16 and bfloat16 inverse-normal mapping are unsupported by `ndtri_cuda`. Mixed-precision Quantile transform ran in about 4.85–4.88 ms but had 5.20 maximum absolute output error, so both fail tolerance.
- Float64 Quantile transform took 19.811 ms, peaked at 417.2 MiB, and did not reproduce the current float32 interpolation within tolerance.
- Compiled Quantile inverse measured 0.366 ms but missed tolerance with maximum absolute error `1.04e-4`, so the passing speed of light remains the 4.572 ms eager batched inverse.
- Processing all Quantile features together and retaining a column-major layout both used substantially more memory and were slower than the 32-feature eager transform chunking path.
- Retaining per-column Power likelihood reductions while merely removing scalar decisions left the workload dominated by small kernels and did not approach the batched baseline.
- A custom CUDA kernel was not pursued: compiled standard PyTorch already reaches 58 kernels / 2.94 ms for Power fit and 7 kernels / 0.88 ms for Quantile transform.

## Artifacts

- `tabiclv2_processing_large_stress.json` / `.csv`: 194 SDM CPU/GPU
  large-stress results for explicit Power and Quantile Recipes and individual
  Processors.
- `tabiclv2_reference_large_stress.json` / `.csv`: 16 pinned-reference
  large-stress results.
- `tabiclv2_processing_full.json` / `.csv`: historical 512-workload
  Cartesian run from the prior six-factor suite.
- `tabiclv2_processing_large_baseline.json` and
  `tabiclv2_reference_large_baseline.json`: historical numerical-only,
  Identity-member baselines.
- `processor_gpu_speed_of_light.json` / `.csv`: current, optimized, dtype,
  transfer, synchronization, correctness, and peak-memory measurements.

The current seven-factor Cartesian generator contains 128 characteristic
combinations, or 1,024 task/size workloads before Recipe variants. The
large-stress run is the reproducible all-factors subset requested here.
