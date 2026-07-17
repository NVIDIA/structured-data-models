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

Representative classification timings on the same combined-stress workload.
The speed-of-light column is the fastest passing CUDA candidate from
`processor_gpu_speed_of_light.json`; current timings remain the original
large-stress Processor benchmark medians/p95s.

| Processor / operation               | Device |                Current |   Speed of light | Notes                                                                  |
| ----------------------------------- | ------ | ---------------------: | ---------------: | ---------------------------------------------------------------------- |
| `Clip.transform`                    | L4     |       0.165 / 0.180 ms | 0.066 / 0.088 ms | Direct clamp; current already small                                    |
| `Power.fit`                         | L4     | 2531.540 / 2588.057 ms | 2.979 / 3.031 ms | Primary bottleneck; compiled Newton lower bound                        |
| `Power.transform`                   | L4     |     41.254 / 51.740 ms | 0.370 / 0.395 ms | Compiled vectorized Yeo-Johnson; eager direct path is 1.263 ms         |
| `Quantile.fit`                      | L4     |       0.590 / 0.680 ms | 0.451 / 0.464 ms | Direct pre-indexed `nanquantile`; current includes subsample selection |
| `Quantile.transform`                | L4     |     90.761 / 99.187 ms | 0.873 / 0.913 ms | Compiled batched searchsorted; fastest eager path is 5.093 ms          |
| `SigmaClip.fit_transform`           | L4     |       2.212 / 2.241 ms | 2.131 / 2.151 ms | Already near the direct two-pass soft-clip lower bound                 |
| `CategoricalAlign.fit`              | L4     |       5.455 / 5.576 ms | 0.174 / 0.217 ms | Dense observed-mask lower bound for this integer-category workload     |
| `CategoricalAlign.transform`        | L4     |       8.334 / 8.523 ms | 0.268 / 0.304 ms | Precomputed lookup-table gather for missing/unseen-aware remapping     |
| `StandardScale.fit_transform`       | L4     |       0.240 / 0.518 ms | 0.159 / 0.192 ms | Direct reduction plus affine transform                                 |
| Classification target fit/transform | L4     |       0.933 / 0.989 ms | 0.153 / 0.188 ms | Direct permutation gather for the `CategoryShuffle` target Processor   |
| Regression target inverse           | L4     |       0.097 / 0.131 ms | 0.081 / 0.096 ms | One-column direct affine inverse                                       |

The full JSON/CSV artifacts contain fit, transform, fit_transform, and
inverse_transform distributions for every applicable Processor and both tasks.

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

The follow-up ran on 17 July 2026 on the same NVIDIA L4 (23,034 MiB), with
PyTorch 2.12.1+cu130, CUDA 13.0, and float32 inputs. It reuses the
combined-stress data above and isolates both the dominant Processor paths and
the remaining representative rows from `Individual Processor results`:

- `Power.fit`: 40,000 context rows × 90 numerical columns;
- `Power.transform` and `Quantile.transform`: 50,000 total rows × 90 numerical
  columns;
- `Clip`, `StandardScale`, `SigmaClip`, `Quantile.fit`, `CategoricalAlign`, and
  target Processor operations on the same shapes used by the individual
  Processor benchmark table.

Each result has five warmups and 20 measured repetitions. Host wall time and
CUDA-event time are recorded around the already-prepared operation, with a
device synchronization before and after each sample. Dataset creation,
processor preparation, compilation, and correctness checks are outside the
timed region. Incremental peak allocation is reset after preparation. Pinned,
preallocated H2D and D2H copies are measured separately.

Power candidates must match fitted lambdas within `atol=1e-3` and standardized
outputs within `rtol=1e-3, atol=5e-3`. Quantile transform candidates use
`rtol=1e-6, atol=2e-5`, must preserve the exact NaN mask, and must retain the
current repeated-value midpoint behavior. The additional Processor candidates
match exactly for integer outputs and use NaN-aware allclose checks for
floating-point outputs. All headline and individual speed-of-light candidates
pass.

### Minimum work

`Power.fit` must establish safe per-feature lambda bounds, maximize the
Yeo-Johnson log likelihood, apply the fitted transform, and compute the fitted
mean and scale. The current golden-section algorithm requires about 44
likelihood refinements for these bounds. The maintainable baseline batches
that same search over features. The lower-bound prototype instead evaluates
analytic first and second derivatives and converges in four fused Newton
steps.

`Power.transform` requires one sign-dependent Yeo-Johnson transform plus one
affine standardization per value. No feature loop is intrinsically required.

`Quantile.transform` requires two monotonic searches per finite value (forward
and reverse to preserve duplicate midpoints), linear interpolation, endpoint
handling, inverse-normal mapping, and NaN restoration. Batched
`torch.searchsorted` supplies the required GPU primitive.

For the remaining individual Processors, the minimum work is small and mostly
confirms that they are not the dominant bottleneck: fixed clamp for `Clip`,
column reductions plus affine math for `StandardScale`, two nanmean/std passes
plus soft clipping for `SigmaClip`, `nanquantile` over a selected sample for
`Quantile.fit`, dense observed-category detection plus lookup-gather remapping
for integer `CategoricalAlign`, permutation gather for the classification
target, and one affine inverse for the regression target.

### Headline result

Times are median / p95. “Maintainable” is the smallest eager PyTorch prototype
recommended for production. “Lower bound” is the fastest passing prototype,
using `torch.compile` after warmup where compilation materially changes the
limit. “Gap closed” measures how much of the absolute distance from current to
lower-bound latency the maintainable prototype removes.

| Processor operation  |                Current | Practical lower bound | Maintainable prototype | Current / prototype | Prototype / bound | Gap closed |
| -------------------- | ---------------------: | --------------------: | ---------------------: | ------------------: | ----------------: | ---------: |
| `Power.fit`          | 2532.885 / 2581.933 ms |      2.979 / 3.031 ms |     94.449 / 95.353 ms |              26.82× |            31.71× |     96.38% |
| `Power.transform`    |     43.431 / 45.669 ms |      0.370 / 0.395 ms |       1.263 / 1.281 ms |              34.38× |             3.41× |     97.93% |
| `Quantile.transform` |     94.786 / 99.788 ms |      0.873 / 0.913 ms |       5.093 / 5.207 ms |              18.61× |             5.83× |     95.51% |

The current-to-lower-bound speedups are 850.39×, 117.27×, and 108.52×,
respectively. Throughput rises from 1.42 million to 1.21 billion values/s for
`Power.fit`, from 103.6 million to 12.15 billion values/s for
`Power.transform`, and from 47.5 million to 5.15 billion values/s for
`Quantile.transform`.

Peak incremental CUDA allocation is:

| Processor operation  |  Current | Maintainable | Lower bound |
| -------------------- | -------: | -----------: | ----------: |
| `Power.fit`          | 73.2 MiB |     99.8 MiB |    56.3 MiB |
| `Power.transform`    | 53.3 MiB |    104.8 MiB |    39.6 MiB |
| `Quantile.transform` | 20.0 MiB |     87.4 MiB |    40.3 MiB |

The fastest eager Quantile candidate still uses 32-feature chunks: all 90
features at once took 7.891 ms and peaked at 233.0 MiB, while 32-feature
chunks took 5.093 ms and peaked at 87.4 MiB. A pre-transposed column-major
layout took 7.609 ms and 215.0 MiB, so changing the table layout is not
justified. If production maintainability is prioritized over the last 2.8 ms,
an all-column implementation remains a simpler low-risk option.

### Additional individual Processor lower bounds

| Processor operation                   | Current in speed run |   Speed of light | Candidate                       | Current / bound |
| ------------------------------------- | -------------------: | ---------------: | ------------------------------- | --------------: |
| `Clip.transform`                      |     0.143 / 0.190 ms | 0.066 / 0.088 ms | `direct_clamp`                  |           2.16× |
| `StandardScale.fit_transform`         |     0.313 / 0.345 ms | 0.159 / 0.192 ms | `direct_reduce_affine`          |           1.98× |
| `SigmaClip.fit_transform`             |     2.164 / 2.400 ms | 2.131 / 2.151 ms | `direct_two_pass_soft_clip`     |           1.02× |
| `Quantile.fit`                        |     0.630 / 0.675 ms | 0.451 / 0.464 ms | `direct_nanquantile_preindexed` |           1.40× |
| `CategoricalAlign.fit`                |     5.609 / 6.729 ms | 0.174 / 0.217 ms | `dense_observed_mask`           |          32.25× |
| `CategoricalAlign.transform`          |    8.282 / 10.160 ms | 0.268 / 0.304 ms | `direct_lookup_gather`          |          30.93× |
| Classification target `fit_transform` |     1.022 / 1.403 ms | 0.153 / 0.188 ms | `direct_permutation_gather`     |           6.70× |
| Regression target `inverse_transform` |     0.117 / 0.170 ms | 0.081 / 0.096 ms | `direct_affine`                 |           1.46× |

These extra lower bounds do not change the production priority. `SigmaClip`,
`StandardScale`, `Clip`, `Quantile.fit`, and the regression target are already
small. `CategoricalAlign` has a large relative gap, but its absolute latency is
single-digit milliseconds and depends on integer dense categories in this
workload; it is a medium-priority cleanup after the nonlinear numerical paths.

### Why the current paths are slow

CUDA profiling shows orchestration and synchronization, not device arithmetic,
dominate the current nonlinear paths. Profiler instrumentation perturbs wall
time, so summed kernel time is diagnostic rather than directly additive to the
unprofiled medians.

| Operation                             | GPU kernels | Summed kernel time | Scalar syncs | D2H events |
| ------------------------------------- | ----------: | -----------------: | -----------: | ---------: |
| Current `Power.fit`                   |     232,182 |         447.421 ms |       12,736 |     29,328 |
| Compiled Newton `Power.fit`           |          58 |           2.782 ms |            0 |          0 |
| Current `Power.transform`             |       3,603 |           6.987 ms |           90 |        450 |
| Compiled vector `Power.transform`     |           6 |           0.346 ms |            0 |          0 |
| Current `Quantile.transform`          |       7,888 |          15.703 ms |            0 |        360 |
| Compiled batched `Quantile.transform` |           7 |           0.674 ms |            0 |          0 |

`Power.fit` filters and optimizes one feature at a time. Python converts CUDA
scalars for likelihood decisions and fitted lambdas, while boolean indexing
per likelihood evaluation triggers dynamic compaction and D2H size reads. The
result is thousands of synchronizations, repeated temporary allocations, and
over 232,000 small kernels. Only about 19% of the final unprofiled wall time is
represented by summed kernel execution.

`Power.transform` repeats the same masked transform by feature and converts 90
CUDA lambdas to Python floats. Its actual kernels sum to only about 7 ms of a
43.4 ms operation.

`Quantile.transform` has no explicit scalar `.item()` fallback, but its
per-column finite-value boolean indexing creates dynamic output sizes. Those
operations generated 360 D2H events and fragmented 15.7 ms of kernel work
across 7,888 launches. A NaN-safe batched search removes that compaction.

There is no CPU numerical fallback and no input/output transfer inside the
timed Processor operations. The dominant limits are CPU-side orchestration,
kernel-launch fragmentation, and synchronization. After fusion, the paths are
primarily pointwise/reduction compute (`Power`) or search/memory access
(`Quantile`).

### Transfers and synchronization

Pinned transfers use preallocated source and destination buffers:

| Transfer                        |       Shape |     Median / p95 |
| ------------------------------- | ----------: | ---------------: |
| `Power.fit` H2D input           | 40,000 × 90 | 1.111 / 1.120 ms |
| `Power.fit` D2H fitted state    |      5 × 90 | 0.053 / 0.084 ms |
| Transform H2D input             | 50,000 × 90 | 1.377 / 1.386 ms |
| `Power.transform` D2H output    | 50,000 × 90 | 1.373 / 1.380 ms |
| `Quantile.transform` H2D input  | 50,000 × 90 | 1.375 / 1.387 ms |
| `Quantile.transform` D2H output | 50,000 × 90 | 1.367 / 1.381 ms |

An idle `torch.cuda.synchronize()` costs 0.0059 ms median and 0.0093 ms p95.
If data is not resident, the lower-bound end-to-end times become about 4.14 ms
for `Power.fit`, 3.12 ms for `Power.transform`, and 3.62 ms for
`Quantile.transform`; transfer then dominates both transform paths.

### Production recommendation

| Priority      | Change                                                                                                                                                                                    | Expected result                                                                                | Complexity / risk                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Quick win     | Replace the feature loop in `Power.transform` with one broadcasted, NaN-safe Yeo-Johnson expression.                                                                                      | 1.263 ms, 34.38× faster; 97.93% of the gap closed.                                             | Small; low risk. Maximum non-outlier drift is below float32 tolerance.                                            |
| Quick win     | Replace Quantile’s column loop and finite-value compaction with NaN-safe searchsorted. Use all-column execution for the smallest change, or chunk if measured memory/runtime requires it. | 5.093 ms fastest eager with chunks; 7.891 ms all-column; both close most of the transform gap. | Small-to-medium; low numerical risk, with a temporary-memory tradeoff for all-column execution.                   |
| Medium effort | Batch the existing golden-section `Power.fit` search over all features, precompute reusable masks/Jacobian terms, and keep it compile-friendly.                                           | 94.449 ms eager (26.82×); 23.251 ms when compiled after warmup.                                | Medium; reduction order changes fitted lambdas by at most `1e-3` here, so strict downstream parity must be rerun. |
| Medium effort | Replace integer `CategoricalAlign.transform` with a precomputed mapping tensor and one gather when category dtypes support device-side matching.                                          | 0.268 ms lower bound from 8.282 ms current in the speed run.                                   | Medium; dtype-specific behavior and vocabulary metadata require careful coverage.                                 |
| Long term     | Use the four-step analytic Newton solver only after broader convergence and model-quality validation.                                                                                     | 2.979 ms, the practical L4 lower bound for `Power.fit`.                                        | High numerical/algorithmic risk despite low runtime complexity.                                                   |

The minimal production package is therefore: vectorize both transform paths
and batch the existing golden search, without embedding `torch.compile` inside
the Processors. That package removes 95–98% of the absolute latency gap using
ordinary PyTorch. Keeping the helpers graph-friendly lets callers that already
compile their model approach the measured 0.37–0.87 ms transform bounds and
the 23.3 ms same-algorithm Power-fit bound. No custom CUDA kernel is warranted
at this stage.

### Rejected experiments

- Float16 and bfloat16 inverse-normal mapping are unsupported by
  `ndtri_cuda`. Mixed-precision search ran in about 5.13 ms but had 5.20
  maximum absolute output error, so both fail tolerance.
- Float64 Quantile took 19.837 ms, peaked at 417.2 MiB, and did not reproduce
  the current float32 interpolation within tolerance.
- Processing all Quantile features together and retaining a column-major
  layout both used substantially more memory and were slower than 32-feature
  chunks.
- Retaining per-column Power likelihood reductions while merely removing
  scalar decisions left the workload dominated by small kernels and did not
  approach the batched baseline.
- A custom CUDA kernel was not pursued: compiled standard PyTorch already
  reaches 58 kernels / 2.98 ms for Power fit and 7 kernels / 0.87 ms for
  Quantile transform.

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
