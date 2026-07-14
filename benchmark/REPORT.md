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

Representative classification timings on the same combined-stress workload:

| Processor / operation               | Device |      Median |         p95 | Notes                                      |
| ----------------------------------- | ------ | ----------: | ----------: | ------------------------------------------ |
| `Clip.transform`                    | L4     |    0.165 ms |    0.180 ms | 18.0 MB incremental peak                   |
| `Power.fit`                         | L4     | 2531.540 ms | 2588.057 ms | Primary bottleneck; CPU median 4419.076 ms |
| `Power.transform`                   | L4     |   41.254 ms |   51.740 ms | CPU median 114.987 ms                      |
| `Quantile.fit`                      | L4     |    0.590 ms |    0.680 ms | CPU median 34.673 ms                       |
| `Quantile.transform`                | L4     |   90.761 ms |   99.187 ms | CPU median 543.585 ms                      |
| `SigmaClip.fit_transform`           | L4     |    2.212 ms |    2.241 ms | CPU median 55.454 ms                       |
| `CategoricalAlign.fit`              | L4     |    5.455 ms |    5.576 ms | 10 columns, 4,096-value vocabulary         |
| `CategoricalAlign.transform`        | L4     |    8.334 ms |    8.523 ms | Includes missing and unseen query values   |
| `StandardScale.fit_transform`       | L4     |    0.240 ms |    0.518 ms | CPU median 12.589 ms                       |
| Classification target fit/transform | L4     |    0.933 ms |    0.989 ms | Sorted alignment plus CategoryShuffle      |
| Regression target inverse           | L4     |    0.097 ms |    0.131 ms | One-column target Processor only           |

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

The current seven-factor Cartesian generator contains 128 characteristic
combinations, or 1,024 task/size workloads before Recipe variants. The
large-stress run is the reproducible all-factors subset requested here.
