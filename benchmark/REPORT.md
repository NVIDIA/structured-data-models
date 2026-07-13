# TabICLv2 processing performance report

## Method

Benchmarks ran on 13 July 2026 on an AWS Linux host with four AMD EPYC 7R13
vCPUs and 15 GiB RAM. An NVIDIA L4 (23,034 MiB) was present and used for the
pinned model parity tests, but all Processors in this benchmark ran on CPU.
SDM used PyTorch 2.12.1+cu130 with float32 tensors. The pinned reference used
NumPy/sklearn float32 inputs from TabICLv2 commit
`f719c886a586ed4a29236345e319ac1ea596c478`.

Dataset sizes were selected around the pinned TabICLv2 50,000-row/100-feature
headline workload while retaining smaller development-sized cases:

| Size   |   Rows | Features | Context rows | Query rows |
| ------ | -----: | -------: | -----------: | ---------: |
| Tiny   |    300 |       10 |          240 |         60 |
| Small  |  1,000 |       50 |          800 |        200 |
| Medium | 10,000 |      100 |        8,000 |      2,000 |
| Large  | 50,000 |      100 |       40,000 |     10,000 |

The full SDM run contains exactly 512 total-Recipe workloads: four sizes,
binary classification and regression, and the Cartesian product of six
binary characteristics (constant columns, HardClip outliers, sigma outliers,
categorical columns, missing values, and unknown query categories). Detailed
stage measurements use the baseline, every one-factor-at-a-time case, and two
combined stress cases. Individual Processor measurements cover fit,
transform, fit_transform, and inverse_transform where supported.

Dataset creation, fresh-Processor preparation, assertions, and output
comparisons occur outside timed regions. The full matrix uses one warmup and
three repetitions. The Large baseline headline uses one warmup and ten
repetitions. Median, p95, and population standard deviation are recorded.
CPU peak memory is `null`: sampling it would perturb these short operations.
Observed process RSS during the full run was approximately 808 MB, but it is
not reported as a per-operation peak.

## Large baseline headline

Times below are milliseconds for 50,000 rows, 100 numerical features, 40,000
context rows, 10,000 query rows, CPU, float32 input, and ten repetitions.
The explicit member uses Identity normalization in both implementations.

### Binary classification

| Stage                                    | Pinned TabICLv2 median / p95 |  SDM median / p95 |      SDM difference |
| ---------------------------------------- | ---------------------------: | ----------------: | ------------------: |
| Feature + target preprocessing           |            259.272 / 275.178 | 200.334 / 243.763 | -58.939 ms (-22.7%) |
| Map model output to original class space |                0.008 / 0.010 |     0.097 / 0.125 |           +0.089 ms |
| Output transform                         |                0.061 / 0.084 |     0.437 / 0.471 |           +0.376 ms |
| E2E deterministic zero-head pipeline     |            261.080 / 299.176 | 207.153 / 210.961 | -53.927 ms (-20.7%) |
| Total processing overhead                |            261.080 / 299.176 | 207.153 / 210.961 | -53.927 ms (-20.7%) |

### Regression

The model-output mapping uses a 999-coordinate head, matching TabICLv2's raw
regression output width.

| Stage                                     | Pinned TabICLv2 median / p95 |  SDM median / p95 |      SDM difference |
| ----------------------------------------- | ---------------------------: | ----------------: | ------------------: |
| Feature + target preprocessing            |            244.691 / 267.188 | 211.536 / 217.519 | -33.155 ms (-13.5%) |
| Map model output to original target space |              38.366 / 42.665 |   43.393 / 44.594 |  +5.027 ms (+13.1%) |
| Output transform                          |            0.00018 / 0.00046 |     0.023 / 0.028 |           +0.023 ms |
| E2E deterministic zero-head pipeline      |            277.552 / 290.675 | 315.298 / 363.493 | +37.746 ms (+13.6%) |
| Total processing overhead                 |            277.552 / 290.675 | 315.298 / 363.493 | +37.746 ms (+13.6%) |

Stage medians are independent measurements and are not additive. In
particular, CPU scheduling and allocator reuse explain why a separately timed
preprocessing median can exceed the median of an E2E distribution.

## Individual Processor results

Representative Large transform medians from the three-repetition full run:

| Processor            | Scenario                 | Classification | Regression |
| -------------------- | ------------------------ | -------------: | ---------: |
| `StandardScale`      | baseline                 |       2.107 ms |   2.179 ms |
| `HardClip`           | baseline                 |       1.261 ms |   1.085 ms |
| `Power`              | baseline                 |     141.216 ms | 135.644 ms |
| `SigmaClip`          | baseline                 |      11.982 ms |  10.830 ms |
| `FeaturePermute`     | baseline                 |       3.161 ms |   2.435 ms |
| `CategoryShuffle`    | baseline                 |       0.992 ms |        n/a |
| `CategoricalAlign`   | categorical              |       8.301 ms |   9.890 ms |
| `SoftmaxTemperature` | 10-column numerical head |       2.640 ms |   2.471 ms |

For `CategoricalAlign` with ten categorical columns and 50,000 rows, fitted on
40,000 rows, classification medians were 7.121 ms fit, 8.301 ms transform,
and 14.340 ms fit_transform. Corresponding regression-labelled workload
medians were 7.171, 9.890, and 17.591 ms; feature processing itself is
task-independent and the difference is run variance.

## Factor sweeps and correctness

Every recorded workload has `correctness_status="pass"`. Assertions include
finite final output, expected class/quantile width, retained row count, and
unknown query categories becoming `-1` immediately after vocabulary
alignment. Dataset creation and these assertions were not timed.

Across all Large Cartesian combinations, measured total-Recipe medians were:

| Task           |                                                            Minimum |                                                       Maximum |
| -------------- | -----------------------------------------------------------------: | ------------------------------------------------------------: |
| Classification | 227.857 ms (`constant+sigma_outlier+categorical+unknown_category`) |                                  413.335 ms (`sigma_outlier`) |
| Regression     |  256.927 ms (`sigma_outlier+categorical+missing+unknown_category`) | 693.775 ms (`hard_outlier+sigma_outlier+categorical+missing`) |

Three repetitions make the full matrix suitable for regression detection and
factor ranking, not fine-grained micro-optimization. The ten-repetition Large
baseline should be used for headline comparisons.

## Artifacts

- `tabiclv2_processing_full.json` / `.csv`: 1,740 SDM result rows, including
  the complete 512-workload Cartesian matrix.
- `tabiclv2_processing_large_baseline.json` / `.csv`: ten-repetition SDM
  headline.
- `tabiclv2_reference_large_baseline.json` / `.csv`: ten-repetition pinned
  reference headline.

All JSON artifacts include the exact reference commit, system metadata,
dataset sizes, operation, task, Processor/Recipe, characteristics, median,
p95, standard deviation, memory field, device, dtype, repetitions, and
correctness status.
