# Ensemble-Aware Processing Performance Report

## Outcome

On an NVIDIA L4, the optimized eight-member TabICLv2 Recipe processes 40k context + 10k query rows with 100 features in **129.3 ms median for classification** and **123.0 ms for regression**. The measured processing path is below the earlier 190–200 ms reference, while the optimization target remains the fastest valid implementation rather than a fixed runtime.

These are practical production baselines, not theoretical hardware speed-of-light results. No independent minimum-computation CUDA implementation was added. In a preprocessing-only memory search, all eight members completed at **3.45 million total rows** and first failed at **3.50 million rows** on this 22.0 GiB L4. This boundary excludes the model forward pass and output postprocessing.

## Environment and method

- Hardware: NVIDIA L4 with 23,659,151,360 bytes device memory; AMD EPYC 7R13 host.
- Software: PyTorch 2.13.0+cu130, CUDA 13.0, float32.
- TabICLv2 data: 40k context + 10k query rows, 100 features (90 numerical and 10 categorical), vocabulary 4096, missing values, a constant feature, hard and sigma outliers, and an unseen query category.
- Ensemble: eight members with four identity and four power-normalized paths plus reference feature/class permutations.
- Timing: five warm-ups and 20 repetitions. Median and nearest-rank p95 are computed from synchronized wall time.
- CUDA timing: inputs are resident before processing; events enclose the operation; explicit synchronization follows each call. Enqueue time measures host call entry to return, and synchronization wait measures only the final explicit wait. A separate profiler pass reports summed self CUDA-kernel time; this may exceed wall time when kernels overlap.
- Memory: CUDA values are peak allocated bytes relative to the pre-call baseline plus absolute peak allocated bytes. CPU stage values use sampled RSS; a fresh external process measurement reached 2.01 GiB maximum RSS for the complete CPU matrix.
- Dataset construction, correctness checks, and external transfers are excluded from processing intervals. H2D and D2H are reported separately.

## TabICLv2 50k processing

| Task           | CPU Recipe median / p95 | L4 Recipe median / p95 | GPU speedup | L4 Recipe summed kernels | L4 Recipe peak delta / absolute | Recipe throughput |
| -------------- | ----------------------: | ---------------------: | ----------: | -----------------------: | ------------------------------: | ----------------: |
| Classification |      2282.3 / 2288.6 ms |       129.3 / 132.3 ms |       17.6× |                 103.9 ms |               292.7 / 313.0 MiB |    386,599 rows/s |
| Regression     |      2651.7 / 2863.5 ms |       123.0 / 126.7 ms |       21.6× |                 119.3 ms |               914.9 / 973.2 MiB |    406,615 rows/s |

### L4 Recipe stages

| Task                        | Fit + context transform | Query transform | Output transform |  Complete Recipe |
| --------------------------- | ----------------------: | --------------: | ---------------: | ---------------: |
| Classification median / p95 |          94.9 / 95.9 ms |  29.7 / 31.5 ms |   0.87 / 1.18 ms | 129.3 / 132.3 ms |
| Regression median / p95     |          85.5 / 86.6 ms |  29.9 / 31.0 ms | 10.08 / 10.13 ms | 123.0 / 126.7 ms |

| Task           | Recipe enqueue median | Final synchronization wait | CUDA-event median |
| -------------- | --------------------: | -------------------------: | ----------------: |
| Classification |             129.31 ms |                    0.02 ms |         129.31 ms |
| Regression     |             114.98 ms |                    8.10 ms |         122.96 ms |

### Dominant Processor timings on L4

| Operation                           |     Median / p95 | Enqueue / final sync wait | Summed kernels | Peak allocation |
| ----------------------------------- | ---------------: | ------------------------: | -------------: | --------------: |
| `PowerTransform.fit`                | 32.74 / 34.46 ms |           29.26 / 3.57 ms |       62.57 ms |        81.5 MiB |
| `PowerTransform.fit_transform`      | 33.63 / 35.32 ms |           29.53 / 4.07 ms |       65.38 ms |        81.5 MiB |
| `AlignCategories.fit_transform`     |   5.95 / 6.33 ms |            5.93 / 0.01 ms |        2.66 ms |         4.6 MiB |
| `AlignCategories.transform`         |   7.66 / 8.50 ms |            7.65 / 0.01 ms |        3.02 ms |         1.1 MiB |
| `PowerTransform.inverse_transform`  |   1.86 / 1.88 ms |            1.00 / 0.86 ms |        3.69 ms |       142.6 MiB |
| `ClipSigma.fit_transform`           |   1.04 / 1.48 ms |            0.95 / 0.09 ms |        1.61 ms |        54.3 MiB |
| `DropConstantColumns.fit_transform` |   0.70 / 0.74 ms |            0.69 / 0.01 ms |        0.58 ms |        13.6 MiB |
| `ShuffleColumns.fit_transform`      |   0.41 / 0.53 ms |            0.39 / 0.01 ms |        0.06 ms |        13.6 MiB |

`PowerTransform.fit` remains the dominant individual GPU Processor. Kernel time comes from a separate profiled execution and sums individual kernel durations, so it is not directly additive with wall time and may be larger when kernels overlap.

### Regression output materialization fix

Profiling exposed redundant stacking before and after the fitted target inverse. Passing the already stacked ensemble tensor directly through the proven-shared fitted state changed:

| Metric                   |     Before |     After |  Improvement |
| ------------------------ | ---------: | --------: | -----------: |
| Regression output median |   48.65 ms |  10.08 ms |        4.83× |
| Output peak allocation   | 1524.4 MiB | 914.6 MiB |       −40.0% |
| Complete Recipe median   |   217.4 ms |  123.0 ms | 43.4% faster |

The generic multiple-fitted-state inverse path remains for genuinely distinct target states; only the single proven-shared state avoids repacking.

### Transfers

| Task           | H2D median / p95 | D2H median / p95 |
| -------------- | ---------------: | ---------------: |
| Classification |   3.49 / 4.19 ms |   3.54 / 8.59 ms |
| Regression     |   3.41 / 3.86 ms |   3.53 / 3.84 ms |

## L4 preprocessing memory boundary

The controlled searches execute only `default_recipe()`: after input transfer, each attempt fits and transforms context features and target for eight members, transforms query features, and keeps all three `EnsembleTable` outputs alive through the peak-memory measurement. Dataset construction, transfers, model execution, and output postprocessing are excluded. The workload uses 100 float32 features, a 4:1 context/query split, four identity and four power-normalized members, exponential growth, and a final 50k-row resolution.

| Task           | Largest success |     Context / query | First OOM | Failing stage   | Success runtime | Success peak delta / absolute | Success peak reserved |
| -------------- | --------------: | ------------------: | --------: | --------------- | --------------: | ----------------------------: | --------------------: |
| Classification |  3,450,000 rows | 2,760,000 / 690,000 | 3,500,000 | `fit_transform` |          7.34 s |             19.35 / 20.65 GiB |             21.71 GiB |
| Regression     |  3,450,000 rows | 2,760,000 / 690,000 | 3,500,000 | `fit_transform` |          7.32 s |             19.35 / 20.65 GiB |             21.71 GiB |

The result establishes a **3.45–3.50 million-row bracket** for this exact shape, dtype, Recipe, and eight-member configuration, not a universal row limit. The existing `ensemble_mode="sequential"` affects model-core materialization only and therefore does not lower this preprocessing peak. Every attempt releases outputs and workload data, runs garbage collection, and empties the CUDA cache after success or OOM.

## Canonical RFM processing

The canonical relational dataset uses `customers`, `orders`, and `products` tables and includes primary and foreign IDs, datetimes, numerical and categorical features, missing values, constant features, outliers, and unseen query categories. Exact tables, columns, blocks, codes, and category metadata are compared before timing. The cuDF environment uses Python 3.12.3, cuDF 26.6.0, CuPy 14.1.1, pylibcudf 26.6.0, PyTorch 2.13.0+cu130, and CUDA 13.0.

### TabICLv2-aligned 50k task-row workload

This workload uses the same 40k context + 10k query task-row count as the TabICLv2 benchmark, eight estimators, and 1k products. RFM additionally processes 40k/10k customer rows, 80k/20k order rows, and 1k/1k product rows for context/query respectively.

| Task           | CPU with cuDF median / p95 | L4 with cuDF median / p95 | GPU speedup | L4 without cuDF median / p95 | cuDF median effect | L4 cuDF kernels | L4 cuDF peak delta |
| -------------- | -------------------------: | ------------------------: | ----------: | ---------------------------: | -----------------: | --------------: | -----------------: |
| Classification |           685.3 / 794.5 ms |          308.5 / 323.2 ms |       2.22× |             271.4 / 352.8 ms |       13.7% slower |        43.67 ms |          46.32 MiB |
| Regression     |           696.0 / 885.8 ms |          334.9 / 353.5 ms |       2.08× |             302.9 / 308.4 ms |       10.5% slower |        42.14 ms |          46.01 MiB |

The exact `kumo-ml` default cannot execute this shape: `RunMode.FAST` limits context and query to 1k rows each. Even non-FAST context is limited to 10k rows, so the 40k/10k reference comparison is intentionally reported only for SDM preprocessing.

### Six-thousand-row ensemble reuse workload

The original canonical workload uses 5k context + 1k query task rows, eight estimators, and 1k products.

| Task / device        | Ensemble-shared median / p95 | Member-isolated median / p95 | Shared speedup | Shared summed kernels |  Shared peak delta |
| -------------------- | ---------------------------: | ---------------------------: | -------------: | --------------------: | -----------------: |
| Classification / CPU |             298.2 / 330.5 ms |           1193.4 / 1290.3 ms |          4.00× |                     — |             28 KiB |
| Regression / CPU     |             254.6 / 312.8 ms |           1159.7 / 1250.6 ms |          4.55× |                     — | allocator-retained |
| Classification / L4  |             300.7 / 310.3 ms |           1680.1 / 1818.2 ms |          5.59× |              38.65 ms |           5.58 MiB |
| Regression / L4      |             334.4 / 340.1 ms |           1580.4 / 1738.2 ms |          4.73× |              37.18 ms |           5.53 MiB |

A matched Python 3.12/PyTorch 2.13 control without cuDF measured 265.3 ms classification and 297.7 ms regression for shared L4 execution. cuDF therefore made the shared medians 13.3% and 12.3% slower; member-isolated medians were 22.6% and 19.8% slower. CPU medians changed by only low single-digit percentages. For the 50k task-row workload, cuDF remained 13.7% and 10.5% slower on L4. cuDF removes the CPU fallback warnings, but the extra columnar conversion, launch, and synchronization overhead outweighs that benefit at both tested sizes. Shared kernel time increased by only 1.61 ms classification and 1.08 ms regression at 6k rows while wall time increased by 35–37 ms, locating most of the slowdown outside GPU kernels. cuDF should remain optional and should not be selected unconditionally for these shapes without a lower-overhead path or a demonstrated crossover.

### Original `kumo-ml` default reference

The reference uses exact commit `e978685d46478758a09e3e472ea2fde9b5c14957`, `kumo-api==0.92.0`, PyTorch 2.12.0+cu130, and the locally cached v2.1 checkpoints. Its public default is `RunMode.FAST`. Classification uses one estimator with all shuffle options disabled. Regression uses one estimator, quantile target normalization, median output, and all shuffle options disabled.

The same canonical raw frames are converted into the sampled `kumo-api.Context` representation expected by the original driver. Dataset creation and checkpoint loading are excluded. `Data.from_context` includes pandas-to-tensor conversion and H2D for the GPU case; public `predict` includes that conversion, model execution, and output materialization.

| Task           | Reference `Data.from_context` CPU median / p95 | Reference `Data.from_context` L4 median / p95 | Reference public `predict` CPU median / p95 | Reference public `predict` L4 median / p95 | Reference L4 summed kernels |
| -------------- | ---------------------------------------------: | --------------------------------------------: | ------------------------------------------: | -----------------------------------------: | --------------------------: |
| Classification |                                 23.4 / 25.5 ms |                                24.1 / 25.9 ms |                          7678.5 / 7910.3 ms |                           179.4 / 181.8 ms |                   215.17 ms |
| Regression     |                                 21.9 / 24.0 ms |                                22.9 / 24.9 ms |                          7894.9 / 7964.8 ms |                           182.8 / 186.7 ms |                   216.48 ms |

At the same 1k context + 1k query shape and one estimator, current SDM measures:

| Task           | SDM Recipe CPU median / p95 | SDM Recipe L4 median / p95 | SDM public forward CPU median / p95 | SDM public forward L4 median / p95 |
| -------------- | --------------------------: | -------------------------: | ----------------------------------: | ---------------------------------: |
| Classification |            128.9 / 137.7 ms |           188.6 / 202.9 ms |                  3097.3 / 3179.9 ms |                   305.1 / 314.1 ms |
| Regression     |            124.9 / 129.4 ms |           207.9 / 216.1 ms |                  2894.4 / 2953.1 ms |                   296.3 / 301.0 ms |

| Task           | Reference public CPU peak delta / absolute | SDM public CPU peak delta / absolute | Reference public L4 peak delta / absolute | SDM public L4 peak delta / absolute |
| -------------- | -----------------------------------------: | -----------------------------------: | ----------------------------------------: | ----------------------------------: |
| Classification |                         771.4 / 2090.8 MiB |                   187.5 / 1235.0 MiB |                         618.0 / 868.5 MiB |                   202.1 / 569.9 MiB |
| Regression     |                         770.7 / 2071.3 MiB |                   117.2 / 1270.7 MiB |                         617.3 / 867.7 MiB |                   202.4 / 570.3 MiB |

Observed default-path ratios are directional, not parity claims: SDM is 2.48× classification and 2.73× regression faster on CPU, while the original is 1.70× and 1.62× faster on L4. The APIs do not expose identical boundaries or outputs: SDM inputs are already device-resident, the original starts from pandas-backed sampled subgraphs, original regression returns the configured median, and current SDM returns 999 decoded quantiles. The preprocessing-only values are especially not equivalent because original feature encoding continues inside the model. A strict performance-parity claim requires aligning graph materialization, output semantics, and timed boundaries first.

## Recommendation and limitations

- Keep the provenance-aware `EnsembleTable`, structural `EnsembleProcessor` nodes, normal-Processor adapter, and variable-schema mixin: each represents a current design invariant and has direct test coverage.
- The next Processor optimization target is `PowerTransform.fit`; a custom kernel is justified only if standard PyTorch compilation/fusion cannot reduce its measured 33 ms latency.
- RFM GPU improvement requires reducing string/join conversion and orchestration overhead; merely installing cuDF is 10–23% slower in the tested GPU workloads.
- Serialization of dynamically fitted ensemble trees and decisions remains an explicit design open question.
- The current sequential model schedule does not provide a Recipe-level preprocessing fallback. Workloads beyond the measured preprocessing bracket require fewer rows or members until a chunked or sequential Recipe execution path is implemented.
- No cross-table fitted-state sharing, content hashing, multi-GPU execution, row-changing Processor support, or speculative schema wrapper was added.

## Reproduce

```bash
uv run python -m benchmark.tabiclv2_ensemble --devices cuda --tasks classification regression --context-rows 40000 --query-rows 10000 --features 100 --categorical-features 10 --vocabulary-size 4096 --num-estimators 8 --repetitions 20 --warmups 5 --include-processors --include-transfers --profile-kernels --output /tmp/tabiclv2_gpu.json
/usr/bin/time -v -o /tmp/tabiclv2_cpu_time.txt uv run python -m benchmark.tabiclv2_ensemble --devices cpu --tasks classification regression --context-rows 40000 --query-rows 10000 --features 100 --categorical-features 10 --vocabulary-size 4096 --num-estimators 8 --repetitions 20 --warmups 5 --include-processors --output /tmp/tabiclv2_cpu.json
uv run python -m benchmark.rfm_ensemble --devices cpu cuda --tasks classification regression --schedules ensemble_shared member_isolated --context-rows 5000 --query-rows 1000 --products 1000 --num-estimators 8 --repetitions 20 --warmups 5 --profile-kernels --output /tmp/rfm.json
uv run python -m benchmark.rfm_ensemble --devices cpu cuda --tasks classification regression --schedules ensemble_shared --context-rows 40000 --query-rows 10000 --products 1000 --num-estimators 8 --repetitions 20 --warmups 5 --profile-kernels --output /tmp/rfm_50k.json
python -m benchmark.rfm_kumo_reference --devices cpu cuda --tasks classification regression --stages data_from_context predict --context-rows 1000 --query-rows 1000 --products 1000 --repetitions 20 --warmups 5 --profile-kernels --output /tmp/rfm_kumo_reference.json
uv run python -m benchmark.tabiclv2_preprocessing_memory_limit --task classification --start-rows 100000 --max-rows 5000000 --resolution 50000 --output /tmp/tabiclv2_preprocessing_l4_classification.json
uv run python -m benchmark.tabiclv2_preprocessing_memory_limit --task regression --start-rows 100000 --max-rows 5000000 --resolution 50000 --output /tmp/tabiclv2_preprocessing_l4_regression.json
```
