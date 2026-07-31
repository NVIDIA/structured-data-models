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

The canonical relational dataset has 5k context + 1k query task rows and `customers`, `orders`, and `products` tables. It includes primary and foreign IDs, datetimes, numerical and categorical features, missing values, constant features, outliers, and unseen query categories. Exact tables, columns, blocks, codes, and category metadata are compared before timing.

| Task / device        | Ensemble-shared median / p95 | Member-isolated median / p95 | Shared speedup | Shared summed kernels |  Shared peak delta |
| -------------------- | ---------------------------: | ---------------------------: | -------------: | --------------------: | -----------------: |
| Classification / CPU |             266.4 / 293.4 ms |           1183.3 / 1275.0 ms |          4.44× |                     — |     sampled 28 KiB |
| Regression / CPU     |             253.4 / 269.1 ms |           1157.5 / 1232.7 ms |          4.57× |                     — | allocator-retained |
| Classification / L4  |             265.1 / 273.8 ms |           1366.7 / 1461.6 ms |          5.15× |              37.03 ms |           5.58 MiB |
| Regression / L4      |             295.9 / 299.7 ms |           1303.3 / 1407.3 ms |          4.40× |              36.11 ms |           5.53 MiB |

The L4 is not faster than CPU for this mixed workload because cuDF is unavailable: string sorting and relational joins fall back to CPU and force synchronization. The low kernel-to-wall ratio confirms CPU fallback/orchestration, not GPU compute, is the dominant RFM limit. The benchmark still demonstrates the intended reuse benefit independently of device.

## Recommendation and limitations

- Keep the provenance-aware `EnsembleTable`, structural `EnsembleProcessor` nodes, normal-Processor adapter, and variable-schema mixin: each represents a current design invariant and has direct test coverage.
- The next Processor optimization target is `PowerTransform.fit`; a custom kernel is justified only if standard PyTorch compilation/fusion cannot reduce its measured 33 ms latency.
- RFM GPU improvement requires eliminating optional CPU string/join fallbacks, not additional ensemble abstractions.
- Serialization of dynamically fitted ensemble trees and decisions remains an explicit design open question.
- The current sequential model schedule does not provide a Recipe-level preprocessing fallback. Workloads beyond the measured preprocessing bracket require fewer rows or members until a chunked or sequential Recipe execution path is implemented.
- No cross-table fitted-state sharing, content hashing, multi-GPU execution, row-changing Processor support, or speculative schema wrapper was added.

## Reproduce

```bash
uv run python -m benchmark.tabiclv2_ensemble --devices cuda --tasks classification regression --context-rows 40000 --query-rows 10000 --features 100 --categorical-features 10 --vocabulary-size 4096 --num-estimators 8 --repetitions 20 --warmups 5 --include-processors --include-transfers --profile-kernels --output /tmp/tabiclv2_gpu.json
/usr/bin/time -v -o /tmp/tabiclv2_cpu_time.txt uv run python -m benchmark.tabiclv2_ensemble --devices cpu --tasks classification regression --context-rows 40000 --query-rows 10000 --features 100 --categorical-features 10 --vocabulary-size 4096 --num-estimators 8 --repetitions 20 --warmups 5 --include-processors --output /tmp/tabiclv2_cpu.json
uv run python -m benchmark.rfm_ensemble --devices cpu cuda --tasks classification regression --schedules ensemble_shared member_isolated --context-rows 5000 --query-rows 1000 --products 1000 --num-estimators 8 --repetitions 20 --warmups 5 --profile-kernels --output /tmp/rfm.json
uv run python -m benchmark.tabiclv2_preprocessing_memory_limit --task classification --start-rows 100000 --max-rows 5000000 --resolution 50000 --output /tmp/tabiclv2_preprocessing_l4_classification.json
uv run python -m benchmark.tabiclv2_preprocessing_memory_limit --task regression --start-rows 100000 --max-rows 5000000 --resolution 50000 --output /tmp/tabiclv2_preprocessing_l4_regression.json
```
