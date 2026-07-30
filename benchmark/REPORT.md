# Ensemble-Aware Processing Performance Report

## Outcome

On the NVIDIA L4, the best measured valid eight-member TabICLv2 implementation processes 40k context + 10k query rows with 100 features in **125.8–131.2 ms Recipe median** and **145.5–146.6 ms public-boundary median**. This meets the discussed 190–200 ms goal.

These measurements are an empirical production baseline, not a theoretical hardware speed of light. `recipe_total` is the fastest complete valid plan measured here; no independent minimum-computation CUDA baseline was implemented, so the 10.9–16.5% public-boundary difference must not be interpreted as a hardware lower-bound gap.

## Environment and method

- Hardware: NVIDIA L4 with 23,659,151,360 bytes device memory; AMD EPYC 7R13 host.
- Software: PyTorch 2.13.0+cu130, CUDA 13.0, float32.
- TabICLv2 workload: 40k context + 10k query rows, 100 features (90 numerical, 10 categorical), categorical vocabulary 4096, missing values, one constant feature, hard/sigma outliers, and an unseen query category.
- Ensemble: eight estimators, four identity normalization and four power normalization, with reference feature/class permutations.
- Headline GPU protocol: five warm-ups and 20 repetitions. CPU, Processor, transfer, RFM, and real-model protocols use two warm-ups and five repetitions; for five samples, nearest-rank p95 is the maximum and should be treated as a tail indicator, not a stable percentile estimate.
- CUDA inputs are resident before processing timing. CUDA events wrap the complete operation, explicit synchronization bounds each measurement, and event values are not summed kernel time.
- Dataset construction, correctness checks, and transfers are excluded from processing intervals. H2D and D2H transfers are measured separately.
- GPU memory is PyTorch peak allocated memory relative to the pre-call baseline. CPU memory is sampled maximum-RSS growth plus absolute process maximum RSS; warmed sampled growth can understate allocator-retained memory, so both are reported.

## TabICLv2 50k results

| Task           | CPU Recipe median / p95 | GPU Recipe median / p95 | GPU public parallel median / p95 | GPU public CUDA-event median | GPU public peak delta / absolute |     Throughput | Public vs Recipe |
| -------------- | ----------------------: | ----------------------: | -------------------------------: | ---------------------------: | -------------------------------: | -------------: | ---------------: |
| Classification |      2323.1 / 2327.8 ms |        131.2 / 135.2 ms |                 145.5 / 147.6 ms |                     145.3 ms |                307.2 / 327.4 MiB | 343,734 rows/s |           +10.9% |
| Regression     |      2759.2 / 2764.0 ms |        125.8 / 193.9 ms |                 146.6 / 220.0 ms |                     146.4 ms |              1372.8 / 1431.1 MiB | 341,151 rows/s |           +16.5% |

GPU Recipe execution is **17.7×** faster than CPU for classification and **21.9×** for regression. CPU public-boundary peak memory is 937.8 MiB absolute / 338.4 MiB sampled growth for classification and 2.64 GiB absolute / 1.28 GiB sampled growth for regression.

Regression output-transform p95 exceeded its median. Median and p95 are retained rather than hiding this variance.

### Stage timings on L4

| Task                        | Fit + context transform | Query transform | Output transform |  Complete Recipe |
| --------------------------- | ----------------------: | --------------: | ---------------: | ---------------: |
| Classification median / p95 |        101.6 / 140.8 ms |  28.4 / 28.9 ms |     1.6 / 2.1 ms | 131.2 / 135.2 ms |
| Regression median / p95     |          87.3 / 87.8 ms |  28.3 / 29.0 ms |   13.0 / 13.1 ms | 125.8 / 193.9 ms |

Each row is measured independently, so stage medians do not necessarily sum to the complete-Recipe median. Fit/context work accounts for about 77% of classification and 69% of regression median Recipe time.

### Processor timings on L4

| Processor operation                 |   Median |      p95 | CUDA-event interval | Peak allocation |
| ----------------------------------- | -------: | -------: | ------------------: | --------------: |
| `PowerTransform.fit`                | 32.56 ms | 32.66 ms |            32.43 ms |        81.5 MiB |
| `PowerTransform.fit_transform`      | 33.54 ms | 33.85 ms |            33.42 ms |        81.5 MiB |
| `AlignCategories.fit_transform`     |  4.78 ms |  4.85 ms |             4.64 ms |         4.6 MiB |
| `AlignCategories.transform`         |  6.08 ms |  6.25 ms |             5.95 ms |         1.1 MiB |
| `PowerTransform.inverse_transform`  |  1.98 ms |  2.04 ms |             1.86 ms |       142.6 MiB |
| `ClipSigma.fit_transform`           |  0.97 ms |  1.03 ms |             0.86 ms |        54.3 MiB |
| `DropConstantColumns.fit_transform` |  0.78 ms |  1.05 ms |             0.65 ms |        13.6 MiB |
| `ImputeMean.fit_transform`          |  0.42 ms |  0.44 ms |             0.28 ms |        38.5 MiB |
| `Standardize.fit_transform`         |  0.37 ms |  0.40 ms |             0.26 ms |        27.2 MiB |
| `ShuffleColumns.fit_transform`      |  0.47 ms |  0.49 ms |             0.33 ms |        13.6 MiB |

### Transfer timings

| Task           | H2D median / p95 | D2H median / p95 |
| -------------- | ---------------: | ---------------: |
| Classification |   3.61 / 3.94 ms |   3.55 / 3.70 ms |
| Regression     |   3.37 / 3.49 ms |   3.53 / 7.62 ms |

## Parallel versus sequential model scheduling

Recipe sharing is identical in both schedules; the mode controls model-input materialization and model-core execution. At the zero-compute 50k boundary:

| Task           |  Parallel median / p95 / peak | Sequential median / p95 / peak | Parallel speedup |
| -------------- | ----------------------------: | -----------------------------: | ---------------: |
| Classification |  145.5 / 147.6 ms / 307.2 MiB |   145.2 / 148.4 ms / 292.7 MiB |            1.00× |
| Regression     | 146.6 / 220.0 ms / 1372.8 MiB |  150.3 / 220.9 ms / 1372.8 MiB |            1.03× |

For the real random-weight TabICLv2 architecture at 2400 context + 600 query rows, parallel execution takes 2400.2 ms and 12.91 GiB incremental memory; sequential execution takes 2395.4 ms and 1.62 GiB. Regression takes 2419.2 ms parallel and 2381.1 ms sequential. Parallel execution is **1.00×** as fast and uses **7.96×** the incremental memory, so sequential execution is the practical model-core schedule when memory matters. `auto` retains a tested correctness-preserving OOM retry with restored RNG state.

## L4 memory boundary

The controlled search used the real random-weight classification model, eight estimators, 100 features, a 4:1 context/query split, and 1000-row resolution. It establishes a tested bracket, not the exact row threshold.

| Schedule   | Rows | Context / query | Result                 |   Runtime |           Peak allocated | Peak reserved |
| ---------- | ---: | --------------: | ---------------------- | --------: | -----------------------: | ------------: |
| Parallel   | 3000 |      2400 / 600 | Largest tested success | 2573.3 ms |                13.14 GiB |     21.18 GiB |
| Parallel   | 4000 |      3200 / 800 | First tested OOM       |         — | 15.87 GiB before failure |     21.10 GiB |
| Sequential | 4000 |      3200 / 800 | Success                | 3286.1 ms |                 2.39 GiB |      4.14 GiB |

OOM attempts are caught, outputs are released, Python garbage collection runs, and the CUDA cache is emptied before the next attempt.

## Canonical RFM processing benchmark

The canonical relational dataset has 5k context + 1k query task rows and `customers`, `orders`, and `products` tables. It includes primary/foreign IDs, datetimes, numerical and categorical features, missing values, constants, outliers, and unseen query categories. `ensemble_shared` executes one eight-member plan; `member_isolated` executes the same table-scoped member/RNG choices one member at a time. Exact outputs, columns, tensor blocks, and categories are compared before timing.

| Task / device        | Ensemble-shared median / p95 | Member-isolated median / p95 | Speedup |                             Shared peak memory |
| -------------------- | ---------------------------: | ---------------------------: | ------: | ---------------------------------------------: |
| Classification / CPU |             304.5 / 318.6 ms |           1329.0 / 1389.0 ms |   4.36× | 641.8 MiB absolute RSS / 28 KiB sampled growth |
| Regression / CPU     |             284.7 / 292.3 ms |           1284.7 / 1362.4 ms |   4.51× |      642.2 MiB absolute RSS / 0 sampled growth |
| Classification / L4  |             310.9 / 316.1 ms |           1295.4 / 1381.2 ms |   4.17× |                            7.87 MiB allocation |
| Regression / L4      |             286.2 / 288.0 ms |           1247.5 / 1343.1 ms |   4.36× |                            7.71 MiB allocation |

The L4 advantage over CPU is modest for this mixed relational workload. This benchmark proves reusable-state/intermediate sharing and schedule equivalence; it does not isolate a fully GPU-native RFM hardware limit.

## Reproduce

```bash
uv run python -m benchmark.tabiclv2_ensemble --devices cpu cuda --include-processors --include-transfers --warmups 2 --repetitions 5 --output /tmp/tabiclv2_ensemble.json
uv run python -m benchmark.tabiclv2_ensemble --devices cuda --warmups 5 --repetitions 20 --output /tmp/tabiclv2_ensemble_headline.json
uv run python -m benchmark.tabiclv2_ensemble --devices cuda --context-rows 2400 --query-rows 600 --only-model --warmups 2 --repetitions 5 --output /tmp/tabiclv2_model.json
uv run python -m benchmark.rfm_ensemble --devices cpu cuda --warmups 2 --repetitions 5 --output /tmp/rfm_ensemble.json
uv run python -m benchmark.tabiclv2_memory_limit --output /tmp/tabiclv2_l4_memory_limit.json
```

## Recommendation

- Implemented quick win: preserve shared variants as views, fit deterministic prefixes once, compute each unique `Choice` branch once per table scope, split only on provenance or incompatible schema, and delay member materialization until required.
- Current bottleneck: `PowerTransform.fit` remains the largest individual GPU Processor at about 33–34 ms; the regression output path is the dominant peak-memory case.
- Medium effort: evaluate `torch.compile` or standard-operation fusion for Power fitting and reduce temporary materialization in the 999-quantile target decode/reduction path.
- Model scheduling: keep both parallel and sequential modes, with `auto` fallback; do not prefer full model stacking on L4 until it shows a latency benefit.
- Long term: consider a custom fused Power kernel only if standard PyTorch fusion cannot materially reduce the measured bottleneck.
- Rejected for the first implementation: content-based tensor deduplication, cross-table fitted-state sharing without equivalent scope proof, and eager full-model member stacking.

Realistically achievable on this L4 with the current maintainable implementation is **about 126–131 ms for complete Recipe work and 145–147 ms through the public zero-compute boundary**. The smallest change that closes most of the latest-`main` gap is the implemented provenance-aware sharing plus delayed materialization; a custom CUDA kernel is not required to meet the 190–200 ms target.
