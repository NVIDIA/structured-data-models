# Ensemble refactor: validation and performance

The implementation and full before/after matrix are local. All nine primary-run medians improve, but a longer paired E1 run still regresses about 2.2%. Lower memory consumption is not achieved: E16 GPU peaks increase about 0.9–1.7%. These are observed measurements, not a statistically significant or across-the-board performance guarantee.

## Preserved evidence

Artifacts are in `/home/rbendias/code/sdm-105/ensemble-refactor-797-results`:

- `baseline/`: immutable original nine-case measurements before library changes. Its original traces omitted CUDA activity and must not be used for GPU kernel claims.
- `exclusive-20261111/baseline/`, `exclusive-20261111/after/`: matched nine-case measurements with an otherwise idle GPU. Final source SHA256 is `9a5ffe73746dafc1cbaafbf2f41d24711d1ce6b238bafe92f78666e2bf5ea831`; fixture hashes match across all cases.
- `exclusive-20261111/profiles/baseline/` and `exclusive-20261111/after/*-trace.json`: two genuine CPU/CUDA traces per version. CUDA profiling is initialized before CPU-only profiling to avoid a confirmed PyTorch profiler issue.
- `exclusive-20261111/memory/`: separate before/after per-processor CPU/GPU peak measurements, without contaminating latency samples.
- `exclusive-20261111/stability-21/`: additional paired 21-repetition E1/E4 measurements. Earlier unfavorable results and intermediate after runs remain preserved separately.
- `exclusive-20261111/shared-physical-micro.json`: raw executor microbenchmark before/after the shared physical fast path; not a Kumo pipeline speedup claim.
- `data/`, `parity/`: identical fixed inputs and per-member baseline context, query and target outputs.
- `counters/`: earlier untimed allocation diagnostics; final counters are included in the exclusive before/after JSONs.
- `exclusive-20261111/{processors,operations,totals}.csv` and `SUMMARY.md`: complete per-processor/lifecycle timings, median/p95, peaks, allocations, copies, ATen dispatch counts and deltas for every case. Missing or uninvoked lifecycles remain empty, not zero.
- `after-shared-gpu-20261111/`: intermediate measurements, not final comparable results. They must not be presented as the refactor's speedup.
- `prediction-{cpu,cuda}-{baseline,after}.pt`: actual Kumo-Small classification/regression forward and fit/predict outputs, with identical non-constant weight hashes. These are numerical correctness checks, not pretrained-model quality results.

Hardware is an NVIDIA L4 (driver 595.71.05), AMD EPYC 7R13 host with four logical CPUs, Python 3.11.15 and PyTorch 2.14.0+cu130. Baseline uses two warmups, seven repetitions and four CPU threads. The baseline dataset and dependency commit hashes are recorded in each JSON.

## What is validated

- Complete nine-case preprocessing parity, including the 100,000-row case, before and after replacing the adapter architecture. The untimed counter pass independently rechecks these outputs after optimizations.
- Ordinary/ensemble lifecycle, independent member state, schema changes, compatible physical batches, shared results, ragged queries and regrouping across fitted groups.
- Logical tensor axes, global/method mean, arithmetic, tuple outputs, CPU scalar broadcasting, scalar dtype promotion, safe mutation and rejected conflicting writes into shared storage.
- Registered state replacement, persistence, device transfers, state dictionaries, deepcopy, and a safe numerical checkpoint loaded from CPU into a CUDA processor.
- Numerical, categorical, datetime and text processors; actual Kumo recipes, CPU/CUDA prediction parity and fullgraph prediction capture for classification/regression. Graph capture is not an Inductor performance measurement.
- A broad final integration run passed 781 tests (5 skipped, 21 medium/large cases deselected); a separate Tensor/Buffer/container run passed 244 (1 skipped). The skipped integration cases are three MPS tests and two caller-compiled power-optimizer tests. Test logs remain in `/tmp/ensemble-*.log`.
- Ordinary fitted tensors used by a physically batched ensemble query. A real AddCategoryCounts regression exposed missing leading dimensions; the shared executor now broadcasts ordinary logical tensors as views, and categorical sizes broadcast over rows explicitly.
- Individual members without a processor's semantic type remain unchanged, including inverse transforms. The existing shared contract covers heterogeneous schemas; a separate CUDA smoke checked all 36 non-reducer configurations. Empty quantile state, zero-width sign/inverse operations and projection no-ops are covered. Fixed Standardize statistics use the fitted row-broadcast shape; four actual Kumo-Forecast CPU/CUDA smoke cases passed.

## Demonstrated bottlenecks and changes

- Baseline host work includes table/metadata reconstruction, category alignment, power-transform optimization and column shuffling. The power optimizer still performs its original fixed 44-step search; no numerical shortcut was substituted.
- The executor retains the largest existing physical input, aligns small fitted statistics to it, reuses shared values, and uses views where possible. It separates Python metadata only where required instead of forcing everything into tensors.
- Fully shared logical and physical inputs execute directly without partition planning or vmap. Tuple outputs preserve sharing, and in-place/out operations retain their original destination. A tiny shared-mean microbenchmark reduces synchronized CUDA-call wall time from about 40 to 30 microseconds; this measures executor overhead, not Kumo latency.
- Tensor callbacks preserve vectorization for chunked numerical kernels. ClipSigma fits its two-stage bounds inside a batched tensor callback, avoiding repeated ensemble dispatch for each elementary statistic.
- ShuffleColumns gathers numerical values before reconstructing different column schemas, allowing a physical batched gather instead of one gather per member.
- FlipSign stores its broadcast shape during fitting, avoiding repeated per-group unsqueezes. Tensor in-place operators retain actual in-place semantics rather than silently allocating new results.
- Allocation diagnostics exposed a draft regression: replacing numerical blocks also aligned and copied the old numerical table. Direct `EnsembleTable.replace_blocks` now constructs the result from replacement and retained blocks only; empty blocks reuse views. This removes the measured extra full-table materialization. Exact allocation counts and byte comparisons are in `operations.csv` and `processors.csv`; cumulative allocated bytes must not be called peak memory.
- Quantile interpolation uses a single shared reference boundary when an expanded grid has identical rows, avoiding searchsorted's implicit contiguous copy.

For 10,000 × 500 at E16, physical ATen dispatches fell from 34,647 to 31,908 and allocation-producing CUDA events from 5,365 to 5,339. The draft block-replacement regression increased cumulative CUDA allocation volume from the baseline's 12.370 GB to 13.693 GB; direct replacement reduced it to 12.560 GB. This remains approximately 1.5% above baseline, not a memory improvement claim. It is cumulative temporary allocation volume, not peak live GPU memory. The remaining repeated categorical-block materialization is described below.

Every remaining fused override has a specific purpose: AlignCategories reuses category counts/remapping; Sequential avoids running intermediate transforms twice; StypeDispatch and Choice preserve child fusion while routing once; TaskDispatch and TableDispatch forward to the selected child's fused path. Other processors use the base implementation.

## Complete latency matrix

Total is actual recipe fit-transform plus query transform, not model inference. Positive speedup means faster. Fit, transform and unfused feature-fit medians/p95, and every processor callback, are reported separately in the CSVs.

| Rows × columns |   E | Before median / p95 ms | After median / p95 ms | Median speedup |
| -------------- | --: | ---------------------: | --------------------: | -------------: |
| 1,000 × 100    |  16 |        227.29 / 352.20 |       194.98 / 302.80 |         +14.2% |
| 1,000 × 500    |  16 |        527.78 / 587.07 |       503.55 / 507.78 |          +4.6% |
| 1,000 × 2,000  |  16 |    1,413.02 / 1,488.42 |   1,171.36 / 1,200.74 |         +17.1% |
| 10,000 × 500   |  16 |        752.66 / 815.47 |       724.74 / 728.09 |          +3.7% |
| 10,000 × 2,000 |  16 |    2,629.20 / 2,713.73 |   2,471.02 / 2,492.05 |          +6.0% |
| 100,000 × 100  |  16 |        798.76 / 943.61 |       791.51 / 910.11 |          +0.9% |
| 10,000 × 500   |   1 |        189.14 / 205.99 |       172.70 / 177.11 |          +8.7% |
| 10,000 × 500   |   4 |        473.35 / 594.56 |       454.58 / 562.91 |          +4.0% |
| 10,000 × 500   |   8 |        576.19 / 687.74 |       508.02 / 668.69 |         +11.8% |

| Rows × columns |   E | CPU RSS peak before / after GiB | GPU allocated peak before / after GiB |
| -------------- | --: | ------------------------------: | ------------------------------------: |
| 1,000 × 100    |  16 |                   1.643 / 1.637 |                         0.039 / 0.040 |
| 1,000 × 500    |  16 |                   1.648 / 1.648 |                         0.200 / 0.202 |
| 1,000 × 2,000  |  16 |                   1.680 / 1.676 |                         0.611 / 0.621 |
| 10,000 × 500   |  16 |                   1.723 / 1.723 |                         1.939 / 1.962 |
| 10,000 × 2,000 |  16 |                   1.853 / 1.849 |                         5.967 / 6.057 |
| 100,000 × 100  |  16 |                   1.760 / 1.760 |                         3.789 / 3.833 |
| 10,000 × 500   |   1 |                   1.634 / 1.635 |                         0.214 / 0.214 |
| 10,000 × 500   |   4 |                   1.719 / 1.720 |                         0.720 / 0.720 |
| 10,000 × 500   |   8 |                   1.721 / 1.721 |                         1.064 / 1.064 |

These are observed measurements, not significance claims. Seven repetitions give only a descriptive p95; substantial host-side variance is visible in raw samples and earlier runs. RSS is sampled, and GPU peaks are PyTorch allocated bytes, not NVML usage or reserved/RMM memory. Repeated-workflow GPU peaks retain the previous returned tables until replacement; see README for exact scope. Separate per-processor peaks exclude that previous-call retention and must not be substituted for the repeated-workflow total.

An additional paired check on 10,000 × 500 uses the same fixtures, hardware, two warmups and 21 repetitions per version. It does not replace the primary matrix or discard unfavorable results:

| E   | Before median / p95 ms | After median / p95 ms | Median speedup |
| --- | ---------------------: | --------------------: | -------------: |
| 1   |        171.36 / 219.74 |       175.10 / 200.73 |          −2.2% |
| 4   |        454.27 / 613.64 |       448.88 / 594.28 |          +1.2% |

This limits any E1 speedup claim: the result is sensitive to host variance, and the longer run still shows a small median regression. Before the shared physical fast path, the retained 21-repeat after results were 180.78 ms at E1 and 466.45 ms at E4. The optimization improves those observed medians without removing the E1 baseline gap.

The earlier 11 GiB Kumo-Forecast session was terminated only after explicit user permission. Original shared-GPU attempts and failed/aborted reruns are retained separately; they are not used above. The final matched suite completed all nine cases with matching fixture/source hashes and no foreign GPU process reported at case startup.

## Selective profiles and remaining bottlenecks

| Case              | CUDA kernels before / after | Summed kernel time before / after ms |
| ----------------- | --------------------------: | -----------------------------------: |
| 1,000 × 100, E16  |               6,149 / 6,088 |                      18.648 / 18.575 |
| 10,000 × 500, E16 |               8,793 / 8,732 |                    317.028 / 317.787 |

Kernel work is essentially unchanged; fewer host-side dispatches do not imply a comparable reduction in GPU arithmetic. On 10,000 × 500, category fitting/remapping, the unchanged power optimizer, robust quantiles, casts, column selection and shuffling remain expensive. Exclusive host time per pipeline for Cast(float32) fell from about 127 to 59 ms and SelectColumns from 58 to 10 ms, while category fit/remapping rose from 110 to 127 ms. These instrumented callback times are separate from the uninstrumented totals and include host variance.

Peak memory and cumulative allocation volume still regress slightly. The remaining repeated categorical block is materialized to align with the larger numerical batch before concatenation. Avoiding it requires a deliberate sharing/vectorization tradeoff; no padding, dynamic rebatching or unmeasured heuristic is hidden in this change. The lower-memory goal remains open.

## Limitations

Remaining engineering boundaries:

- PyTorch's global function type stubs still require Tensor arguments; runtime `__torch_function__` support does not change those stubs. Methods and the shared executor avoid casts at processor operation sites, but dynamic method delegation is not a fully statically typed Torch surface.
- Fullgraph Kumo prediction passes. This does not promise that every global Torch function accepting a plain ensemble container is compilable. PyTorch currently emits a warning when tracing the pure cached operator-metadata lookup.
- Categorical fitting with independently differing vocabularies uses logical EnsembleTable members. A regular higher-rank CategoricalTensor cannot represent per-batch vocabularies; AlignCategories now rejects fitting it rather than silently returning the first batch's result, as the previous ordinary path did.
- A pre-existing ordinary higher-rank DropConstantColumns limitation remains: a two-batch example with a constant first column retains both columns in both the frozen baseline and refactor. Independent schema selection is verified through EnsembleTable; this refactor does not claim to fix every existing ordinary batched-metadata operation.
- Regrouping can require materialization: combining a full numerical batch with repeated categorical route outputs currently stacks the smaller repeated block before concatenation. This can trade allocation bytes for fewer executions. No padding, dynamic rebatching or unmeasured packing heuristic was introduced.
- CPU string/TF-IDF vocabulary boundaries and column-name selection still need host metadata. Arbitrary mutation of aliased member storage is not allowed when different logical results would overwrite the same location.

## Remaining publication gate

The replacement target-PR plan is [ensemble-aware-data-design.md](../../ensemble-aware-data-design.md). The repository requires the user to read code before submission and approve exact GitHub text; neither unreviewed code nor a rewritten PR description has been published.
