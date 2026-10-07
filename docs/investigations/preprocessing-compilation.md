# Preprocessing compilation investigation

**Status: active.** Experimental branches now compile useful public prediction and fitting paths. They are not merged support, and CUDA validation is in progress on one automatically terminating L4 Spot instance. On 2.14/L4, Tabular public prediction passes both graph modes; the first relational fullgraph run fails in Inductor CUDA stream ordering. A resident-cache transfer fix is being tested.

Baseline main: `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. Results are labeled CPU or CUDA; successful compilation claims use actual Inductor unless explicitly marked as tracing diagnostics. Eager and compiled comparisons use the same data, weights and datatype. Prediction tolerances remain atol=1e-5, rtol=1e-4.

## What is being compiled

- `torch.compile(model.predict)` after eager `fit`: fitted preprocessing, relational graph preparation, internal model computation and output processing.
- `torch.compile(model.fit)`: recipe setup, learning preprocessing state and context/model cache preparation. Prediction afterward checks the fitted result.
- `torch.compile(model)`: the outer `forward` call, including fresh preprocessing fitting and uncached context/query computation. It does not automatically compile the separate fit/predict methods.

## Current public-call results

| Call / workload | PyTorch 2.14 CPU | PyTorch 2.7.1 CPU |
|---|---|---|
| Tabular predict, numerical classification | Both graph policies pass; 1/4 estimators tested | Partial compilation passes rows 4→7→3→4; max error 1.13e-6 |
| Tabular predict, numerical regression | Both policies pass the initial small-query cases; larger query sets expose a few strict-tolerance misses already present with internal-only compilation | Not established |
| Relational predict, one-hop real RelBench | Both policies pass query rows 4→1→8→4; max error 4.83e-6 | Partial compilation passes the same rows; max error 4.83e-6 |
| Relational predict, two-hop real RelBench | Both policies execute; some queries miss strict tolerance, also with internal-only compilation | Partial compilation executes; a one-row query misses strict tolerance, also with internal-only compilation |
| Tabular fit, classification | Partial compilation passes, including 4 estimators and 31 query rows; max prediction difference 5.36e-7 | Partial compilation passes with a prebuilt recipe; four estimators, max difference 4.77e-7 |
| Tabular fit, regression | Partial compilation passes the tested case | Still under investigation |
| Tabular outer forward, classification/regression | Partial compilation passes; fullgraph still encounters learned vocabulary decisions / explicit random generators | Classification partial compilation passes with a prebuilt recipe and isolated compiler cache; max difference 5.96e-8. Default recipe construction remains blocked |
| Relational fit | Partial compilation passes two estimators; max prediction difference 1.41e-5, random-generator state exact | Still fails in compiler resume/metadata handling with fresh caches; no support claim |
| Relational outer forward | Not established | Not established |

Relational **fullgraph requires** experimental external-join operations, a finite hop count at graph construction and scoped dynamic-output capture. Fitted prediction already reads the finite hop count cached by eager fitting, including when fitting inferred it; users do not necessarily need to specify a new limit. This example makes the tested one-hop setting explicit:

```python
model.fit(context, target, related_context, num_hops=1)
with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
    predict = torch.compile(model.predict, fullgraph=True, dynamic=True)
    result = predict(query, related_query)
```

Arrow/cuDF joins are opaque operations inside the graph. Their existing algorithms execute at runtime; Inductor does not optimize Arrow. Small runtime validation operations preserve invalid-input errors. This distinction matters when assessing performance.

Across relational query sizes 4→1→8→4, fullgraph counts are 1→2→3→3. The repeated query reuses its graph; `dynamic=True` does not establish universal reuse across new sizes. A longer stream of ten real 16-row queries exposes unnecessary specialization on neighborhood sizes: fullgraph reaches the default eight-graph limit and fails on the ninth case in a tracing diagnostic. Three causes were isolated: reconstruction from concrete wrapper sizes, hashing symbolic block sizes during ensemble packing, and the native COO-to-CSR operator requiring a concrete node count. On source `6df9aeab5`, actual 2.14 CPU Inductor completes all ten neighborhoods with graph counts 1→2→2→2→2→2→2→2→2→2, at the default limit and with isolated caches. All 160 predicted classes match eager. Three of ten cases still miss the unchanged strict probability tolerance (maximum error 6.25e-5); this shape fix is not a numerical-parity claim. Partial compilation can instead fall back to eager for affected functions; raising the limit alone is not a fix.

## Numerical quality

The two-hop relational example changes one probability from 0.0907173976 eager to 0.0906983167 with only the internal model compiled. Compiling public prediction gives 0.0906982943. Predicted classes are unchanged. Fitted preprocessing independently returns bitwise-equal values for all 12 checked numerical/datetime blocks. Scalar CPU code generation passes the original tolerance in this experiment, but is not a recommended speed fix.

On the larger Tabular regression checks, whole-prediction and internal-only compiled outputs are bitwise identical. Strict eager tolerance failures affect 1/110,889 quantile values with one estimator and 3/110,889 with four estimators. Four-estimator median-prediction RMSE changes from 64.8680573 to 64.8680649, and MAE from 53.9904861 to 53.9904900. These are small measured differences, not a claim of exact parity. No tolerance was relaxed.

Compiled preprocessing fitting has a separate issue: reduction rounding can change the PowerTransform search result. A focused experimental patch preserves native nansum/nanmean during fitting and native expm1 where needed. It retains datatype, optimizer and random-generator behavior. Real numerical data with missing values passes the existing tolerance on both runtimes; fitted lambda/mean/scale are exact in tested cases. Cached transformation is unchanged. GPU cost remains unmeasured, and CPU 2.7 can be slower with this precision-preserving path.

A separate, minimal Linear candidate preserves functional `bmm` instead of substituting `matmul` for the compiled strided-output path. This eliminates a proven source-level rounding difference: tracing with the eager backend becomes bitwise equal through all 14 inspected neural layers. Actual Inductor still changes arithmetic elsewhere. Across four real queries, three pass strict tolerance on each runtime, but the failing query differs between runtimes. This candidate is not a blanket numerical-parity fix; see [the isolated branch](https://github.com/NVIDIA/structured-data-models/tree/compile/preserve-linear-bmm).

A diagnostic combines that Linear correction with native LayerNorm calls inside opaque compiler operations. All four checked real relational queries pass the original tolerance on both runtimes; 2.14 is bitwise equal in all four, while 2.7 has maximum differences up to 8.14e-6. Alternating CPU 2.14 timing for one 16-row query is 46.16 ms eager, 41.19 ms with generated normalization and 42.47 ms with native normalization. The native boundary costs about 3% versus that compiled variant, while remaining about 8% faster than eager. The combined dynamic-shape/native-normalization diagnostic on source `d8e17e00e` subsequently passes all ten real neighborhoods with two full graphs and maximum error 2.38e-7, where the default generated arithmetic missed three cases. This remains a CPU diagnostic, not yet a CUDA recommendation or an integrated production change. [Precision experiments](https://github.com/NVIDIA/structured-data-models/tree/experiment/relational-compile-precision) preserve the exact cases and failures.

## Measured CPU benefit

These shared-host FP32 measurements compare the same eagerly fitted state, using nine rotating-order warm repetitions. Compilation time is excluded and recorded separately in the experiment. They are not GPU estimates.

| Classification workload | Eager | Internal model compiled | Public predict compiled |
|---|---:|---:|---:|
| 1 estimator, 32 query rows | 17.52 ms | 15.90 ms | 14.77 ms |
| 1 estimator, 128 rows | 57.33 ms | 56.45 ms | 55.18 ms |
| 4 estimators, 32 rows | 68.55 ms | 63.01 ms | 58.54 ms |
| 4 estimators, 128 rows | 221.11 ms | 218.56 ms | 213.68 ms |

All classification comparisons pass. Public compilation adds about 7% over internal-only compilation at 32 rows and about 2% at 128 rows in these cases. First calls cost tens of seconds, and new shapes can trigger more compilation. Regression timings and every failed numerical comparison are preserved in the experiment report; noisy runs are identified explicitly.

Repeated whole-fit validation uses the same compiled callable with contexts of 32→48→32→32→32 rows, including different real rows of the same size. Predictions and generator states pass on both runtimes, but new compiled regions still appear on the last calls. On 2.14 those last fits take 176/106 ms versus 60/37 ms eager; on 2.7, 303/194 ms versus 70/52 ms. These are diagnostic shared-host timings, not a GPU estimate. They do not support recommending whole-fit compilation for small contexts.

## Branches and implementation scope

These branches overlap. Integration branches contain their prerequisites; they are not independent patches to merge blindly. Each component report identifies source commits and tested dependencies. No PRs were opened.

| Branch | Purpose |
|---|---|
| [compile/public-preprocessing-investigation](https://github.com/NVIDIA/structured-data-models/tree/compile/public-preprocessing-investigation) | Combined prediction implementation, real-data probes and current public-call results |
| [compile/fitted-recipe-boundaries](https://github.com/NVIDIA/structured-data-models/tree/compile/fitted-recipe-boundaries) | Combined fitting experiments, buffer registration and preserved random-state checks |
| [compile/outer-forward-investigation](https://github.com/NVIDIA/structured-data-models/tree/compile/outer-forward-investigation) | Combined outer-forward validation with exact errors on both runtimes |
| [compile/numerical-fit-parity](https://github.com/NVIDIA/structured-data-models/tree/compile/numerical-fit-parity) | Native fitting arithmetic where compiled differences affect learned parameters |
| [relational-dynamic-join-compile](https://github.com/NVIDIA/structured-data-models/tree/relational-dynamic-join-compile) | Variable-output identifier joins, runtime readout validation and bounded task propagation |
| [compile/categorical-recipe-support](https://github.com/NVIDIA/structured-data-models/tree/compile/categorical-recipe-support) | Traceable category dictionaries and fixed-output string lookup |
| [compile/categorical-fit-support](https://github.com/NVIDIA/structured-data-models/tree/compile/categorical-fit-support) | Vocabulary fitting and opaque external string sorting; remaining limits documented |
| [compile/atomic-tensor-construction](https://github.com/NVIDIA/structured-data-models/tree/compile/atomic-tensor-construction) | Finish tensor-wrapper construction before tracing resumes after graph breaks |
| [compile/table-input-mutation](https://github.com/NVIDIA/structured-data-models/tree/compile/table-input-mutation) | Compiler writeback for matching encoded storage; rejects unsafe overlap before writes |
| [compile/table-schema-guards](https://github.com/NVIDIA/structured-data-models/tree/compile/table-schema-guards) | Internal immutable schema metadata, preserving the public columns dictionary API |
| [compile/varlen-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/varlen-preprocessing) | Variable-length tensor metadata and dispatch-compatible writes |
| [compile/tabletensor-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/tabletensor-preprocessing) | Table/columnar flatten-unflatten and view support |
| [compile/nested-tensor-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/nested-tensor-preprocessing) | Categorical/string tensor tracing foundations |
| [compile/recipe-preparation](https://github.com/NVIDIA/structured-data-models/tree/compile/recipe-preparation) | Recipe construction and packing foundations |
| [compile/cache-stream-events](https://github.com/NVIDIA/structured-data-models/tree/compile/cache-stream-events) | Two-line equivalent CUDA event synchronization; 2.14 L4 relational fullgraph prediction passes, offloaded-cache validation pending |
| [compile/dynamic-table-neighborhoods](https://github.com/NVIDIA/structured-data-models/tree/compile/dynamic-table-neighborhoods) | Preserve symbolic table sizes and shape-compatible ensemble packing; keep the native CSR kernel behind a symbolic-size operator |
| [compile/preserve-linear-bmm](https://github.com/NVIDIA/structured-data-models/tree/compile/preserve-linear-bmm) | Preserve eager batched-matmul arithmetic in the compiled strided-output path; residual Inductor differences remain |
| [compile/tensor-subclass-cache](https://github.com/NVIDIA/structured-data-models/tree/compile/tensor-subclass-cache) | Experimental 2.14 persistent-cache metadata hashing; alias coverage under review, and a separate upstream plain-Tensor AOT cache mutation bug remains |
| [experiment/kumotabular-predict-compile-cpu](https://github.com/NVIDIA/structured-data-models/tree/experiment/kumotabular-predict-compile-cpu) | Timing, graph counts, cold-call costs and exact prediction-quality evidence |

## Initial CUDA result

On an L4 Spot GPU, PyTorch 2.14.0+cu130, source `c75ba3af6`, Tabular classification public prediction passes both graph policies on query rows 32→128→32, with FP32 weights and autocast disabled. Maximum difference is 1.52e-6. With 32 context rows and 128 query rows, warm wall latency is approximately 17.0 ms eager versus 4.3 ms compiled; peak PyTorch allocated tensor memory is 166.8 versus 158.6 MiB (additional live allocations 25.5 versus 17.3 MiB). These compare fully eager prediction with compiled public prediction, including neural operations; they do not isolate the benefit of compiling preprocessing. These are initial small-context measurements; larger-context, internal-only, alternating-order and BF16 validation are still running. PyTorch reserved memory is unchanged at 218 MiB. Relational cuDF/RMM memory is outside these PyTorch counters.

Relational public fullgraph compilation fails on the same runtime with `Argument '_tensor_constant9' of Node 'control_deps' was used before it has been defined`. A small PyTorch-only reproduction identifies a graph-ordering bug around CUDA stream waits. The preferred two-line candidate replaces `compute_stream.wait_stream(transfer_stream)` with its eager-equivalent `compute_stream.wait_event(transfer_stream.record_event())`. It preserves transfers and allocator lifetime tracking, while avoiding the faulty compiler transformation. The complete one-hop CUDA run passes rows 4→1→8→4 at the unchanged FP32 tolerance (maximum error 2.29e-6), without graph breaks, using three graphs. At eight query rows, warm latency is approximately 74.9 ms eager versus 41.0 ms compiled; peak PyTorch allocated memory changes only slightly, 142.19→141.86 MiB. This measurement excludes cuDF/RMM allocations. Multiple offloaded context caches and CUDA 2.7 remain to be checked. The optional resident-cache optimization is not required by this candidate.

## Suggested production changes

Do not merge the investigation history wholesale. Extract these groups with their focused regression checks:

1. Tensor wrapper support: expose constituent tensors, schema and view metadata to tracing.
2. Processor and ensemble routing: use those tensors/metadata during fitted transforms and prediction; preserve symbolic row sizes.
3. External categorical lookup: keep existing Arrow/cuDF string lookup behind a compiler-visible operation.
4. Relational graph preparation: variable-length identifier joins, validated readout, bounded traversal and a symbolic-size CSR adapter.

The relative-time/internal-model changes are separate from public preprocessing support. Numerical-fitting precision, generated neural rounding and persistent compilation caching are separate concerns; they should not be bundled into a compatibility patch without their own justification. Full integration still needs the CUDA and dynamic-path checks below.

## Remaining work and rejected shortcuts

- Finish symbolic-shape fixes and test long streams of realistic related-table neighborhoods without increasing the recompilation limit.
- Resolve or precisely bound generated neural arithmetic differences using the same dtype and unchanged tolerances. Preserving `bmm` fixes one source-level issue, not all Inductor rounding.
- Validate custom operations on CUDA/cuDF, memory use and warm/cold latency. One L4 Spot instance is running with automatic termination and explicit cleanup scheduled.
- Relational public fitting on 2.7 still encounters compiler resume failures. Classification fitting works with a prebuilt recipe; whole-fit speed is not established.
- Fullgraph fitting still needs support for data-dependent schema decisions and explicit random generators. Removing generators or changing learned categories would alter behavior and is not accepted.
- Complete the experimental cache-hash review before integration. A mutation/aliasing wrong-result reproduction also fails with plain PyTorch tensors and never calls the subclass hook; it is an upstream AOT cache issue. Adding storage topology to the subclass hash cannot fix a cache key that bypasses that hook. A conservative cache bypass is being validated.
- Review persistent compilation-cache behavior. PyTorch 2.14 supports a metadata-hash hook; 2.7 ignores it and can reuse incompatible artifacts after tensor-flattening code changes. Experiments use fresh, source-specific cache directories; no global cache deletion or production configuration change is required.
- Keep external Arrow/cuDF work explicit. An opaque operation lets a compiled graph invoke it; it does not turn that external implementation into fused PyTorch kernels.

Rejected approaches include a read-only replacement for the public columns dictionary (broke a passing 2.14 path), unchecked cached applicability metadata (could become stale), and sequential overlapping table copies (could overwrite source values).

Independent regression checks at integration snapshot `09e934174` passed 105 categorical/string/variable-length/join tests on each runtime, with 45 CUDA skips each. Component branches record subsequent focused and broader tests. These CPU checks do not substitute for GPU validation.
