# Preprocessing compilation investigation

**Status: active.** Experimental branches now compile useful public prediction and fitting paths. They are not merged support, and GPU validation is pending AWS authentication.

Baseline main: `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. All results below use actual CPU Inductor, not the eager tracing backend. Eager and compiled comparisons use the same data, weights and datatype. Prediction tolerances remain atol=1e-5, rtol=1e-4.

## What is being compiled

- `torch.compile(model.predict)` after eager `fit`: fitted preprocessing, relational graph preparation, internal model computation and output processing.
- `torch.compile(model.fit)`: recipe setup, learning preprocessing state and context/model cache preparation. Prediction afterward checks the fitted result.
- `torch.compile(model)`: the outer `forward` call, including fresh preprocessing fitting and uncached context/query computation. It does not automatically compile the separate fit/predict methods.

## Current public-call results

| Call / workload | PyTorch 2.14 CPU | PyTorch 2.7.1 CPU |
|---|---|---|
| Tabular predict, numerical classification, 1/4 estimators | Both graph policies pass | Public metadata/setup compatibility still under investigation |
| Tabular predict, numerical regression | Both policies pass the initial small-query cases; larger query sets expose a few strict-tolerance misses already present with internal-only compilation | Not established |
| Relational predict, one-hop real RelBench | Both policies pass query rows 4→1→8→4; max error 4.83e-6 | Schema guard fixes are advancing the call; no complete pass claimed |
| Relational predict, two-hop real RelBench | Both policies execute; one of eight probabilities narrowly exceeds tolerance in one query; internal-only compilation reproduces it | Not established |
| Tabular fit, classification | Partial compilation passes, including 4 estimators and 31 query rows; max prediction difference 5.36e-7 | Prebuilt recipe plus narrow fixes advances past preprocessing; further metadata guards remain |
| Tabular fit, regression | Partial compilation passes the tested case | Still under investigation |
| Tabular outer forward, classification/regression | Partial compilation passes; fullgraph still encounters learned vocabulary decisions / explicit random generators | Default calls fail in recipe setup, before neural computation |
| Relational fit / outer forward | String vocabulary fitting still being integrated and validated | Not established |

Relational **fullgraph requires** experimental external-join operations, a finite specified hop count and scoped dynamic-output capture:

```python
model.fit(context, target, related_context, num_hops=1)
with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
    predict = torch.compile(model.predict, fullgraph=True, dynamic=True)
    result = predict(query, related_query)
```

Arrow/cuDF joins are opaque operations inside the graph. Their existing algorithms execute at runtime; Inductor does not optimize Arrow. Small runtime validation operations preserve invalid-input errors. This distinction matters when assessing performance.

Across relational query sizes 4→1→8→4, fullgraph counts are 1→2→3→3. The repeated query reuses its graph; `dynamic=True` does not establish universal reuse across new sizes. Partial compilation can hit the default recompilation limit and fall back to eager execution for affected functions.

## Numerical quality

The two-hop relational example changes one probability from 0.0907173976 eager to 0.0906983167 with only the internal model compiled. Compiling public prediction gives 0.0906982943. Predicted classes are unchanged. Fitted preprocessing independently returns bitwise-equal values for all 12 checked numerical/datetime blocks. Scalar CPU code generation passes the original tolerance in this experiment, but is not a recommended speed fix.

On the larger Tabular regression checks, whole-prediction and internal-only compiled outputs are bitwise identical. Strict eager tolerance failures affect 1/110,889 quantile values with one estimator and 3/110,889 with four estimators. Four-estimator median-prediction RMSE changes from 64.8680573 to 64.8680649, and MAE from 53.9904861 to 53.9904900. These are small measured differences, not a claim of exact parity. No tolerance was relaxed.

Compiled preprocessing fitting has a separate issue: reduction rounding can change the PowerTransform search result. A focused experimental patch preserves native nansum/nanmean during fitting and native expm1 where needed. It retains datatype, optimizer and random-generator behavior. Real numerical data with missing values passes the existing tolerance on both runtimes; fitted lambda/mean/scale are exact in tested cases. Cached transformation is unchanged. GPU cost remains unmeasured, and CPU 2.7 can be slower with this precision-preserving path.

## Measured CPU benefit

These shared-host FP32 measurements compare the same eagerly fitted state, using nine rotating-order warm repetitions. Compilation time is excluded and recorded separately in the experiment. They are not GPU estimates.

| Classification workload | Eager | Internal model compiled | Public predict compiled |
|---|---:|---:|---:|
| 1 estimator, 32 query rows | 17.52 ms | 15.90 ms | 14.77 ms |
| 1 estimator, 128 rows | 57.33 ms | 56.45 ms | 55.18 ms |
| 4 estimators, 32 rows | 68.55 ms | 63.01 ms | 58.54 ms |
| 4 estimators, 128 rows | 221.11 ms | 218.56 ms | 213.68 ms |

All classification comparisons pass. Public compilation adds about 7% over internal-only compilation at 32 rows and about 2% at 128 rows in these cases. First calls cost tens of seconds, and new shapes can trigger more compilation. Regression timings and every failed numerical comparison are preserved in the experiment report; noisy runs are identified explicitly.

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
| [experiment/kumotabular-predict-compile-cpu](https://github.com/NVIDIA/structured-data-models/tree/experiment/kumotabular-predict-compile-cpu) | Timing, graph counts, cold-call costs and exact prediction-quality evidence |

## Remaining work and rejected shortcuts

- Complete relational string-vocabulary fitting, including int32 offset promotion without assuming small data or changing category selection.
- Complete 2.7 public routing checks. A plain-PyTorch reproduction confirms an inference-context bytecode bug; moving metadata updates outside that context avoids it. Frozenset/enum metadata and further schema guards remain distinct issues.
- Preserve explicit generator state in compiled fitting. Removing the generator, retaining unobserved categories or bypassing all fitting would change behavior and is not accepted.
- Validate custom operations on CUDA/cuDF, memory use and warm/cold latency. AWS login is pending; no Spot instance is currently running.
- Reconcile zero-copy string-layout metadata with mutation writeback; both require preserving aliasing rather than copying all payloads or permitting ambiguous overlapping writes.
- Keep numerical precision limits separate from preprocessing compatibility. A successful graph capture alone does not establish speed or prediction parity.

Rejected approaches include a read-only replacement for the public columns dictionary (broke a passing 2.14 path), unchecked cached applicability metadata (could become stale), and sequential overlapping table copies (could overwrite source values).

Independent regression checks at integration snapshot `09e934174` passed 105 categorical/string/variable-length/join tests on each runtime, with 45 CUDA skips each. Component branches record subsequent focused and broader tests. These CPU checks do not substitute for GPU validation.
