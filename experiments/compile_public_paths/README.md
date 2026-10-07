# Compiling public model prediction

This experimental branch supports compiling tensor preprocessing and prediction together. It is based on main `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`; these are branch results, not current-main guarantees. The successful cases **fit eagerly, then compile and call `model.predict`**. No GPU speed or memory claim is made here.

## Current GPU limits

**General CUDA compilation support is not established.** The passing resident-cache FP32 case is narrower than the following configurations. Each numerical comparison uses the same precision mode for eager and compiled execution.

| L4 public prediction case | PyTorch | Observed result |
|---|---|---|
| Relational, one estimator, resident cache, FP32 | 2.14 | Fullgraph passes; max difference `2.30e-6` |
| Tabular, two estimators, CPU-offloaded caches, FP32 | 2.14 | Fullgraph executes, but probabilities are unstable: max difference `0.372`, all 256 probability values fail tolerance in the worst call. Stream/cache ordering is being investigated. |
| Tabular, resident cache, BF16 autocast | 2.14 | Empty-payload fix removes the tracing error, but observed predictions miss strict same-mode tolerance: max difference `0.00776`. Inner-versus-public comparison is being investigated. |
| Tabular, resident cache, FP32 | 2.7.1 | Fullgraph fails with `Graph break under GenericContextWrappingVariable` at `recipe_execution.transform(...)`. |

The first, second, and fourth runs use `c75ba3af6` plus event fix `18a4d151a`; the BF16 run additionally uses empty-payload fix `4071789e4`. These are **not GPU validation of the final combined source `e2abde154`**, whose run is queued. [The evidence snapshot](gpu-limit-snapshot.json) records source identifiers, observed samples, and raw-file hashes. The substantial offloaded-cache probability mismatch is unresolved; successful capture or class agreement must not be treated as prediction parity.

## Current results

Real cached data, pretrained weights, CPU Inductor, and the same FP32 weights/inputs for eager and compiled comparisons:

| Model / data | PyTorch | Graph breaks allowed | Fullgraph | Prediction parity |
|---|---|---|---|---|
| KumoTabular, numerical classification | 2.14 | Pass | Pass | Max absolute difference `8.94e-7` |
| KumoTabular, four estimators / numerical classification | 2.14 | Pass | Pass | Max `3.28e-7`; initial snapshot |
| KumoTabular, numerical regression | 2.14 | Pass | Pass | Max `1.84e-4`; initial snapshot |
| KumoRelational, one-hop RelBench / four related tables | 2.14 | Pass | Pass* | All query sizes pass; max `4.83e-6` |
| KumoRelational, two-hop RelBench / six related tables | 2.14 | Runs | Runs* | Some samples miss strict tolerance; inner-only compilation also reproduces drift |
| KumoTabular, numerical classification | 2.7.1 | Pass; partial capture | Blocked | All query sizes pass; max `1.13e-6` |
| KumoRelational, one-hop RelBench | 2.7.1 | Pass; partial capture | Blocked | All query sizes pass; max `4.83e-6` |
| KumoRelational, two-hop RelBench | 2.7.1 | Runs; partial capture | Blocked | One-row sample misses strict tolerance; inner-only compilation reproduces it |
| Public `fit()` / outer `model(...)` | Both | Separate investigation | Separate investigation | Not covered by the successful prediction cases |

*Relational fullgraph needs scoped `capture_dynamic_output_shape_ops=True` and an explicit finite `num_hops`. Arrow/cuDF joins remain opaque external operations inside the captured graph; their algorithms are not compiled.

The initial relational fullgraph checks run **4 → 1 → 8 → 4 query rows with their matching related neighborhoods** through one compiled callable. Captured graph counts are **1 → 2 → 3 → 3**; first graphs contain 5,714 calls for one hop and 8,076 for two hops. New sizes recompile despite `dynamic=True`; repeating the query reuses its graph. Tabular checks use **4 → 7 → 3 → 4**, with fresh tables and sliced views.

Relational fullgraph source: `d8cbc7383`. The latest tabular classifier repeat passes after atomic wrapper construction (`aa3876c33`). Initial tabular regression/four-estimator source: `284f612cf`. Exact comparisons use `atol=1e-5, rtol=1e-4`; they are not bitwise or dataset-wide quality guarantees. [Prediction results](relational-prediction-progress.json) preserve source commits, failures, graph counts, and exact probability values; [initial results](prediction-final.json) preserve the earlier tabular matrix.

A final independent-cache repeat at `450444af9` confirms both one-hop prediction successes and the small two-hop precision misses above. Each process uses its own cache directory with FX/AOT caches disabled; [fresh-validation.json](fresh-validation.json) records all settings and outcomes. This excludes stale artifacts from earlier subclass-layout experiments. Cross-version compilation-cache compatibility is still being reviewed separately.

## Latest integration checkpoint

Source `e2abde154` additionally includes three compatibility fixes:

- Empty ragged payload conversion avoids reading unnecessary tensor-valued slice bounds. This covers empty text blocks during output dtype conversion and actual empty/missing strings.
- Cache transfer dependencies use explicit CUDA events, which Dynamo can capture, while preserving stream ordering.
- The compiled CSR adapter makes strided indices contiguous before calling the native kernel. Its regression oracle now uses independent counts/cumulative sums: comparing only with the native operator had hidden its stride limitation. Normal relational graph construction already supplies contiguous sorted indices.

Focused checks on this combined source pass on both CPU runtimes: **98 passed, 13 skipped**, plus all **12 actual-Inductor empty-conversion combinations** (three input kinds × two modes × two versions, each with four row counts). The ten-neighborhood result above remains qualified to source `6df9aeab5`; it was not repeated for these targeted changes. The native-normalization and cache-hashing experiments remain separate ([checkpoint evidence](final-compatibility.json)).

A separate NVIDIA L4 run passed fullgraph FP32 public relational prediction for rows **4 → 1 → 8 → 4**, maximum difference `2.30e-6`. That GPU source was **`c75ba3af6` plus `18a4d151a`**, not this combined checkpoint. [The source receipt and GPU evidence](https://github.com/NVIDIA/structured-data-models/blob/compile/cache-stream-events/experiments/cache_stream_events/gpu_summary.json) retain the exact patch/data/result hashes. GPU validation of the final combined source is pending; these earlier GPU measurements must not be attributed to it.

## Usage

```python
model.fit(context, target, related_context, num_hops=1)
with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
    predict = torch.compile(model.predict, fullgraph=True, dynamic=True)
    result = predict(query, related_query)
```

For KumoTabular numerical prediction, the dynamic-output setting and related-table arguments are unnecessary. Omitting `fullgraph=True` allows graph breaks; it does not guarantee that every region gets compiled.

## What changed

- Tensor subclasses expose their tensor leaves and preserve layout during reconstruction and view replay. Wrapper construction stays atomic when execution resumes after a graph break.
- Recipe dispatch, column selection, ensemble packing, and query validation use traceable schema metadata. Fitting caches class output names so prediction avoids tensor-to-Python string conversion.
- Columnar wrappers reconstruct symbolic row sizes from tensor leaves; ensemble grouping compares shapes without hashing their concrete values. CSR conversion retains a symbolic node count through a custom operation that calls the same native kernel.
- String category lookup and identifier joins use custom operations that call the existing Arrow/cuDF implementation. Join outputs have a data-dependent length; this is why the scoped compiler setting is needed.
- Task graph validation remains active at runtime. With finite `num_hops`, propagation executes a bounded number of tensor updates; updates after an empty frontier are no-ops.
- Internal-model prerequisites are included: Fourier-buffer compilation, relational graph preparation, and relative-time processing. Chunk-size preservation is separate; compilation success alone does not establish memory parity.

The join prototype preserves tested duplicate, empty, missing, nullable, string, multikey, dtype, and validation-error behavior. Details and standalone commands are in [DYNAMIC_JOIN.md](DYNAMIC_JOIN.md).

## Precision and limitations

The two-hop probability miss also occurs with entirely eager preprocessing and **only the inner model compiled**: eager `0.0907173976`, inner-compiled `0.0906983167`, public-compiled `0.0906982943`. Predicted classes are unchanged. The public-versus-inner difference for that value is `2.24e-8`; the tolerance was not relaxed. Direct compiled preprocessing gives bitwise-equal values in all 12 nonempty numerical/datetime blocks across the task and six related tables ([results](relational-preprocessing-parity.json)).

Two numerical effects are now distinguished. The compiled inference branch in `Linear` substitutes `matmul` for eager `bmm` on rank-three inputs. Preserving functional `bmm` in a separate candidate restores bitwise agreement for all 14 captured neural outputs under the tracing-only backend. Actual Inductor still differs later in attention/normalization; the candidate changes which samples miss tolerance, so it is not integrated here as a complete parity fix. A separate CPU diagnostic using scalar code generation (`cpp.simdlen=1`) passes the original tolerance across all four two-hop query sizes. Both diagnostics keep FP32 unchanged. Neither is a speed recommendation; no precision flags were changed in the library.

A later isolated diagnostic combines functional `bmm` with a native LayerNorm custom operation. All ten real neighborhoods then pass the original tolerance, with maximum difference `2.38e-7` and the same two reusable full graphs. This keeps FP32 unchanged and prevents Inductor from replacing the normalization arithmetic. It is [documented on a separate branch](https://github.com/NVIDIA/structured-data-models/blob/compile/public-native-normalization-diagnostic/experiments/compile_public_paths/NATIVE_NORMALIZATION.md), since speed and memory effects from reduced fusion are not yet measured; this integration branch retains its original numerical paths.

PyTorch 2.7 public prediction now passes with graph breaks after traceable schema lookups, atomic wrapper construction, and atomic table dispatch. The dispatch boundary leaves tensor handlers eligible for compilation (`recursive=False`). It prevents Dynamo from incorrectly resuming an operator through another operator's wrapper. The initial one-hop sweep (`f47b96742`, recursive dispatch boundary) captures **71 → 118 → 131 → 131** graphs; tabular captures **24 → 45 → 65 → 66**. Default recompilation limits still cause eager fallbacks, and the repeated tabular query adds a graph. This is partial compilation with substantial fragmentation, not a speed recommendation. The latest symbolic-shape source also passes rows 4 → 8 → 4 with max `4.83e-6`, capturing 82 → 95 → 103 graphs; repeating the first input still adds graphs ([results](dynamic-onehop-27.json)).

An auditing backend delegates to real Inductor and saves every graph. The relational run includes **627 linear and 108 attention calls across recompilations**, confirming that neural computation is captured as well as preprocessing ([capture summary](relational-27-capture.json)). These are compile-time counts, not runtime operation counts. Reproduce with `audit_inductor.py --backend audited_inductor` and `SDM_GRAPH_DIRECTORY` set to a new output directory.

Fullgraph remains blocked for both models on 2.7: the reported generic-context error masks unsupported recipe-applicability tracing. A direct-context diagnostic exposes the earlier failure; changing the inference context alone is insufficient. The successful graph-break paths are not fullgraph support.

The changing-neighborhood recompilation problem is fixed in source `6df9aeab5`: **ten real 16-row queries now complete with actual Inductor using only two full graphs**, at the default recompilation limit. The first call captures one graph, the second captures another, and all remaining calls reuse them. Previously, the ninth distinct neighborhood exceeded the default limit of eight. All 160 predicted classes match eager; three query batches still miss the strict probability tolerance (maximum `6.25e-5`), so the recorded status remains `parity_fail`. This run uses isolated caches with cache reuse disabled ([full results](dynamic-neighborhoods.json)).

An independent grouping check varies row counts, shape compatibility, column names, dtypes, and member counts. All **32 actual-Inductor checks pass** across both versions and compiler modes. Rows 3 → 5 → 7 reuse one graph; schema or grouping changes correctly recompile while preserving exact member order ([script](grouping.py), [results](grouping-results.json)). No global recompile limit was raised.

Unbounded graph traversal, arbitrary schemas, input mutation, nonempty variable-length operations outside the tested paths, CUDA/cuDF, and full dataset metrics still need separate validation. Opaque external operations may add overhead; no speedup is inferred from capturing fewer graphs.

## Reproduce

`probe.py` actually invokes the selected compiled entry. `predict` fits eagerly first; `fit` compiles and calls fitting, then checks eager prediction; `forward` compiles and invokes the outer model. Use trusted external data/checkpoint files, which are not committed.

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/compile_public_paths/probe.py \
  --model relational --entry predict --fullgraph --capture-dynamic-outputs \
  --arm-index 0 --relational-query-indices 1 0 2 1 \
  --data /path/to/driver-dnf_bundle.pt \
  --checkpoint /path/to/Kumo-Relational/classifier.pt \
  --output /tmp/relational-predict.json
```

For tabular classification, use `--model tabular --entry predict --fullgraph --query-rows 4 7 3 4`, its `.npz` data, and `small/classifier.pt`; omit relational options. `--estimators 4` checks ensembling. Regression uses `--task regression` and `regressor.pt`. The PCH setting is specific to the local macOS 2.14 compiler setup. `--inner-only` is a diagnostic that compiles internal modules while keeping public preprocessing eager.

Latest focused existing suites at `6df9aeab5` pass on both runtimes: **118 passed, 69 skipped** (categorical/conversion/constant/calendar, relational, base-model, and relational-model checks). Component branches provide additional focused two-version checks. [HISTORY.md](HISTORY.md) and [baseline.json](baseline.json) retain earlier failures and the original 24-case current-main baseline.

To repeat the longer relational stream, change the command above to `--arm-index 1 --relational-query-indices 3 4 5 6 7 8 9 10 11 3`. Run `grouping.py` directly with each runtime for the independent metadata/member checks.
