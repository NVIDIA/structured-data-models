# Compiling public model prediction

This experimental branch supports compiling tensor preprocessing and prediction together. It is based on main `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`; these are branch results, not current-main guarantees. The successful cases **fit eagerly, then compile and call `model.predict`**. No GPU speed or memory claim is made here.

## Current results

Real cached data, pretrained weights, CPU Inductor, and the same FP32 weights/inputs for eager and compiled comparisons:

| Model / data | PyTorch | Graph breaks allowed | Fullgraph | Prediction parity |
|---|---|---|---|---|
| KumoTabular, numerical classification | 2.14 | Pass | Pass | Max absolute difference `8.94e-7` |
| KumoTabular, four estimators / numerical classification | 2.14 | Pass | Pass | Max `3.28e-7`; initial snapshot |
| KumoTabular, numerical regression | 2.14 | Pass | Pass | Max `1.84e-4`; initial snapshot |
| KumoRelational, one-hop RelBench / four related tables | 2.14 | Pass | Pass* | All query sizes pass; max `4.83e-6` |
| KumoRelational, two-hop RelBench / six related tables | 2.14 | Runs | Runs* | One of eight probabilities narrowly misses tolerance on the four-row sample; reproduced by compiling only the inner model |
| KumoTabular, numerical classification | 2.7.1 | Pass; partial capture | Blocked | All query sizes pass; max `1.13e-6` |
| KumoRelational, one-hop RelBench | 2.7.1 | Pass; partial capture | Blocked | All query sizes pass; max `4.83e-6` |
| Public `fit()` / outer `model(...)` | Both | Separate investigation | Separate investigation | Not covered by the successful prediction cases |

*Relational fullgraph needs scoped `capture_dynamic_output_shape_ops=True` and an explicit finite `num_hops`. Arrow/cuDF joins remain opaque external operations inside the captured graph; their algorithms are not compiled.

The relational fullgraph checks run **4 → 1 → 8 → 4 query rows with their matching related neighborhoods** through one compiled callable. Captured graph counts are **1 → 2 → 3 → 3**; first graphs contain 5,714 calls for one hop and 8,076 for two hops. New sizes recompile despite `dynamic=True`; repeating the query reuses its graph. Tabular checks use **4 → 7 → 3 → 4**, with fresh tables and sliced views.

Relational fullgraph source: `d8cbc7383`. The latest tabular classifier repeat passes after atomic wrapper construction (`aa3876c33`). Initial tabular regression/four-estimator source: `284f612cf`. Exact comparisons use `atol=1e-5, rtol=1e-4`; they are not bitwise or dataset-wide quality guarantees. [Prediction results](relational-prediction-progress.json) preserve source commits, failures, graph counts, and exact probability values; [initial results](prediction-final.json) preserve the earlier tabular matrix.

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
- String category lookup and identifier joins use custom operations that call the existing Arrow/cuDF implementation. Join outputs have a data-dependent length; this is why the scoped compiler setting is needed.
- Task graph validation remains active at runtime. With finite `num_hops`, propagation executes a bounded number of tensor updates; updates after an empty frontier are no-ops.
- Internal-model prerequisites are included: Fourier-buffer compilation, relational graph preparation, and relative-time processing. Chunk-size preservation is separate; compilation success alone does not establish memory parity.

The join prototype preserves tested duplicate, empty, missing, nullable, string, multikey, dtype, and validation-error behavior. Details and standalone commands are in [DYNAMIC_JOIN.md](DYNAMIC_JOIN.md).

## Precision and limitations

The two-hop probability miss also occurs with entirely eager preprocessing and **only the inner model compiled**: eager `0.0907173976`, inner-compiled `0.0906983167`, public-compiled `0.0906982943`. Predicted classes are unchanged. The public-versus-inner difference for that value is `2.24e-8`; the tolerance was not relaxed. Direct compiled preprocessing gives bitwise-equal values in all 12 nonempty numerical/datetime blocks across the task and six related tables ([results](relational-preprocessing-parity.json)).

A CPU diagnostic using scalar code generation (`cpp.simdlen=1`) passes the original tolerance across all four two-hop query sizes. FP32 is unchanged. This points to vector/reduction arithmetic in the inner compiler, but scalar code generation is not recommended as a performance fix without further measurement. No precision flags were changed in the library.

PyTorch 2.7 public prediction now passes with graph breaks after traceable schema lookups, atomic wrapper construction, and atomic table dispatch. The dispatch boundary leaves tensor handlers eligible for compilation (`recursive=False`). It prevents Dynamo from incorrectly resuming an operator through another operator's wrapper. One-hop relational queries capture **71 → 118 → 131 → 131** graphs; tabular captures **24 → 45 → 65 → 66**. Default recompilation limits still cause eager fallbacks, and the repeated tabular query adds a graph. This is partial compilation with substantial fragmentation, not a speed recommendation.

An auditing backend delegates to real Inductor and saves every graph. The relational run includes **627 linear and 108 attention calls across recompilations**, confirming that neural computation is captured as well as preprocessing ([capture summary](relational-27-capture.json)). These are compile-time counts, not runtime operation counts. Reproduce with `audit_inductor.py --backend audited_inductor` and `SDM_GRAPH_DIRECTORY` set to a new output directory.

Fullgraph remains blocked for both models on 2.7: the reported generic-context error masks unsupported recipe-applicability tracing. A direct-context diagnostic exposes the earlier failure; changing the inference context alone is insufficient. The successful graph-break paths are not fullgraph support.

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

Latest focused existing suites pass on both runtimes: **118 passed, 69 skipped** (categorical/conversion/constant/calendar, relational, base-model, and relational-model checks). Component branches provide additional focused two-version checks. [HISTORY.md](HISTORY.md) and [baseline.json](baseline.json) retain earlier failures and the original 24-case current-main baseline.
