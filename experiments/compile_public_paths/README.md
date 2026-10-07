# Public Kumo compilation investigation

This branch is an integration prototype, based on main `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. It includes table/recipe tracing work, PR #1054's Fourier-buffer fix, relational graph preparation, and PR #1068's relative-time fix. No PR, GPU validation, speed claim, or memory claim is attached to these results.

## What works

Actual CPU Inductor, PyTorch 2.14, **public** `torch.compile(model.predict, fullgraph=..., dynamic=True)` after eager `fit()`:

| Pretrained KumoTabular workload | Estimators | `fullgraph=True` | `fullgraph=False` | Largest absolute prediction difference |
|---|---:|---|---|---:|
| Classification, numerical input features | 1 | Pass | Pass | 8.94e-7 |
| Classification, numerical input features | 4 | Pass | Pass | 3.28e-7 |
| Regression, numerical input features | 1 | Pass | Pass | 1.84e-4 |

Each case runs query row counts **4 → 7 → 3 → 4** through the same compiled callable. Freshly constructed tables and sliced `TableTensor` views both pass. All comparisons use the same weights, data, and dtype in eager and compiled execution (`atol=1e-5`, `rtol=1e-4`); regression targets are explicitly FP32 to match the pretrained model. This is tolerance-based parity, not bitwise equality or a dataset-wide quality evaluation.

The one-estimator classifier captured 2,234 tensor calls in its first graph; four estimators captured 8,610. This covers tensor preprocessing, internal model computation, and output processing. Graph counts were **1 → 2 → 3 → 3**: new row counts still caused recompilation despite `dynamic=True`; the repeated four-row input reused a graph. These are genuine compiled calls, not an eager tracing backend or merely constructing an unused wrapper.

Final model/processor source for these checks: `284f612cf`. `prediction-final.json` contains exact outcomes and counters. Existing base-model, relational-model, and Standardize tests pass on both runtimes: **49 passed, 14 CUDA-only skips** each.

## Changes needed outside the inner model

1. Table and nested-container flatten/unflatten expose tensor leaves and preserve metadata. Empty columnar leaves must also survive view replay: the first sliced-input check failed because an empty ID block allocated a real tensor among fake tensors; the corrected view handler preserves the existing leaf.
2. Recipe applicability and packing read traceable column metadata rather than unsupported container-returned objects. Recipe setup validation traverses registered children without the module mutation failure.
3. Query validation compares cached schema columns directly, preserving the same validation without constructing a `TableSchema` inside tracing.
4. Fit caches class column names once. Previously both Kumo wrappers converted class tensors into Python strings on every prediction; prediction now reuses those names. Class order and numerical calculations are unchanged. The shared fit cache adds this small metadata item for other models too; only the two Kumo wrappers consume it here.
5. Internal-model prerequisites remain necessary. This branch includes them to test the public path rather than stop at a known inner-model error. Chunk-size preservation (#1055) is separate and is not included; passing compilation does not establish memory-efficiency parity.

## Remaining blockers

| Path | Current evidence |
|---|---|
| KumoTabular public prediction, PyTorch 2.7.1 | Fails with and without graph breaks. Fullgraph reports a break under the generic context manager. A fresh Inductor cache confirms a dictionary-source guard failure in recipe schema metadata with breaks allowed (`ConstDictKeySource can only work on DictGuardManager`). |
| KumoRelational public prediction, PyTorch 2.14 | Fullgraph fails reading string category metadata in `AlignCategories` for related-table features. Allowing breaks reaches a data-dependent scalar extraction in nonempty variable-length string concatenation. Both failures precede the inner model. |
| KumoRelational public prediction, PyTorch 2.7.1 | Generic-context and dictionary-source guard failures remain. |
| Compile public `fit()` or outer model | No passing end-to-end case. Baseline and intermediate results identify recipe construction, category alignment, and data-dependent preprocessing blockers. The successful prediction cases fit eagerly. |
| Relational Arrow joins | Moving joins before the inner model does not move them outside a compiled public `predict()`. Even after categorical preprocessing is fixed, fullgraph needs a supported representation for joins or a narrower compiled boundary; graph-break mode needs an explicit boundary around external table work. |
| General container support | Nonempty variable-length strings, mixed-table input views, input mutation, categorical fitting, and schema-changing processors still require work; see the component branch documentation. |

The relational checks use a real RelBench driver-dnf bundle with related tables and pretrained weights. They include graph preparation and relative-time fixes, but still fail in preceding categorical processing. The integration does not include later experimental categorical-processor changes from the separate investigation branch.

## Reproduce

`probe.py` invokes the selected compiled callable:

- `fit`: compile and call `model.fit`, then eager prediction to check fitted state.
- `predict`: eager fit, then compile and call `model.predict`.
- `forward`: compile and invoke the outer model with context and query data.

Use trusted external data and checkpoint files; they are not committed. Tabular data is an `.npz` containing `x`, `y`, `train_ids`, and `validation_ids`. This probe uses 32 context rows. The relational bundle contains pickled SDM objects, with the first arm and second query selected. Both models use seeded generators for the eager and compiled comparison.

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/compile_public_paths/probe.py \
  --model tabular --task classification --entry predict --fullgraph \
  --estimators 4 --query-input view --query-rows 4 7 3 4 \
  --data /path/to/data_train_validation.npz \
  --checkpoint /path/to/Kumo-Tabular/small/classifier.pt \
  --output /tmp/tabular-predict.json
```

Omit `--fullgraph` to allow breaks. Select `--query-input fresh` to construct fresh numeric tables instead of slicing a table. For regression use `--task regression` and `regressor.pt`. For relational use `--model relational --entry predict`, its driver-dnf bundle and classifier checkpoint, and omit the tabular query-row options. Run each configuration in a fresh process. The PCH setting above is a workaround for the local macOS PyTorch 2.14 compiler setup.

## Baseline evidence

`baseline.json` records all 24 current-main combinations (two models × public fit/predict/outer forward × two graph-break settings × two runtimes). All eager reference calls succeeded and all compiled entries failed before the fixes. `containers-initial.json` and `recipe-integration.json` preserve intermediate first errors and source commits. Local ignored `results/` retains full tracebacks. No failure or pure eager fallback is counted as successful compilation.

## Follow-up: external join boundary

The continuing integration adds individual category access during alignment and an explicit compiler boundary around `join_index`. Arrow/cuDF joins remain eager; surrounding tensor arithmetic can compile. Before this boundary, even `fullgraph=False` attempted `TableTensor.to_arrow()` with fake tensors and raised `.numpy() is not supported for tensor subclasses`. With the boundary, actual Inductor passes a duplicate-key join followed by tensor arithmetic on 2.7.1 and 2.14, including inside SDM's inference context. Fullgraph correctly rejects this explicit graph break; it is not fullgraph support for Arrow. Existing join checks pass (6 CPU cases, 5 CUDA skips on 2.14).

This alone does not finish public relational prediction: string dictionary packing still fails before the join, and a diagnostic that keeps just string lookup eager exposes partially initialized `CategoricalTensor` reconstruction after a graph break. These are being handled separately. The earlier result table remains evidence for the explicitly recorded source commit, not a claim that every ongoing integration change has passed that entire matrix again.
