# Public Kumo compilation investigation

Baseline: main `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`, CPU, PyTorch 2.7.1 and 2.14.0. These are actual Inductor invocations with `dynamic=True`, not an eager capture backend. `baseline.json` records all 24 outcomes and first errors. All eager reference calls succeeded; every compiled public entry failed. A compiler error after a graph break is a failure, not successful eager fallback.

## Invocation and coverage

`probe.py` calls the selected compiled callable; it does not merely construct a wrapper:

- `fit`: compile `model.fit`, invoke it, then eager `model.predict` to check fitted state and predictions.
- `predict`: eager fit, then compile and invoke `model.predict`.
- `forward`: compile the outer model and invoke it with context and query data.

All use one estimator, CPU inference mode, identical pretrained weights, and seeded generators. Tabular uses 32 training / 4 validation rows from a cached classification dataset. Relational uses the first arm and second query of the cached RelBench driver-dnf validation bundle. These files and checkpoints are external inputs, not committed data. This small matrix establishes first blockers, not GPU performance, every recipe choice, or changing-shape support.

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/compile_public_paths/probe.py \
  --model relational --entry predict --fullgraph \
  --data /path/to/driver-dnf_bundle.pt \
  --checkpoint /path/to/Kumo-Relational/classifier.pt \
  --output /tmp/relational-predict.json
```

Omit `--fullgraph` to allow breaks. Select `--model tabular` with a `.npz` containing `x`, `y`, `train_ids`, `validation_ids` and the small Kumo-Tabular classifier checkpoint. The relational bundle contains pickled SDM objects; use only a trusted validation bundle. Run each combination in a fresh process. PyTorch 2.14's PCH workaround above is specific to this macOS environment.

## First failures on main

| Public entry | PyTorch 2.7.1, no breaks | PyTorch 2.7.1, breaks allowed | PyTorch 2.14, no breaks | PyTorch 2.14, breaks allowed |
|---|---|---|---|---|
| Tabular fit / outer forward | Constructing recipe: `frozenset(generator)` | Partially constructed `TaskDispatch` lacks `_modules` | Recipe construction: module mutation during `named_modules` | `Stype.categorical` used as compiler dictionary source key |
| Relational fit / outer forward | Same recipe constructor failure | Resumed table dispatch incorrectly receives a list as `unsqueeze` dimension | Same recipe constructor failure | Same categorical dictionary-key assertion |
| Tabular predict | Generic context manager rejects an earlier graph break | `StringTensor` lacks `_valid` | `TableTensor.dim()` unrecognized | `Stype.numerical` dictionary-key assertion |
| Relational predict | Same context-manager failure | Same `StringTensor` failure | Same `TableTensor.dim()` failure | `Stype.datetime` dictionary-key assertion |

The recipe failures precede the internal model. Table flatten/unflatten support alone cannot fix recipe module construction, external joins, or all table-operation dispatch. Disabling the whole recipe would hide these errors but would not compile preprocessing.

## Work to investigate

1. Traceable table/nested-container input and reconstruction; preserve aliases, metadata, missing values, and mixed semantic types.
2. Recipe construction and copying are Python setup. Establish a preparation boundary without excluding processor tensor arithmetic.
3. Traceable recipe execution, semantic dispatch, row/column selection, stacking, and output conversion.
4. Compile processor arithmetic and validate fitted state against eager.
5. Integrate the internal-model fixes and validate the actual public paths again before claiming model support.
