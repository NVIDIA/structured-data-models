# Compiling fitting and recipe preprocessing

This is an integration investigation branch, based on `8950a466c`, with
container, categorical, numerical and schema patches from the companion
branches. It is not a standalone proposed change to main. No GPU measurements
or inference speed claims are made here.

The fitting-specific source change is `0f323da0d`: register fitted buffers
explicitly, retaining their persistent/nonpersistent state. When a processor
is constructed inside a compiled call, ordinary assignment to its existing
buffers can make PyTorch inspect `register_buffer` at runtime. PyTorch 2.14
then fails with `SourcelessBuilder cannot wrap SourceFileLoader`. Explicit
registration avoids this introspection without changing the fitted values.
The Standardize, fitted-guard and recipe tests pass on both runtimes:
21 passed, 6 unavailable-CUDA skips each. Persistent and nonpersistent fitted
buffers were also checked through `state_dict()`.

## Actual public fit results

Every run uses CPU Inductor, `dynamic=True`, matching eager data and dtype,
and an explicit `torch.Generator().manual_seed(123)`. Public fits use the
pretrained model weights, and are checked by predicting after fitting.
The prediction check is `atol=1e-5, rtol=1e-4`; this is the probe's stated
tolerance, not a claim of bitwise equality or changed model precision.

| Call | Runtime / graph policy | Result |
|---|---|---|
| Tabular classification `torch.compile(model.fit)`; 32 context rows, four estimators, 31 query rows | 2.14 / breaks allowed | Pass: max prediction difference `5.36e-7`; 51 captured graphs and 10,086 captured calls; final generator state exactly matches eager. |
| Tabular regression `torch.compile(model.fit)`; diabetes, 32 context rows, one estimator | 2.14 / breaks allowed | Pass: max prediction difference `2.75e-4`; 36 graphs / 3,430 calls. |
| Tabular regression `torch.compile(model.fit, fullgraph=True)` | 2.14 / no breaks | Buffer registration fix advances tracing to explicit-generator `FlipSign.bernoulli_`, which Dynamo cannot proxy. |
| Tabular classification `torch.compile(model.fit)` | 2.7.1 / breaks allowed | Default recipe construction can resume with an uninitialized `TaskDispatch` module. Passing a prebuilt recipe avoids that failure. |
| Same, with `recipe=model.default_recipe()` prepared before the compiled call | 2.7.1 / breaks allowed | Pass after schema/view/mask patches, four estimators / 31 query rows: max prediction difference `4.77e-7`; 88 graphs / 8,614 calls; generator state exact. |
| Relational classification public fit; actual RelBench driver-DNF bundle with related tables | 2.14 / breaks allowed | Pass with string-sort support and a narrow eager boundary for external Arrow/cuDF joins: two estimators: max prediction difference `1.41e-5`; 139 graphs / 15,055 calls; generator and all fitted recipe buffers exact. |
| Same relational public fit, with a prebuilt recipe | 2.7.1 / breaks allowed | Atomic dispatch fixes a wrong-handler resume failure; dynamic tracing then fails a compiler symbol-to-source guard assertion. Still unsupported. |

Tabular classification's fitted mean, scale, power-transform parameters and
random state match eager exactly. Only clipping bounds differ, by `8.88e-16` on 2.14 and up to `4.44e-15` on 2.7.
The four-estimator result was reconfirmed with the subsequent schema,
string-sort, ragged-selection and view-cleanup patches at source `43c9a142b`.

All public probes use a process-local compiler cache limit of 64 to avoid
the default limit of eight causing shared processor methods to fall back
after repeated specialization. No production configuration is changed.
Captured graph counts establish that fitting arithmetic is compiled; they
do not establish that every operation is compiled, nor measure speedup.

## Fitted numerical parity

The default relational feature recipe has two numerical branches. A real
160-row breast-cancer fixture, with a constant column added, exposed a
significant FP32 fitting difference under ordinary Inductor: the Power
branch differed by `0.00225`, with fitted lambda differences up to `0.00127`.
This compares FP32 eager against FP32 compiled; no dtype switch is involved.

The numerical patch keeps the original native reductions during fitting
inside opaque compiler operations. The missing-value case also requires
preserving the imputation mean: a small difference there changes the input
to the power-parameter optimizer. After the complete numerical fixes, the
same default recipe with sparse NaNs has:

| Check | Result, CPU 2.14 Inductor |
|---|---|
| Power branch output | Exact eager equality |
| Identity branch output | Max difference `9.54e-7`; ordinary `torch.testing.assert_close` passes |
| Imputation, Standardize and Power fitted buffers | Exact eager equality |
| Remaining buffer differences | Only clipping bounds, max `9.54e-7` |
| Captured regions | 23 graphs / 2,245 calls |

This is an independent default **feature recipe** check. The public fitting
result above additionally validates categorical and related-table preprocessing
before the neural context computation. `native_sum_backend.py` retains the diagnostic backend
used to isolate the cause; the passing final result uses the production
source patch and ordinary Inductor, without that backend.

## What cannot be a single graph yet

Fitting changes Python structure based on data: constant-column selection,
learned vocabulary lengths and permutation deduplication. It also consumes
the caller's explicit random generator. Merely removing the generator or
fixing the learned schema would change behavior and is not an acceptable fix.
With graph breaks allowed these decisions can run eagerly between compiled
tensor regions. A no-break API needs these decisions and random draws
prepared before the compiled tensor application, or separately supported
operators with correct shape and generator semantics. The existing fitted
`transform` path already separates much of this work from query arithmetic.

Preparing a recipe is only constructing processing modules; it does not fit
statistics or alter the input-dependent behavior:

```python
recipe = model.default_recipe()
compiled_fit = torch.compile(model.fit, fullgraph=False, dynamic=True)
compiled_fit(context, target, recipe=recipe, generator=generator)
```

This avoids the 2.7 constructor problem and, with the schema and view fixes
in this branch, passes the public fitting probe. `model.fit` still deep-copies the supplied recipe,
so the caller's recipe remains reusable with independent fitted state.

## Reproduce

From this checkout, set `PYTHON` to the desired 2.7.1 or 2.14 environment and
set the data/checkpoint paths to local assets. The NPZ contains `x`, `y`,
`train_ids` and `validation_ids`; the public probe uses context rows only
for fitting. The relational bundle uses the existing driver-DNF fixture.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 "$PYTHON" \
  experiments/fitted-recipe-boundaries/public_probe.py \
  --model tabular --entry fit --estimators 4 --query-rows 31 \
  --data "$TABULAR_DATA" --checkpoint "$TABULAR_CHECKPOINT" \
  --output /tmp/tabular-fit.json

PYTHONPATH=. OMP_NUM_THREADS=1 "$PYTHON" \
  experiments/fitted-recipe-boundaries/public_probe.py \
  --model relational --entry fit \
  --data "$RELATIONAL_DATA" --checkpoint "$RELATIONAL_CHECKPOINT" \
  --output /tmp/relational-fit.json

PYTHONPATH=. OMP_NUM_THREADS=1 "$PYTHON" \
  experiments/fitted-recipe-boundaries/probe.py \
  --model relational --members 2 --missing --cache-limit 64 \
  --data /tmp/breast-cancer.pt --output /tmp/relational-recipe
```

The recipe fixture can be saved as
`torch.save((torch.tensor(load_breast_cancer().data, dtype=torch.float32),),
"/tmp/breast-cancer.pt")` using scikit-learn's bundled dataset.
The probe adds a constant column and sparse NaNs itself.
Add `--fullgraph` to check the no-break policy; add `--prepared-recipe` to
the public probe to prepare only the recipe construction outside compilation.
Other atomic-boundary flags are diagnostic experiments, not required by the
successful source-patched 2.14 public fit.

Raw outcomes and tracebacks are in `results/`. They intentionally include
remaining failures; none of those failures are counted as support.
