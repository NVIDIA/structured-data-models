# Processor non-finite-value strategy

Status: design only. This document does not change runtime behavior and must
not be published with an implementation PR.

## Decision

Numerical processor inputs are finite-only by default. `MeanImpute` is the
only current processor allowed to consume NaN; it still rejects positive and
negative infinity. Missing categorical values remain represented by their
existing integer sentinel and are not part of this policy.

Use protected validation hooks instead of a public capability flag. Composite
processors delegate validation to their selected children, and `Sequential`
checks its final numerical output once. `Recipe` normalizes every configured
role, including a bare processor, to `Sequential` so it cannot silently pass
NaN or infinity to a model.

This gives three explicit guarantees:

1. A processor cannot learn fitted state from unsupported non-finite input.
2. The processor that owns a supported NaN operation handles it explicitly.
3. A model-facing recipe either returns finite numerical data or raises.

## Evidence

Current behavior is inconsistent. Probes with one non-finite value observed:

| Operation                                     | Current result                                       |
| --------------------------------------------- | ---------------------------------------------------- |
| `Clip.fit(NaN)` or `StandardScale.fit(NaN)`   | poisons fitted state; later finite input returns NaN |
| `Power.fit(inf)` or `Quantile.fit(inf)`       | later finite input returns NaN                       |
| `SigmaClip.fit(inf)`                          | later finite input returns infinity                  |
| `SoftmaxTemperature.transform(NaN or inf)`    | spreads NaN across the row                           |
| `Power`, `Quantile`, and `SigmaClip` with NaN | preserve NaN silently                                |
| `MeanImpute` with NaN                         | returns finite values as intended                    |

The proposed route order already works with current processors. A mixed table
containing numerical NaN and a missing categorical sentinel was run through:

```python
StypeDispatch(
    numerical=MeanImpute(),
    categorical=ToNumerical(),
    remainder="drop",
)
StandardScale(epsilon=1e-6)
```

It returned a finite `8 x 3` table with columns `("x0", "x1", "kind")`.
The current TabICLv2 default recipe also returned finite features, target, and
predictions for the same data.

## Options

| Option                                                                    | Result                                                                       |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Model-boundary validation only                                            | Cheap, but fitted processor state may already be corrupted                   |
| Independent checks in every processor                                     | Local errors, but duplicated policy and inconsistent messages                |
| Implicit imputation                                                       | Keeps execution running, but changes semantics and can leak query/test data  |
| Default leaf validation plus explicit consumers and a recipe output guard | **Selected:** local state safety, explicit semantics, and finite model input |

## Processor hooks

`Processor` gets a protected input-validation hook. Its default implementation
requires a finite numerical block and produces an actionable `ValueError`.
`MeanImpute` overrides the hook to allow NaN while rejecting infinity and a
non-finite `fill_value`. This override is behavior, so it is clearer than a
boolean such as `allows_nan_input`.

Do not generically scan every leaf output. In a sequence, the next leaf checks
the preceding output and `Sequential` checks the final output. Processors for
which finite input is not sufficient to guarantee finite output must enforce
their own mathematical domain; in particular, `Power` transform and inverse
paths need focused finite-output coverage.

### `fit`

Check supported stypes and numerical input before `_fit`. A validation failure
occurs before fitting starts and therefore preserves previously fitted state.
Immediately before `_fit`, mark the processor unfitted; if `_fit` fails, no
partially updated state remains usable. Mark it fitted only after `_fit`
succeeds.

### `transform`

Check fitted state, supported stypes, and numerical input before `_transform`.
The next processor or the enclosing `Sequential` output guard detects a
non-finite result at runtime. A standalone processor's focused tests must also
prove that valid-domain input produces finite output.

### `forward`

Continue to call `transform` without another validation. Errors should name
`transform`, because `forward` is only the PyTorch call interface.

### `fit_transform`

Validate the input once, perform the same failure-safe fit-state transition as
`fit`, and then call `_transform` directly. Calling public `fit` followed by
public `transform` would scan the same table twice and make error behavior
depend on wrapper composition.

### `inverse_transform`

Check fitted state and numerical input before `_inverse_transform`. Keep this
independent of inverse stype validation in PR #212. Invertible sequences apply
children in reverse order and run the same final finite-output guard.

## Composite behavior

- `Sequential` delegates to child public methods. It does not reject initial
  NaN because its first child may be `MeanImpute`; it requires finite final
  numerical output from `fit`, `fit_transform`, `transform`, and
  `inverse_transform`. An empty sequence therefore rejects non-finite input.
- `StypeDispatch` lets configured route processors validate selected tables.
  It validates a non-empty numerical passthrough block itself because no child
  owns that block. Dropped blocks require no value scan.
- `Choice` in PR #208 delegates to the selected processor. The unresolved
  choice does not define a separate NaN policy.

Before any child fit starts, a fitting composite marks itself unfitted. It is
marked fitted only after every child and the final finite guard succeed; any
child or final-guard failure leaves the parent unfitted. Child state is not
rolled back. Unlike a leaf validation failure, a composite child-validation
failure therefore invalidates previously fitted parent state.

These rules avoid permissive `Identity`, `ToNumerical`, or composite flags.
`Identity` remains finite-only. `ToNumerical` receives a categorical-only
table in the intended dispatch route, where categorical missing sentinels are
finite.

## Processor behavior

| Processor                                                              | Policy                                                                                                  |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `MeanImpute`                                                           | allow NaN, reject infinity, require finite `fill_value`, return finite output including all-NaN columns |
| `Identity`, `ToNumerical`                                              | default finite-only numerical input                                                                     |
| `Clip`, `StandardScale`, `SigmaClip`, `Quantile`, `SoftmaxTemperature` | default finite-only input; retain focused finite-output/extreme-value tests                             |
| `Power`                                                                | default finite-only input plus transform/inverse domain and finite-output tests                         |
| `ConstantFilter` (PR #213), `FeaturePermute` (PR #149)                 | default finite-only; place after imputation                                                             |

## Recipe boundaries

Normalize a bare role such as `Recipe(features=StandardScale())` to a
one-element `Sequential`, just like an iterable role. Preserve an already
constructed `Sequential` without adding another layer. This makes the final
finite guard universal without changing the user-facing processor methods.

TabICLv2 should impute inside the numerical dispatch route:

```python
StypeDispatch(
    numerical=MeanImpute(),
    categorical=ToNumerical(),
)
ConstantFilter()  # once available
StandardScale(epsilon=1e-6)
SigmaClip(threshold=4.0)
```

This removes the temporary need to carry numerical NaN through `Identity`.
Numerical target NaN is rejected unless the target recipe explicitly includes
`MeanImpute`; missing labels require a separate target-label policy.

KumoRFM must keep ID and datetime columns in raw relational tables while
sampling. Only after temporally valid context/query rows have been sampled
should each model-feature projection use the dispatch recipe above. Fit each
table's processors on context/training rows only, reuse them for query rows,
and let the enclosing `Sequential` verify finite hop tensors before model
execution. Never use `remainder="drop"` on source relational tables before
sampling.

## Performance and compilation

A synthetic `MeanImpute -> StandardScale -> SigmaClip` `TableTensor` pipeline
was measured with four scans: one infinity check before imputation, two finite
leaf-input checks, and one final sequence-output check.

| Device/shape             | `fit_transform` baseline -> checked | `transform` baseline -> checked |
| ------------------------ | ----------------------------------: | ------------------------------: |
| CPU `1000 x 100`         |             2.75 -> 3.25 ms (1.18x) |         0.49 -> 1.33 ms (2.70x) |
| NVIDIA L4 `1000 x 100`   |             1.83 -> 2.45 ms (1.34x) |         0.37 -> 0.69 ms (1.85x) |
| CPU `10000 x 1000`       |               589 -> 719 ms (1.22x) |           152 -> 295 ms (1.94x) |
| NVIDIA L4 `10000 x 1000` |             14.0 -> 17.0 ms (1.21x) |         3.96 -> 6.67 ms (1.69x) |

Environment: PyTorch 2.12.1, two CPU threads, NVIDIA L4. The checks are a
material inference cost, but preprocessing correctness takes priority. Avoid
additional generic output scans and benchmark real model workloads during
implementation.

Eager reductions synchronize CUDA and did not compile with
`fullgraph=True` using `torch._check_tensor_all` in this environment.
`torch._assert_async` is not suitable for regular input errors and can
invalidate a CUDA context after failure. PR #109 should therefore compile the
already-validated protected numerical kernel, not promise fullgraph
compilation of public `transform`. Recipe validation remains eager before the
compiled model core.

## Verification and related PRs

Implementation tests must cover all public methods, NaN and both infinities,
failed initial fit and refit state, all-NaN imputation, empty and passthrough
composites, and finite TabICLv2 model inputs/predictions on CPU and CUDA. Add
the equivalent post-sampling KumoRFM feature-projection test when its recipe is
introduced. Composite tests must fail at the first child, a later child, and
the final guard during both initial fit and refit. A bare Recipe role that
produces non-finite output from finite input must fail at its normalized
`Sequential` boundary.

- #109: keep public validation outside the compiled kernel; replace public
  `Power`/`Quantile` NaN-preservation and `Power.inverse_transform(inf)`
  recovery tests with rejection tests. Retain internal overflow repair only
  when finite public input creates a non-finite intermediate.
- #149: keep `FeaturePermute` after imputation.
- #202: route inverse validation through child processors.
- #208: let the selected processor own validation.
- #211: add post-sampling KumoRFM recipe coverage, not raw-table dropping.
- #212: keep finite validation independent of inverse stype checks.
- #213: remove or revise public NaN semantics in `ConstantFilter` before merge.
