# Atomic wrapper construction

This branch keeps `CategoricalTensor.__new__` and `TableTensor.__new__` atomic when Dynamo falls back to tracing their Python implementation. It adds one `torch.compiler.disable` decorator to each constructor. Tensor computations in preprocessing remain eligible for compilation.

## Failure and fix

During `torch.compile(model.fit, fullgraph=False, dynamic=True)`, a graph break at `_make_wrapper_subclass` can resume tracing before all wrapper attributes exist. Dynamo then flattens that incomplete wrapper:

```
AttributeError: 'CategoricalTensor' object has no attribute '_categories'
AttributeError: 'TableTensor' object has no attribute '_columns'
```

Avoiding a pause inside allocation lets the constructor finish initializing its metadata. Normal traceable tensor-subclass construction is still represented as a graph operation; the primitive fullgraph tests below verify this is preserved. The change does not disable `fit` or introduce eager fallback around all preprocessing.

## Validation

CPU Inductor, PyTorch 2.7.1 and 2.14.0, each `fullgraph=False` and `True`, `dynamic=True`:

- Construct categorical/table wrappers inside the compiled function, then compute on the resulting tensor. Changing row counts 3/5 passes.
- Independent mixed-table check constructs both wrappers, replaces numerical data and reads numerical/categorical output; rows 5/9/0 pass on both versions and modes.
- Existing table compilation and categorical suites: 2.7.1: 34 passed, 14 skipped. On 2.14.0: 33 passed, 14 skipped; the final mixed parametrization hits the default recompile limit when run together. The same failure reproduces on the parent branch. That case passes when run alone.
- Public KumoTabular classification fit on real data passes with both constructors patched: 39 compiled graphs, 3,394 graph calls; subsequent prediction maximum absolute difference 1.07e-6. This is a CPU correctness check, not a speed result. The public-path investigation owns the broader fitted-model matrix.

Reproduce the focused test from this branch:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python experiments/atomic_tensor_construction/probe.py
```

This is a focused investigation branch atop `compile/varlen-preprocessing`; it includes its prerequisite container/preprocessing support. The two production decorators are commit `ed1cedee5`, which can be applied independently to the integration branch.

The same partial-initialization failure also affects `VarLenTensor.__new__`, inherited by StringTensor (`_valid` is missing). Its constructor now uses the same atomic boundary. On the integrated string-selection branch, all six bounded-selection cases still pass 2.14 Inductor and the six-case nested-tensor dynamic matrix passes both 2.7.1 and 2.14.0 in both graph modes. Native nonempty cloning/concatenation is tested separately there; this decorator does not remove its strict-fullgraph scalar-slicing limitation.
