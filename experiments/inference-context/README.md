# Native inference contexts

Source commit: `931334d3e` (one file, based on `4a98a9563`).

`sdm._inference.inference_mode()` selects a native PyTorch context. Return that
context directly instead of nesting it inside a generator context manager.
This removes an extra context layer Dynamo must track. The choice between
inference mode, no-grad, enabled gradients, and preserving ambient state is
unchanged, including the existing no-grad fallback during compilation.

This is a candidate for a PyTorch 2.7 CUDA failure reporting
`GenericContextWrappingVariable` under recipe transformation. CPU compilation
also passes before the change: the CPU results validate behavior but do not
reproduce or prove a fix for that CUDA failure. Unsupported `aten.record_stream`
in full-graph CUDA compilation is a separate blocker.

The helper is private and all repository callers use `with`. The old generator
wrapper also exposed ContextDecorator behavior; that behavior is not retained
for `mode="none"` because `nullcontext` is not a decorator. No repository caller
uses the helper as a decorator. Invalid-mode assertions now occur when creating
the context rather than on entering it.

## CPU validation

Run with each environment's Python (tested PyTorch 2.7.1 and 2.14.0):

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 python experiments/inference-context/compile_probe.py
PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/inference-context/mode_probe.py
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest -q test/models/test_base.py test/processing/test_fitted_guard.py test/processing/test_recipe.py 'test/models/kumo/tabular/test_model.py::test_forward[small]' 'test/models/kumo/tabular/test_model.py::test_fit_predict[small]' test/models/kumo/relational/test_model.py::test_forward
```

Each runtime passes:

- 8 actual Inductor cases: four modes, both graph policies. The partial-graph
  case deliberately enters an eager function inside the context. Outputs and
  gradient requirements match eager execution.
- 24 ambient-mode / selected-mode / normal-or-exception cases. Check state at
  construction, inside the context, and after leaving it.
- 55 existing tests, with 3 CUDA cases skipped. Includes Tabular forward and
  fit/predict, Relational forward and fit/predict, base gradient behavior,
  recipe validation, and fitted-state guards.

These are correctness checks, not performance benchmarks or proof of complete
model compilation. GPU validation belongs to the separate integration run.
