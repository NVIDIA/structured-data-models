# Atomic TableTensor dispatch during partial compilation

This branch adds one decorator to `TableTensor.__torch_dispatch__`:

```python
@classmethod
@torch.compiler.disable(recursive=False)
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    ...
```

It is based on `compile/tabletensor-preprocessing` at `af49b2706`. Public-model validation additionally uses the preprocessing integration branch and its schema, categorical, recipe, and constructor fixes.

## Observed failure

On PyTorch 2.7.1, a fitted KumoRelational model with real RelBench driver-DNF tables reaches this failure:

```python
predict = torch.compile(model.predict, fullgraph=False, dynamic=True)
predict(query_table, related_query_tables)
# AttributeError: 'list' object has no attribute 'is_inference'
```

The trace goes through `EnsembleTable.concatenate_columns` → `torch.cat(groups, dim=-1)` → `TableTensor.__torch_dispatch__` → the **view** wrapper. Eager dispatch registers concatenation to the separate non-view wrapper. Another compiler-cache setting instead produced `unsqueeze(): argument dim must be int, not torch.dtype`. These are dispatcher/resumption failures, not invalid model inputs. The exact wrong-handler failure has not yet been reduced to a standalone PyTorch-only example.

## Change

The dispatcher selects a Python handler and invokes it atomically. Dynamo cannot resume halfway through that selection. `recursive=False` still permits compilation of functions the dispatcher calls. Tensor operations already represented in an FX graph continue to lower through the normal wrapper-subclass protocol; this does not make the entire preprocessing method eager.

No operator semantics, inference-mode rules, dtypes, or prediction algorithms change. No broad exception suppression is used.

## Validation and limits

Actual CPU Inductor results:

| Case | Result |
|---|---|
| Container compilation, both graph policies, 2.7.1 and 2.14 | Pass |
| Existing table suite, 2.14 | 60 passed, 8 CUDA skips |
| Existing table suite, 2.7.1 | 58 passed, 8 CUDA skips; two known main failures excluded |
| Integrated public KumoRelational prediction, 2.7.1, one real four-row query | Pass; max same-dtype absolute difference 1.55e-6 |

The public prediction captures 81 graphs / 2,496 calls with the default compiler cache limit. It is **partial compilation**: graph breaks and eager fallback remain. These counts do not demonstrate that every operation compiles, or measure performance. Tests of changing queries and more related tables are tracked by the integration investigation. This branch does not claim that public `fit()` or fullgraph prediction works on 2.7.

Source-level regression command:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/tensor/test_table_compile.py test/tensor/test_table.py
```

On 2.7, add `-k 'not default_device'` for the two pre-existing pandas default-device failures. No GPU benchmark or speed/memory benefit is established by this correctness change.
