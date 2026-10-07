# Nested symbolic dimension dispatch

A dynamic relational prediction graph emitted `aten.sym_size.int` on a StringTensor. Execution failed because VarLenTensor/StringTensor did not register that operation. NullableTensor had the same missing handler.

The fix forwards the dimension query to the leaf that defines logical shape:

```python
# VarLenTensor and StringTensor
return inp._layout.size(dim)

# NullableTensor
return inp._data.size(dim)
```

CategoricalTensor already forwards unhandled operators to its code tensor, so it needs no additional implementation. No buffers are allocated or copied and no flatten metadata changes.

Validation: direct operator calls and registered handlers, plus CPU Inductor with `dynamic=True`, both fullgraph modes, PyTorch 2.7.1 and 2.14.0. All four wrapper types pass changing row counts 4/7/3/0, including nullable strings and negative dimension queries: eight configurations per version. Compiler disk caches were disabled to isolate this check from the separate tensor-subclass cache investigation.

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 PYTHONPATH=. python experiments/nested_symbolic_size/probe.py
```

The dynamic-neighborhood integration branch owns full-model validation. This focused branch is based on its integration prerequisite `c75ba3af6`; the standalone production delta is commit `eba932257` (10 added lines across two files).
