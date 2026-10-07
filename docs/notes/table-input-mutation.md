# Compiled mutation of table inputs

This branch adds `aten.copy_` support to `TableTensor` and `ColumnarTensor` for matching encoded storage. It builds on the preprocessing integration and schema-metadata branches. It is separate from the atomic-constructor change.

## Failure

```python
def mutate(table):
    view = table[1::2]
    view.numerical.add_(3)
    return view

compiled = torch.compile(mutate, fullgraph=True)
compiled(TableTensor.from_tensor(torch.randn(5, 3)))
```

Before this change, both PyTorch 2.7.1 and 2.14 fail with:

```text
NotImplementedError: 'aten.copy_.default' is not supported for 'TableTensor'
```

The compiler converts in-place computation into functional operations and then writes changed values back to the original input. That writeback requires copying into the table wrapper. Allowing graph breaks does not resolve this backend failure.

## Fix and limits

A shared helper uses the existing flatten protocol to validate the storage structure, then copies into the existing tensor leaves. It preserves the original wrappers, buffers, and aliases. Repeated private/public aliases are copied once. Schema, container type, storage shape, and metadata mismatches are rejected before any writes. Conflicting repeated destination aliases are also rejected. Distinct destination leaves sharing a storage allocation are rejected before copying. A source leaf sharing storage with a different destination leaf is also conservatively rejected before copying, so swapped columns cannot silently overwrite each other.

This is **matching-storage copy support**, not arbitrary table assignment. Ragged payload buffers and categorical dictionaries must already have compatible shapes and metadata. The helper does not resize buffers, add validity masks, rename columns, or replace dictionaries with differently sized ones. Ordinary same-shape tensor dtype conversion follows PyTorch's existing `copy_` behavior; the compiler fix does not select a new dtype.

In particular, a table view still observes writes to the original table after copying. A compiled function that returns the mutated input preserves that input object's identity in the tested cases. Returning a mutated numerical view preserves its storage alias.

## Validation

Actual CPU Inductor, PyTorch 2.7.1 and 2.14, both graph-break settings:

| Case | Result |
|---|---|
| Mutate a numerical row view, varying 5/9/0 rows with `dynamic=True` | Pass; values and aliasing match eager |
| Mutate numerical, categorical codes, datetime, string bytes, and nullable ID data together | Pass; same input identity and values |
| Numerical and ID blocks share underlying storage | Pass; alias preserved |
| Copy a mixed table and read an existing destination view | Pass |
| Copy from different ragged storage sizes | Raises before changing the destination |
| Swap aliased columns, with separate tensors or views into one allocation | Raises before changing either destination |
| Distinct destination views overlap, including an unchanged source alias | Raises before changing either destination |
| Numerical mutation in FP16, BF16, FP32 and FP64 | Exact same-dtype eager/compiled parity for the tested addition |

Mixed string cases use `dynamic=False` to isolate writeback from the separate ragged symbolic-shape issue. GPU execution and speed/memory effects are not established here. General cross-buffer overlapping copies are rejected, including disjoint views that share a storage allocation; this conservative restriction avoids adding snapshot allocations. No additional tensor clones are introduced by the helper, but compiler functionalization can itself create intermediate tensors.

Reproduce the maintained checks:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/tensor/test_table_copy.py \
  test/tensor/test_table_compile.py test/tensor/test_table.py \
  test/tensor/test_columnar.py
```

The full CPU checks report 85 passed and 13 CUDA skipped on 2.14. On 2.7 they report 83 passed and 13 skipped with the two pre-existing pandas default-device failures excluded (`-k 'not default_device'`). Ruff passes; type checking reports only the existing `TableTensor.tolist` override diagnostic.

After the destination-alias check, the focused copy/compile suite reports 18 passed on each runtime. Shared numerical/ID storage mutation still passes both graph policies. The restriction also rejects disjoint destination views into one allocation; arbitrary overlapping assignment is outside this branch.
