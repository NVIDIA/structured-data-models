# Column schema guards in PyTorch 2.7

This branch is based on preprocessing integration commit `8950a466c`; the schema-specific production change is two lines. It adds an immutable `_column_items` tuple alongside the table's existing schema dictionary. The public `table.columns` property still returns the same detached dictionary with `Stype` enum keys. No column ordering or public mapping behavior changes.

## Failure

PyTorch 2.7 can fail while constructing guards for an existing dictionary keyed by a `StrEnum`:

```python
def preprocess(table):
    return table.numerical * len(table.columns[Stype.numerical])

torch.compile(preprocess, fullgraph=True)(table)
```

Observed errors include:

```text
AssertionError: ConstDictKeySource can only work on DictGuardManager
AssertionError in DictGetItemSource.__post_init__
```

Allowing graph breaks does not reliably avoid guard construction failures. The numerical computation is supported; the error concerns Python schema metadata. PyTorch 2.14 does not exhibit the same guard failure in this reproduction.

## Scoped fix

Processors can iterate `table._column_items`, or construct their lookup dictionary inside the traced function:

```python
def preprocess(table):
    columns = dict(table._column_items)
    return table.numerical * len(columns[Stype.numerical])
```

The compiler guards a tuple of enum/name pairs instead of an existing enum-keyed dictionary. Schema-dependent branches remain guarded: tests change column names while keeping the input tensor shape unchanged and verify that the compiled output changes correctly. Constructing a new table inside compilation and reading this tuple also works.

This is a compiler-facing representation for SDM's own call sites. It does not claim to repair arbitrary user code that accesses `table.columns` under PyTorch 2.7. Recipes and processors must use the tuple at affected sites; their changes belong to the separate integration branch. The tuple is created by the normal table constructor, including replacement, loading, and unflattening paths. Table schemas remain immutable after construction.

## Validation

Actual CPU Inductor on PyTorch 2.7.1 and 2.14, both `fullgraph=False` and `fullgraph=True`: tuple metadata lookup, iteration, schema-change guard invalidation, and newly constructed table metadata pass. Inputs include changing row counts and unchanged shapes with changed column names. No dtype conversion or numerical algorithm changes are involved.

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/tensor/test_table_compile.py -k schema_guards
```

The existing table and Standardize tests also run as regression checks. PyTorch 2.7's two pre-existing pandas default-device failures are excluded from that suite; they reproduce on main. GPU behavior and performance are not measured by this metadata-focused change.
