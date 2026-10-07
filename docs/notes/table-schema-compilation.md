# Column schema guards in PyTorch 2.7

This branch is based on preprocessing integration commit `8950a466c`; the initial container change is two lines, with affected processor call sites updated separately. It adds an immutable `_column_items` tuple alongside the table's existing schema dictionary. The public `table.columns` property still returns the same detached dictionary with `Stype` enum keys. No column ordering or public mapping behavior changes.

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


## Real relational prediction follow-up

Tracing the public KumoRelational `predict()` method on the cached RelBench driver-DNF task identified the same 2.7 guard failure in `AddCalendarFields._transform` (`sdm/processing/datetime/calendar.py`), specifically the datetime column-name lookup. Both raw and cyclic naming paths now reconstruct their dictionary from `_column_items` inside tracing. This leaves calendar arithmetic unchanged.

The calendar tensor computation passes actual Inductor with both graph policies on both versions. Its public `transform()` passes both policies on 2.14 and graph-break-allowed mode on 2.7. Fullgraph 2.7 still fails earlier on the separate `handles_stypes` frozenset handling. This scoped fix is not a claim that every remaining relational preprocessing stage compiles.

The next public relational trace reached `DropConstantColumns._transform_ensemble` and failed guarding `ensemble_table._groups[0].columns`. Its `_select_columns` helper now reads the immutable schema tuple for numerical names and column ordering. Public ensemble transform passes actual Inductor with changing row counts and noncontiguous numerical inputs on 2.7 with graph breaks, and on 2.14 with both graph policies. Existing constant-column tests: 7 passed, 4 CUDA skips. On 2.7, fullgraph remains blocked by `copy.copy` in ensemble replacement and by processor applicability metadata; these are separate from column selection.

A later trace in both public relational prediction and public tabular fitting reached `TableTensor.replace_blocks`, then failed guarding `self.columns` after a constructor boundary. Block replacement now reconstructs that dictionary from `_column_items` as well. Both-version actual compile regression tests pass; the complete table suite reports 62 passed / 8 skipped on 2.14, and 60 passed / 8 skipped on 2.7 with its two known default-device failures excluded.

The following public prediction stage, `ToNumerical._transform`, had the same two enum-keyed lookups when naming converted columns. Those now use the tuple-backed schema; four existing eager tests pass. This branch alone still encounters earlier generator/constructor compilation issues already addressed by the integration dependencies.

## Independent PyTorch reproduction

`experiments/table_compilation/schema_guard_repro.py` imports only PyTorch and Python's `enum` module. A minimal tensor wrapper exposes an enum-keyed schema property; compiling a multiplication that reads that property raises `ConstDictKeySource can only work on DictGuardManager` on 2.7.1 with either graph policy. Both policies pass on 2.14. A regular Python object exposing the same property passes on both versions: the problematic case is tensor-subclass metadata handling.

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/table_compilation/schema_guard_repro.py
```
