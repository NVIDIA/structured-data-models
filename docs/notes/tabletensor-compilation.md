# TableTensor compilation investigation

This branch adds compiler-visible tensor leaves to `TableTensor`, `ColumnarTensor`, and `CategoricalTensor`. It is a prerequisite for compiling tensor preprocessing; it does not make Arrow conversion or every processor compilable. The branch includes the nested-container changes from `compile/nested-tensor-preprocessing`.

## What failed and what changes

Before this change, the compiler treats `TableTensor` as an unsupported Python object or encounters an unsupported nested constructor before reaching its arithmetic:

```python
compiled = torch.compile(lambda table: table.numerical.sin(), fullgraph=True)
compiled(TableTensor.from_tensor(torch.randn(5, 3)))
```

`__tensor_flatten__` exposes each contained tensor, while `__tensor_unflatten__` rebuilds the same schema and blocks from the compiler's tensor values. This allows tensor operations to be traced without converting tables to another public input type.

- Both private storage attributes and public block properties are exposed. Public properties are required when constructing a table inside a graph; PyTorch 2.7 additionally needs the private aliases to avoid recursive property guards. These are aliases of the same tensors, not copied data.
- Slicing an input table must also preserve the empty identifier block's tensor leaf. PyTorch replays views while rebuilding fake inputs, sometimes outside an active fake mode. Empty-column views therefore reshape the existing zero-element leaf instead of retaining a newly allocated real tensor. This fixes `All fake tensor modes must be the same` for sliced inputs.
- An empty `ColumnarTensor` keeps one zero-element tensor so that device, symbolic shape, and fake-tensor mode remain available even without any columns. Reconstruction reuses the supplied leaf and its device; it does not allocate a new symbolic tensor.
- Schema names remain immutable metadata. Changing the schema can legitimately require recompilation.
- Fake-tensor representations avoid reading values. The nested variable-length concatenation change also avoids value-dependent slicing when the entire payload is empty, including absent text columns.

## Validation scope

CPU actual Inductor, PyTorch 2.7.1 and 2.14, both `fullgraph=False` and `fullgraph=True`, `dynamic=True`:

| Operation | Numeric tables | Mixed numerical/categorical/datetime/text/ID tables |
|---|---|---|
| Construct from a tensor inside compilation | Pass | Not applicable to `from_tensor` |
| Replace numerical block and return a table | Pass | Pass |
| Row slicing and named column selection | Pass | Pass |
| Stack tables / transpose the stacked result | Pass | Fails for nonempty string payloads |
| Return a numerical view, then mutate it outside compilation | Preserves aliasing | Not separately tested |
| Mutate an input block inside compilation | Fails: missing `aten.copy_` implementation | Not separately tested |

The focused compile tests pass all eight cases on both versions. The existing table/columnar suites plus these tests report 77 passed and 13 CUDA-skipped on 2.14. On 2.7 they report 75 passed, 13 skipped, and two pre-existing `from_pandas` default-device failures; both failures reproduce unchanged on main `842c408fe`. Ruff passes. Type checking reports only the existing `TableTensor.tolist` override diagnostic.

The exploratory matrix varies row counts 5, 9, and 5 with noncontiguous numerical blocks. Committed tests additionally cover zero rows. These checks establish supported changing row counts, not a guarantee of zero recompilations. They compare against eager execution of the same dtype; no precision changes are introduced.

A separate flatten/unflatten check moves leaves from CPU to meta and verifies reconstruction uses the new device. CPU roundtrips preserve block strides, nonzero storage offsets, and aliasing. GPU execution and performance are not measured here.

Reproduce the maintained tests from the repository root, using either runtime:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/tensor/test_table_compile.py \
  test/tensor/test_table.py test/tensor/test_columnar.py
```

The precompiled-header setting works around the local macOS PyTorch 2.14 C++ environment; it is not a library configuration change.

## Remaining work

**Nonempty variable-length payloads.** Concatenating string tables still reaches `VarLenTensor.data_offset`, which slices storage using tensor-valued offsets. Fullgraph capture fails on the data-dependent integer; allowing graph breaks can also fail. Empty-payload support does not solve this general case. Supporting it requires a tensor-based gather/concatenation formulation with correct ragged-view and null semantics, or an explicit boundary around that operation.

**Input mutation.** AOT functionalization copies modified input wrappers back after computation, but `TableTensor` has no `aten.copy_` implementation. For example:

```python
def mutate(table):
    view = table[1::2]
    view.numerical.add_(3)
    return view

compiled = torch.compile(mutate, fullgraph=True)
compiled(TableTensor.from_tensor(torch.randn(5, 3)))
# NotImplementedError: 'aten.copy_.default' is not supported for 'TableTensor'
```

Both tested versions fail with and without graph breaks. General copy support must define behavior for categorical dictionaries and variable-length payloads while preserving aliases; this branch does not add a numerical-only copy workaround. Pure transformations returning a replacement table work in the tested cases.

**Python container methods.** In PyTorch 2.7, some custom methods on traceable tensor subclasses are treated as tensor operators. A generator-returning `table.items()` or newly created `frozenset` metadata can still prevent fullgraph tracing in processors. Separate recipe/processor changes address call sites without changing the table's public metadata types.

**Pre-sliced mixed inputs.** Numeric input-table views now compile after preserving the empty identifier leaf. Passing a pre-sliced table with nonempty text into a compiled replacement-and-slice function still exposes symbolic-shape failures: PyTorch 2.7 reports an undefined symbolic variable and 2.14 reports missing symbolic metadata. Fresh mixed tables pass the same operation. This needs additional nested-container/view work; mixed slicing in the table above means slicing inside compilation from a fresh table, not arbitrary pre-sliced mixed inputs.

**Compilation cache.** PyTorch 2.14 warns that these subclasses lack `_stable_hash_for_caching`; dynamic subclass metadata may prevent persistent AOT cache serialization. The tested computations still execute, but cross-process compilation cache behavior is not established.

## PyTorch 2.7 inference-context bytecode failure

Public relational prediction and tabular fitting reached a PyTorch 2.7 internal exception-table `KeyError` when empty-column view metadata was updated inside `torch.inference_mode`. The failure reproduces without SDM:

```python
import torch

def view(*args):
    with torch.inference_mode(args[0].is_inference()):
        out = args[0].unsqueeze(0)
        if not args[0]._columns:
            for tensor in out if isinstance(out, tuple) else (out,):
                if isinstance(tensor, torch.Tensor):
                    tensor._empty = args[0]._empty.reshape(tensor.shape)
        return out

x = torch.empty(5, 0)
x._columns = ()
x._empty = torch.empty(5, 0)
torch.compile(view)(x)
```

2.7.1 raises `InternalTorchDynamoError: KeyError: Instruction(... LOAD_FAST ... args ...)`; 2.14 succeeds. Moving the metadata loop and return outside the context passes on both versions. The actual tensor view remains inside inference mode. This narrow reordering also removes the failure from the real 2.7 relational prediction trace; subsequent schema guards remain separate blockers. Container/compile regression checks: 25 passed, 5 CUDA skips on each runtime.
