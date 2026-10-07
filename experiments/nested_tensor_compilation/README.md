# Nested tensor compilation investigation

Branch: `compile/nested-tensor-preprocessing`, based directly on main `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. This is a component investigation, not a claim that whole-model preprocessing compiles.

## Changes

- Add `CategoricalTensor.__tensor_flatten__` / `__tensor_unflatten__`. Each dictionary is exposed as a named tensor leaf. Public `code` and its backing `_code` are both listed because PyTorch 2.7 otherwise recurses while guarding the property. No codes or dictionaries are copied; the existing constructor preserves their shape, strides, and storage offsets.
- Add `category(index)` to access a dictionary inside captured code. Dynamo treats the existing tuple-valued `categories` property on newly constructed wrappers as constant metadata and cannot wrap the fake tensors inside that tuple. The original property remains available.
- Make categorical, nullable, variable-length, and string representations avoid reading values when formatting fake tensors. PyTorch 2.7 unconditionally formats AOT metadata; the previous formatting called `tolist()` or computed a null count from fake values and failed before code generation. Eager representations are unchanged.
- Handle concatenation of empty variable-length payloads using shapes and zero offsets. This includes tables without text columns and columns of empty or missing strings. Validity masks and 32/64-bit offset promotion are preserved. General nonempty string compaction is unchanged.

## Commands

Use an environment with SDM dependencies and the desired PyTorch version. Run from this checkout:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 python experiments/nested_tensor_compilation/validate.py
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 python experiments/nested_tensor_compilation/validate.py --no-dynamic
PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/nested_tensor_compilation/category_access.py
PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/nested_tensor_compilation/reproduce_dynamic_stride.py
OMP_NUM_THREADS=1 python experiments/nested_tensor_compilation/raw_wrapper_stride.py
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest test/tensor/test_categorical.py test/tensor/test_var_len.py test/tensor/test_nullable.py test/tensor/test_string.py -q
```

All compilation results below use actual CPU Inductor, not the `eager` capture backend. Both `fullgraph=False` and `fullgraph=True` were checked. The precompiled-header environment setting is a local macOS toolchain workaround.

## Results

| Case | PyTorch 2.7.1 | PyTorch 2.14.0 |
|---|---|---|
| Construct categorical wrapper; numerical code arithmetic; return wrapper | Pass | Pass |
| Numeric dictionaries: strided views, nonzero offsets, alias preservation, row slicing and cloning | Pass | Pass |
| String dictionaries: constructor output and underlying bytes/offset alias preservation | Pass | Pass |
| Read `category(index)` after construction, bound and unbound call | Pass | Pass |
| Stack empty/missing strings; changing rows and 32/64-bit offset promotion; `dynamic=False` | Pass | Not required to resolve the dynamic result |
| Same string stack with `dynamic=True` | Fails on changed rows: upstream stride reconstruction | Pass |
| Existing nested-tensor tests | 79 passed, 31 skipped | 79 passed, 31 skipped |

Categorical validation changes row counts across 5, 9, 0, and 2; codes have nonzero storage offsets and noncontiguous strides. String stacking includes absent columns, present empty strings, missing values, empty rows, and changed nonempty row counts. GPU/cudf tests were skipped. No GPU speed or memory claim is made.

## Remaining limitations

### PyTorch 2.7 dynamic output strides

`raw_wrapper_stride.py` reproduces this without importing SDM. A wrapper output of shape `(4, 2, 2)` has correct strides `(4, 2, 1)`. Reusing the compiled function for seven rows produces shape `(7, 2, 2)` but incorrect strides `(7, 2, 1)` instead of `(4, 2, 1)`. SDM's bounds validation catches the resulting invalid StringTensor reconstruction:

```text
ValueError: 'offset' in 'StringTensor' is out of bounds
(got 29 entries, but expected at least 47 entries)
```

`dynamic=False` passes the tested row changes by compiling separate shapes. Disabling duck shaping did not fix the dynamic result. No bounds checks are removed and no nonzero offsets or noncontiguous layouts are silently replaced to hide this issue.

### Tuple properties and ragged operations

- `CategoricalTensor(...).categories[0]` still fails with `Unexpected type in sourceless builder ... FakeTensor`. Use `category(0)` in compiled code; callers need to adopt this where they inspect newly constructed dictionaries.
- Nonempty string cloning/compaction still slices payloads using tensor-valued byte offsets. General ragged outputs need additional work; copying entire backing storage merely to avoid those bounds can retain unnecessary data and is not implemented here.
- `NullableTensor.data` shadows the base Tensor property. Dynamo interprets it as Tensor's `_get_data_attr`, which invokes unsupported `aten.detach.default`, rather than reading SDM's physical values. Direct `_data` arithmetic works in the focused constructor probe; this public-property issue is not fixed here.
- `StringTensor.data_offset` and logical validity views need further work when their values or storage offsets are read directly inside compiled code.
- PyTorch 2.14 emitted AOT cache serialization warnings for some symbolic wrapper graphs even though execution and parity checks passed. These checks establish behavior, not efficient compilation caching.

No graph-break suppression, global Dynamo configuration changes, or precision changes are used by the implementation.
