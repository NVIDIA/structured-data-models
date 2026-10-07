# Variable-length preprocessing support

This branch builds on public-path integration `8950a466c`. Its focused production changes are `110749ef6` (fake-safe string representation) and `d6497a344` (logical offset metadata and scalar writes).

The new `_storage_offset` attribute stores the same logical offset already passed to the wrapper constructor. A compiled caller can read it without tracing `Tensor.storage_offset()`, whose non-tensor result Dynamo rejects in this path. Shape, strides, storage relationships, flatten/unflatten context, and public `data_offset` behavior are unchanged.

Scalar writes in variable-length concatenation and compaction use `fill_`/`zero_` on one-element slices. The previous scalar assignment failed under the custom string-lookup operator's Python dispatch mode with `Hit Python dispatch key but no arguments had PyInterpreter`. This does not change output values or allocate another payload buffer.

## Validation

All results below use CPU Inductor. No GPU performance or memory measurements are claimed.

| Check | 2.7.1 | 2.14.0 |
|---|---|---|
| Read offset metadata on ordinary string containers and explicitly constructed nonzero-offset containers; changing row/byte counts | Both fullgraph and dynamic settings pass | Both fullgraph and dynamic settings pass |
| Custom operator executing string concatenation with fixed-size numeric output | Both fullgraph settings pass | Both fullgraph settings pass |
| Existing StringTensor/VarLenTensor tests | 48 passed, 15 skipped | 48 passed, 15 skipped |

The metadata matrix contains 16 separate process runs and compares output tensors to eager execution. `offset_results.json` records the results. Explicit nonzero offsets retain the original underlying payload and offset tensors.

Run from this checkout with SDM dependencies installed:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 python experiments/varlen_preprocessing/offset_metadata.py construct true true
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 python experiments/varlen_preprocessing/dispatch_concat.py
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest test/tensor/test_var_len.py test/tensor/test_string.py -q
```

`offset_metadata.py` accepts `full`, `construct`, or `view`, followed by the fullgraph and dynamic flags (`true`/`false`). Native slice-view inputs with dynamic shapes still expose compiler guard-source failures; the complete table/model view investigation remains separate. Successful manual nonzero-offset reconstruction is not proof that every native view works.

## Rejected symbolic slicing experiment

Native nonempty string concatenation still needs more work. `symbolic_slice_probe.py` demonstrates that a plain tensor slice can retain aliasing while using symbolic bounds constrained by `torch._check`. However, putting those separate symbolic byte lengths inside `StringTensor.__torch_dispatch__` fails: one opaque `torch.cat` graph node introduces multiple unbacked symbols, but only their sum survives in the returned packed payload.

```text
PendingUnbackedSymbolNotFound: Pending unbacked symbols {u0, u1, u2, u3}
not in returned outputs StringTensor(...)
```

`symbolic_narrow_rejected.patch` preserves that diagnostic experiment; it is not applied to production code. It was captured against the integration base and is not a ready-to-merge patch. General ragged compaction, string cloning with arbitrary views, and variable-length payload results remain open. Copying entire backing buffers to avoid dynamic bounds was deliberately not implemented: it can retain unrelated bytes for small views.

The categorical string-lookup investigation uses a fixed-size custom operator around its existing external Arrow join. These metadata and dispatch changes support that boundary; they do not claim that Arrow or all string operations become native compiled kernels.
