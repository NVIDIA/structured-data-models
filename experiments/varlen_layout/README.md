# Preserve variable-length tensor layouts

`VarLenTensor` and its `StringTensor` subclass have logical shape/strides separate from the byte payload and offsets. Their previous flatten context retained scalar layout metadata. With dynamic input sizes, PyTorch 2.7.1 reconstructed a stacked output with an incorrect stride: `(7, 2, 1)` instead of `(4, 2, 1)` after the row count changed from 4 to 7.

The fix adds a tensor leaf representing the logical layout:

```python
out._layout = offset.as_strided(size, stride, storage_offset)
```

This view shares the existing offset storage. Flattening includes `_layout`; reconstruction reads its shape, strides and offset. It allocates tensor metadata, not another payload or offset buffer. Reconstruction regenerates the layout view from the supplied offset storage, so the alias remains true even if a compiler independently materializes the input leaves. Existing logical storage offsets and views are preserved.

## Validation

CPU Inductor on PyTorch 2.7.1 and 2.14.0, both `fullgraph` settings:

- Dynamic empty/missing-string stacking now passes changing row counts, including the 2.7.1 regression above.
- Numerical/string categorical dictionaries, dynamic rows, strided codes, storage offsets and alias checks pass the six-case `validate.py` matrix.
- Existing variable-length/string tests pass on each version: 48 passed, 15 skipped (GPU/optional backend cases).

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python experiments/varlen_layout/validate.py
```

This does not resolve every symbolic view problem. A native pre-sliced StringTensor input can still fail Dynamo guard generation on PyTorch 2.14. General nonempty string concatenation/compaction also remains a separate dynamic-output investigation. No prediction speed or GPU memory claim is made by these CPU correctness checks.

The focused production change is commit `766c99636`, atop the previously documented `compile/varlen-preprocessing` support. Public integration tests consume that commit independently.
