# Compiling conversion of empty ragged storage

A numerical `TableTensor` still has an empty text block. During compiled BF16 prediction, converting the output table back to FP32 copies that block. The ragged conversion unnecessarily reads two tensor values to slice its zero-length payload, causing a fullgraph compilation error:

```python
def convert(x):
    return TableTensor.from_tensor(x.to(torch.bfloat16)).to(torch.float32).numerical

compiled = torch.compile(convert, fullgraph=True, dynamic=True)
compiled(torch.ones(4, 3))
# Could not extract specialized integer from data-dependent expression u0
# sdm/tensor/var_len.py, _to_dtype_layout: data[offset[0]:offset[-1]]
```

The fix uses the existing backing tensor directly when its length is zero, then performs the existing conversion. There is nothing to slice: valid offsets all point to zero. Dtype, copy behavior, offset normalization, validity, and nonempty-payload handling are unchanged. This also covers nonzero numbers of empty strings or empty ragged values, not just tables without text columns.

## Validation

Base: frozen integration `c75ba3af6c231c7bdcd181a50d35cdc7fb53c090`. Production change: `4071789e4`, independent of the cache-stream event fix.

| Check | PyTorch 2.7.1 CPU | PyTorch 2.14 CPU |
|---|---|---|
| Actual Inductor table conversion, both graph settings, rows 4/9/0/3 | Pass | Pass |
| Actual Inductor empty-payload dtype conversion, same matrix | Pass | Pass |
| Existing VarLen/String suites | 48 passed, 15 skipped | 48 passed, 15 skipped |

Before the change, the table reproduction fails in 2.14 fullgraph with the same error as actual L4 public BF16 prediction. The partial-graph reproduction succeeds by breaking the graph. After the change all eight version/mode/input-kind combinations pass; table output parity is exact. The probe checks ragged data and offsets as well as logical shape.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
python experiments/empty_varlen_conversion/probe.py

OMP_NUM_THREADS=1 PYTHONPATH=. \
python -m pytest test/tensor/test_var_len.py test/tensor/test_string.py -q
```

The source fix has been sent for an actual L4 rerun of the original BF16 public-prediction case. That result is pending. This fixes a tracing error before predictions exist; it makes no claim about BF16 model prediction parity yet. Conversion of nonempty ragged payloads can still require data-dependent slice bounds.
