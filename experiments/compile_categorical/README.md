# Compiling categorical preprocessing

This prototype is stacked on public-preprocessing integration commit `8950a466c`. It requires the shared table/nested-container compiler support already on that branch. It is not a standalone change against main.

## Changes

- Read category dictionaries through `CategoricalTensor.category(index)`. Dynamo can trace individual tensor leaves but rejects the tuple-valued `.categories` property on newly constructed wrappers.
- Use fixed-size numeric alignment and shuffling lookups instead of data-dependent Boolean indexing. Preserve missing negative codes, empty dictionaries, category order, and integer dtypes. The numeric alignment change is required for fullgraph on 2.7.1; 2.14 already supports that original dynamic operation.
- Pass fitted dictionaries as ordinary tuples instead of `BufferList` modules to tensor construction.
- Call the existing `TableTensor.items(table)` iterator directly during numerical conversion.
- Represent string category lookup as one custom operator in compiled graphs. Its implementation invokes the existing Arrow/cuDF join; its output always has one code per input dictionary entry. The join itself is opaque to the compiler and is **not optimized by Inductor**.

The string operator receives raw byte/offset tensors and logical sizes, strides, and offsets. It reconstructs the existing dictionary views and uses the existing materialization helper for strided dictionaries. No quadratic string comparison or probabilistic hashing is introduced. Dictionary nulls are prohibited by `CategoricalTensor` construction; missing table values are negative codes. Reconstructing dictionaries with `valid=None` therefore preserves the existing invariant.

The compiler needs the custom operator's fake implementation to determine output shape, dtype, and device without running Arrow. Eager execution calls the original join helper directly. Routing eager calls through the custom operator measured 4–16% extra CPU lookup latency (about 20–60 microseconds across the tested small vocabularies); the compiler-only selection avoids that overhead. These are component measurements, not model speedups.

## Validation

CPU, actual Inductor, same dtype as eager. No GPU validation or model-performance claim. Compact results, including the rejected unconditional eager-operator measurements, are in `results.jsonl`.

| Check | PyTorch 2.7.1 | PyTorch 2.14 |
|---|---|---|
| Categorical processor suite, including two compilation checks | 53 passed; 25 CUDA skipped | 53 passed; 25 CUDA skipped |
| String lookup helper, contiguous/sliced/strided dictionaries, both graph-break settings | Passed | Passed |
| Raw string custom operator, `dynamic=True`, both graph-break settings, contiguous/strided dictionaries | Passed | Passed |
| Public string alignment after eager fit, missing/unseen values, varying row counts | Still blocked by surrounding container/applicability tracing | Passed both graph-break settings |
| Public string alignment with strided dictionary views and `dynamic=False` | Not established | Passed both graph-break settings |
| Public string alignment with native strided dictionary views and `dynamic=True` | Fails input symbolic-shape binding | Same failure before processor execution |
| `torch.library.opcheck`: schema, fake tensors, autograd registration, AOT dynamic dispatch | Not run | All passed, contiguous and strided dictionaries |

Public 2.14 alignment tests changed row counts between 4/6/3 and checked codes, dictionaries, and schema against eager. Each shape still produced a separate 26-node graph despite `dynamic=True`; this is not proof of general graph reuse. Public numeric shuffling after eager fit passed both settings with a 30-node graph per row count. Its fitting path passed with graph breaks allowed but still fails fullgraph because permutation deduplication reads sampled values with `.tolist()`.

Run:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/processing/categorical -q

OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python experiments/compile_categorical/lookup_parity.py
```

The second command also covers Unicode, embedded NUL, empty strings/dictionaries, duplicate dictionary values, int32/int64 codes, and keeping matches within their own columns. It compares against the unchanged eager join, preserving its existing duplicate-match behavior.

## Remaining limitations

- `AlignCategories.fit` materializes retained vocabularies with data-dependent lengths and branches on those lengths. This branch does not make fitting fullgraph-compatible.
- `AddCategoryCounts` validates tensor values with Python conditions and `equal()`. Removing those checks would change behavior; fullgraph still needs a validation strategy.
- `ShuffleCategories.fit` deduplicates permutations through Python values. Its tensor remapping is fixed here; its state construction is separate.
- Dynamic native strided `StringTensor` inputs expose a symbolic metadata issue before the processor runs. Static-shape capture and the raw operator work; the limitation is not silently hidden by falling back to eager.
- Full public-model results belong to the public-preprocessing integration investigation. Passing these processor components does not establish complete KumoRelational fitting/prediction support.
