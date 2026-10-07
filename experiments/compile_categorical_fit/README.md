# Categorical fitting compilation experiments

This branch is stacked on `compile/categorical-recipe-support` at `c04de44fc`. The changes below have different readiness levels. Do not treat the entire branch as a production-ready fix.

## Minimal alignment fix: `ff08a40e8`

The existing optimization reuses an input vocabulary when fitting retains all its entries:

```python
if selected_indices.numel() == input_categories.numel():
    ...
```

The retained count depends on training values. A full graph cannot choose that Python branch while tracing. The patch preserves eager behavior and skips only this reuse optimization during compilation, using the existing `index_select` path instead. It still retains only observed categories, preserves their configured ordering, and uses the same aligned codes. It does not pad the vocabulary or assume a particular retained length.

Actual CPU 2.14 Inductor public `AlignCategories.fit_transform` passed `fullgraph=True` for numeric dictionaries with all three sort orders (`code`, `frequency`, `value`), `min_frequency=2`, changing row counts, and retained vocabulary sizes including zero. Fitted-state query transformation also matched eager.

With `ff08a40e8` alone, `fullgraph=False` and default settings still reached a partial-container construction failure. Explicitly enabling `torch._dynamo.config.capture_dynamic_output_shape_ops=True` in the diagnostic avoided it. Subsequent container-construction fixes on this branch also address that failure; no library code changes this global setting.

## String vocabulary ordering and selection

`39a0e1a5d` exposes the existing Arrow/cuDF string-sort permutation as a fixed-output custom operator. The compiler knows the permutation's shape and dtype; the existing external sorter runs unchanged at execution time. Arrow sorting itself is **not** compiled or optimized. The helper returns the original backend ordering, including duplicate values, rather than replacing it with hashes or an approximate ordering.

After a graph break, tracing could re-enter the ordinary `StringTensor.sort()` call and fail with `TypeError: cannot unpack non-iterable int object`. `9f673e44a` uses the existing permutation helper directly in that fallback and keeps only the external sorter outside tracing. This avoids tracing Arrow conversion after resumption. The fullgraph path continues to use the custom operator.

| Validation | Result |
|---|---|
| CPU 2.7.1 and 2.14, string-order operator, actual Inductor, both `fullgraph` settings, `dynamic=True` | Exact permutation parity for contiguous/strided dictionaries, Unicode, embedded NUL, duplicates, empty inputs and changed vocabulary sizes |
| `torch.library.opcheck` on both versions | Schema, fake tensor, autograd registration and dynamic AOT checks pass |
| CPU 2.14, real RelBench driver table, public `AlignCategories(sort_by="value").fit_transform`, `fullgraph=False`, `dynamic=True` | Exact codes and dictionaries for 5 string columns and 32 → 16 → 8 rows |
| Same real-data test with `fullgraph=True` after bounded string selection | Passes all three sort orders with exact parity |

Native variable-length layout/selection changes are included from the container investigation (`766c99636`, `d10cc007c`, `ed1cedee5`, `25b548f27`). They preserve symbolic row counts, empty selections and wrapper construction across graph breaks. The final public numeric/string matrix passes all 12 combinations of dictionary type, `fullgraph` setting and sort order on CPU 2.14 with default settings. String dictionaries in this matrix use int64 offsets from `StringTensor.from_list`; contexts change row counts and retained vocabulary sizes, including zero. Fitted-state transformation of missing and unseen query codes also matches eager.

**Int32-offset packing fix:** Arrow-derived dictionaries use int32 byte offsets. Packing selected strings checks `total_bytes <= 2147483647` to choose int32 or int64 output offsets. That total depends on selected values, so fullgraph tracing originally could not choose the dtype. Commits `ef9a46a6b` and `3b8b0204d` provide a conservative bound from dictionary metadata: fitting selects unique logical entries, so selected bytes cannot exceed `dictionary_size * backing_bytes`. This remains valid for stride-zero dictionaries, where several logical entries refer to the same bytes. When the bound fits int32, the compiler can retain int32 offsets without assuming anything about selected values. A runtime check enforces the bound.

The helper applies only to compiled fitting of int32-offset string dictionaries; eager selection and int64-offset selection retain their existing paths. Actual CPU 2.14 Inductor fitting now passes on all five RelBench driver dictionaries for every sort order and both graph settings, with 32 → 16 → 8 rows and exact codes/dictionary parity. The related native string/variable-length/alignment suite passed 73 tests; 27 CUDA cases were skipped.

**Remaining limit:** If the conservative metadata bound exceeds the int32 limit, the original data-dependent dtype choice remains. Fullgraph support is not claimed for that case, even when the selected strings might happen to fit. This is a representation constraint, not a fixed dictionary-size specialization.

An additional all-observed → partially-observed vocabulary check passes with both graph settings. It exposed a dependency regression: reconstructing a fresh layout leaf lost Dynamo's symbolic sources during partial-graph int64 fitting. `c31f252ed` retains the original flattened `_layout` leaf identity; do not apply the earlier alias-reconstruction change without this restoration.

Direct numeric `_fit_column` checks on 2.7.1 pass all sort modes with graph breaks. Fullgraph numeric selection needs `capture_dynamic_output_shape_ops=True` on that version. This is a helper result, not evidence that every 2.7 public processor/schema path compiles.

## Experimental shuffle change: `4d092d8e2`

Existing fitting samples permutations and deduplicates identical ones using `permutation.tolist()` as Python dictionary keys. The prototype skips this value-dependent deduplication during compilation, keeping one permutation group per ensemble member. It preserves sampled values and RNG draws, but can increase stored groups and downstream computation.

CPU 2.14 Inductor, four ensemble members, random and cyclic-shift strategies:

| Case | Result |
|---|---|
| `fullgraph=True`, explicit `torch.Generator`, 0/1 categories | Passes; no random draw is needed |
| `fullgraph=True`, explicit `torch.Generator`, 3 categories | Fails at `randperm(generator=...)`: Dynamo cannot represent the generator argument |
| `fullgraph=True`, global RNG, 0/1/3 categories | Passes; same-seed outputs match eager |
| `fullgraph=False`, explicit `torch.Generator`, 0/1/3 categories | Passes with the same generator seed; graph breaks remain |

The 0/1-category fullgraph results are **not** evidence that normal categorical fitting works with an explicit generator. With the global RNG diagnostic, groups grew from 1 to 4 for 0/1 categories, and from 3 to 4 for the tested 3-category draws. No model speed or memory improvement is established, and no replacement of SDM's explicit-generator behavior is proposed.

Recommendation: the alignment guard is a small independent compatibility change. The shuffle prototype documents what a static grouping scheme costs; explicit generator support and grouping overhead need resolution before recommending it for integration.

## Reproduce

Run from the branch root with the desired PyTorch environment:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python experiments/compile_categorical_fit/fit_probe.py
```

The probe uses actual Inductor, `dynamic=True`, both graph settings and all three sort orders. It reports failures rather than silently switching backends. To use the existing real-data artifact, add `--bundle /path/to/driver-dnf_bundle.pt`; that checks the five driver string dictionaries with changing row counts. The bundle must be a trusted local artifact because loading it requires its serialized SDM container objects.
