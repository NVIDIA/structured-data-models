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
| Same real-data test with `fullgraph=True` | Still fails in int32-offset string packing, described below |

Native variable-length layout/selection changes are included from the container investigation (`766c99636`, `d10cc007c`, `ed1cedee5`, `25b548f27`). They preserve symbolic row counts, empty selections and wrapper construction across graph breaks. The final public numeric/string matrix passes all 12 combinations of dictionary type, `fullgraph` setting and sort order on CPU 2.14 with default settings. String dictionaries in this matrix use int64 offsets from `StringTensor.from_list`; contexts change row counts and retained vocabulary sizes, including zero. Fitted-state transformation of missing and unseen query codes also matches eager.

**Remaining real-data blocker:** Arrow-derived dictionaries use int32 byte offsets. Packing the selected strings checks `total_bytes <= 2147483647` to decide whether the result still uses int32 or promotes to int64. The total depends on selected string values, so fullgraph tracing cannot choose the output dtype. Int64-offset toy inputs do not exercise this branch. The current code keeps the existing dtype behavior; it does not assume all strings fit within 2 GB.

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
