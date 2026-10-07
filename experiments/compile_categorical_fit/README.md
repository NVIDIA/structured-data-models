# Categorical fitting compilation experiments

This branch is stacked on `compile/categorical-recipe-support` at `c04de44fc`. The two commits below have different readiness levels. Do not treat the entire branch as a production-ready fix.

## Minimal alignment fix: `ff08a40e8`

The existing optimization reuses an input vocabulary when fitting retains all its entries:

```python
if selected_indices.numel() == input_categories.numel():
    ...
```

The retained count depends on training values. A full graph cannot choose that Python branch while tracing. The patch preserves eager behavior and skips only this reuse optimization during compilation, using the existing `index_select` path instead. It still retains only observed categories, preserves their configured ordering, and uses the same aligned codes. It does not pad the vocabulary or assume a particular retained length.

Actual CPU 2.14 Inductor public `AlignCategories.fit_transform` passed `fullgraph=True` for numeric dictionaries with all three sort orders (`code`, `frequency`, `value`), `min_frequency=2`, changing row counts, and retained vocabulary sizes including zero. Fitted-state query transformation also matched eager.

With `fullgraph=False` and default settings, a graph break at the dynamic selection still reaches the existing partial-container construction failure. Explicitly enabling `torch._dynamo.config.capture_dynamic_output_shape_ops=True` in the diagnostic allows the same public numeric tests to pass. The patch does not silently change that global configuration.

String fitting has additional issues: data-dependent string selection reaches `VarLenTensor` shape/materialization code, while value sorting invokes the external Arrow/cuDF backend. Those are separate from the reused-vocabulary check.

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
