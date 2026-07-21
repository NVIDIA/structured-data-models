# Explainability

:::{warning}
The explainability interface is experimental and may change as concrete
methods are added.
:::

SDM separates explanation mathematics from model execution. An
`ExplanationMethod` owns an algorithm and its settings, while an `ICLModel`
owns preprocessing, cache behavior, gradient mode, and repeated model
evaluation.

The model integration is introduced in the next PRs through two explicit
paths:

```python
explanation = model.explain_full_context(
    method,
    x_context=x_context,
    y_context=y_context,
    x_query=x_query,
    target=OutputIndex(row=0, column="approved"),
)

model.fit(x_context, y_context)
explanation = model.explain_fitted(
    method,
    x_query=x_query,
    target=OutputIndex(row=0, column="approved"),
)
```

Methods must explicitly opt into fitted execution. Methods requiring
gradients initially use full-context execution because fitted caches are
created under inference mode.

Both paths return an `Explanation` containing:

- `prediction`: the exact `TableTensor` prediction being explained;
- `target`: the resolved scalar output;
- `method`: the explanation method identity;
- `mode`: whether full-context or fitted execution produced the explanation;
- `attributions`: typed entries whose values are schema-aligned
  `TableTensor` objects; and
- `diagnostics`: optional method-specific typed diagnostics.

Each `FeatureAttribution` records its context/query table location, score
meaning, raw or processed input space, signedness, and normalization. A result
wrapper is necessary because a bare score tensor cannot preserve those
semantics or represent multiple task and related tables.

`OutputIndex` currently selects one row and one numerical output column from
an unbatched prediction. Broader batch and multi-output semantics are deferred
while the interface is prototyped.
