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

## Full-context execution

`ICLModel.explain_full_context()` is the first implemented path. It accepts
the same context, query, recipe, and generator inputs as `forward()`, plus an
explanation method and explicit output target:

```python
model.eval()
explanation = model.explain_full_context(
    method,
    x_context,
    y_context,
    x_query,
    related_context_tables,
    related_query_tables,
    target=OutputIndex(row=0, column="approved"),
)
```

The model fits preprocessing once, exposes schema-aligned processed numerical
input sites, the canonical unmodified prediction, the execution mode, and a
repeatable evaluation callable to the method. The callable reuses the same
undecorated execution used by `forward()`, while the public `forward()` remains
inference-only. Gradient-requiring methods execute under
`torch.inference_mode(False)` and `torch.enable_grad()` without using a fitted
cache.

The model validates that the returned prediction matches the canonical
endpoint and that every attribution site, row, identifier, and numerical
column aligns with an input exposed to the method.

The initial path requires one estimator, an evaluation-mode model, an
unbatched target, and processed input-space attribution. Fitted explanation is
added separately because inference-mode caches require different safety and
support rules.
