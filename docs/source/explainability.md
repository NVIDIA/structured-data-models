# Explainability

:::{warning}
The explainability interface is experimental and may change as concrete
methods are added.
:::

SDM separates explanation mathematics from model execution. An
`ExplanationMethod` owns an algorithm and its settings, while an `ICLModel`
owns preprocessing, cache behavior, gradient mode, and repeated model
evaluation.

The model exposes two explicit execution paths:

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
unbatched target, and processed input-space attribution.

## Fitted execution

`ICLModel.explain_fitted()` explains query predictions using state created by
`fit()`:

```python
model.eval()
model.fit(x_context, y_context, related_context_tables)
explanation = model.explain_fitted(
    method,
    x_query,
    related_query_tables,
    target=OutputIndex(row=0, column="approved"),
)
```

This path shares query preprocessing, cache replay, and output processing with
`predict()`. It exposes query sites only; context inputs are represented by
the fitted cache and cannot be replaced. Methods must include
`ExplanationMode.fitted` in their requirements to use this path.

The initial fitted path supports repeatable processed-numerical perturbations
with fixed cached context. It does not yet expose raw, categorical, structural,
or context interventions. Gradient methods fail before execution because
`fit()` deliberately creates frozen inference-mode caches, which sever the
autograd graph to context and cannot safely participate in backward passes.
Gradient and context attribution therefore use `explain_full_context()`.

## Gradient sensitivity

`GradientSensitivity` differentiates one explicit final prediction with
respect to the processed numerical inputs the model actually consumes:

```python
from sdm.explain import GradientSensitivity

explanation = model.explain_full_context(
    GradientSensitivity(
        magnitude=True,
        normalization="global_max_abs",
    ),
    x_context,
    y_context,
    x_query,
    related_context,
    related_query,
    target=OutputIndex(row=0, column="approved"),
)
```

The example configuration matches Kumo-ML's magnitude and global
normalization choices: absolute gradients divided by one maximum absolute
score across every emitted table. Unlike Kumo-ML's product behavior, SDM uses
one explicit final output target and reports processed context and query sites.
More precisely, each score is the derivative of the selected final prediction
with respect to one processed numerical cell. The default is signed and
unnormalized. `global_max_abs` uses one denominator across all emitted context
and query tables, preserves sign, and leaves all-zero scores as finite zeros.
Scores remain in processed feature space with no feature reduction.

Each processed input site produces one `FeatureAttribution`. Its `values` is a
schema-aligned `TableTensor`: numerical blocks contain scores and identifier
blocks preserve row alignment. The `site` identifies the context or query
split and, for related inputs, the table name. Disconnected numerical inputs
receive zero scores. The full-context model path currently requires one
estimator.

```python
attribution = explanation.attributions[0]
attribution.site          # InputSite(split="query", table="users")
attribution.values        # TableTensor
attribution.input_space   # "processed"
```

Gradient sensitivity is local to the explained input and depends on feature
scaling. It is not an additive contribution or evidence of causality. The
method covers processed numerical sites exposed by prepared execution,
including recipe-generated calendar columns. Model-internal latent features
outside that boundary are not included.

## Captum Integrated Gradients

Install the optional dependency with
`pip install "structured-data-models[captum]"`. Importing `sdm.explain` does
not import Captum.

`CaptumIntegratedGradients` uses the same full-context model execution path as
native gradient sensitivity:

```python
from sdm.explain import CaptumIntegratedGradients, InputSite

method = CaptumIntegratedGradients(
    baselines={
        InputSite(split="query"): processed_query_baseline,
        InputSite(split="query", table="users"): processed_users_baseline,
    },
    n_steps=50,
)
explanation = model.explain_full_context(
    method,
    x_context,
    y_context,
    x_query,
    related_context,
    related_query,
    target=OutputIndex(row=0, column="approved"),
)
```

Baselines are method configuration, not model configuration. They are keyed
by `InputSite` so one method can select task, context, query, or related-table
inputs without changing a model class. A baseline may be a constant scalar, a
numerical tensor, or a schema-aligned `TableTensor`. Values are in the
processed feature space exposed to the method. If `baselines` is omitted,
every floating-point input site is compared with zero. If a mapping is
supplied, only its sites vary; unlisted sites stay fixed at their endpoint
values.

The method adapts SDM's repeatable scalar evaluation callable to Captum's
`IntegratedGradients` callable. Captum repeatedly interpolates numerical
values from each baseline to the endpoint; SDM performs each model evaluation
with identical prepared inputs and random state. The public `forward()`
remains inference-only.

Each varied site yields a signed, unnormalized `FeatureAttribution` whose
`values` is an aligned `TableTensor`. `IntegratedGradientsDiagnostics` records
the baseline prediction, varied sites, integration steps, and Captum's signed
completeness residual:

```text
sum(attributions) - (prediction[target] - reference_prediction[target])
```

The residual is an approximation diagnostic, not an error bound. Integrated
Gradients is baseline-dependent and does not establish causality. The initial
adapter supports full-context execution only; fitted inference caches are
inference tensors and therefore cannot provide the gradient path Captum
requires.
