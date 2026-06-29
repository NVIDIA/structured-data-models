# Recipes

A `Recipe` is an inspectable, deterministic processing *contract* around a
model boundary. It bundles the feature transforms a model expects on its input
with the transforms needed to turn the model's output back into the original
space, so that training and inference share one auditable definition instead of
ad-hoc preprocessing scattered across call sites.

## Mental model

There are two objects:

- A **`Pipeline`** is an ordered sequence of processing `steps` applied to the
  **numerical block** of a `TableTensor`. Categorical blocks pass through
  unchanged.
- A **`Recipe`** groups three named pipelines, called **phases**, around the
  point where you call your model:

  | Phase | Runs | Purpose |
  | --- | --- | --- |
  | `preprocess` | before the model | Prepare features for the model. |
  | `target` | after the model | Invert target-side transforms (model space → original space). |
  | `postprocess` | after `target` | Shape-preserving cleanup of model output. |

The phases execute in a fixed order around inference:

```text
preprocess  ->  model  ->  target (inverse)  ->  postprocess
```

Each step is a `Processor` (see {doc}`api/processing`): a fittable,
tensor-in/tensor-out transform with `fit`, `transform`, and (optionally)
`inverse_transform`.

## Building a recipe

Construct each phase from a list of steps. Stateful steps must be fitted before
they transform data — fit them on the **training** split only, then reuse the
fitted recipe on validation/test data so no test statistics leak into
preprocessing.

```python
from sdm.processing import Clip, Recipe, SoftmaxTemperature, StandardScale

recipe = Recipe(
    preprocess=[
        Clip(q_low=0.01, q_high=0.99),
        StandardScale(),
    ],
    target=[],
    postprocess=[SoftmaxTemperature(temperature=2.0)],
)
```

Phases default to an empty (identity) pipeline, so you only pass the ones you
need. A plain list is coerced into a `Pipeline` automatically.

## Running the full lifecycle

Fit the preprocess phase on training data, then apply the recipe around the
model. `target` steps run in **reverse** order via `inverse_transform_target`
to undo target-side transforms, and `postprocess` runs last.

```python
# Fit on train, then transform any split with the same fitted state.
training_table = recipe.fit_transform_preprocess(train_data)
model_input = recipe.transform_preprocess(test_data)

prediction = model(model_input)

prediction = recipe.inverse_transform_target(prediction)
prediction = recipe.transform_postprocess(prediction)
```

## The `target` phase

`target` holds **pre-fitted** processors whose `inverse_transform` converts
model outputs back to the original space (for example, the same
`StandardScale`/`Power`/`Quantile` instance you fitted on the labels). By
design, `Recipe` has **no `fit_target` method**: fit target-side processors
yourself before passing them in, and every target step must mix in
`InvertibleMixin` or `inverse_transform_target` raises a `TypeError`.

## Inspecting a recipe

Because a recipe is a declared object, you can audit it without running
inference. `describe()` (and `repr`) print the step order per phase:

```python
>>> print(recipe.describe())
Recipe(
  preprocess: Clip -> StandardScale
  target: identity
  postprocess: SoftmaxTemperature
)
```

A `Pipeline` is also `len()`-able and iterable over its steps.

## Behaviors worth knowing

- **Identity fast path.** An empty phase, or one whose steps leave the
  numerical block unchanged, returns the input `TableTensor` object as-is — no
  copy, no rebuild.
- **Numerical-only contract.** Steps see and return only the numerical block;
  the `TableTensor` is rebuilt with its categorical blocks untouched. A step
  that returns a non-`Tensor` raises a `TypeError`.
- **Error context.** Failures are re-raised with the offending phase and step
  position, e.g. `preprocess step 1 (StandardScale): <original message>`,
  which makes multi-step phases easy to debug.

## Available processors

| Processor | Phase(s) | Invertible | Notes |
| --- | --- | --- | --- |
| `StandardScale` | preprocess / target | yes | Per-column center and scale. |
| `Clip` | preprocess | yes (no-op inverse) | Clamp to fitted quantile bounds. |
| `Power` | preprocess / target | yes | Yeo-Johnson power transform. |
| `Quantile` | preprocess / target | yes | Map columns to a uniform/normal distribution. |
| `SoftmaxTemperature` | postprocess | no | Temperature-scaled softmax over logits. |

See {doc}`api/processing` for full signatures and parameters.
