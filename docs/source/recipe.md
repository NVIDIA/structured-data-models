# Recipes

A `Recipe` is an inspectable, deterministic processing *contract* around a
model boundary. It bundles the transforms a model expects on its input with the
transforms that turn the model's output back into the original space, so
training and inference share one auditable definition instead of ad-hoc
preprocessing scattered across call sites.

## Mental model

There are two objects:

- A **`Pipeline`** is an ordered sequence of `steps` applied to the
  **numerical block** of a `TableTensor`. Categorical blocks pass through
  unchanged.

- A **`Recipe`** groups three named pipelines, called **phases**, named after
  the data each one operates on:

  | Phase      | Operates on  | Forward (training)                | Inverse (inference)                       |
  | ---------- | ------------ | --------------------------------- | ----------------------------------------- |
  | `features` | model inputs | transform before the model        | — (never inverted)                        |
  | `target`   | labels       | transform labels before the model | invert predictions back to original space |
  | `output`   | model output | — (shape-preserving cleanup)      | runs after the target inverse             |

The recipe spans the model boundary like this:

```text
training:    features (forward) + target (forward)  ->  model
inference:   features (forward)  ->  model  ->  target (inverse)  ->  output
```

Each step is a `Processor` (see {doc}`api/processing`): a fittable,
tensor-in/tensor-out transform with `fit`, `transform`, and (optionally)
`inverse_transform`.

## Building a recipe

Construct each phase from a list of steps. Stateful steps must be fitted before
they transform data — fit them on the **training** split only, then reuse the
fitted recipe on validation/test data so no test statistics leak in.

```python
from sdm.processing import Clip, Recipe, SoftmaxTemperature, StandardScale

recipe = Recipe(
    features=[
        Clip(q_low=0.01, q_high=0.99),
        StandardScale(),
    ],
    target=[StandardScale()],
    output=[SoftmaxTemperature(temperature=2.0)],
)
```

Phases default to an empty (identity) pipeline, so you only pass the ones you
need. A plain list is coerced into a `Pipeline` automatically.

## Running the full lifecycle

At **training** time, fit and forward-transform both features and labels. The
`fit_transform` convenience does both in one call:

```python
model_features, model_target = recipe.fit_transform(train_features, train_labels)
model.fit(model_features, model_target)
```

At **inference** time, transform features, call the model, then invert the
target and clean up the output. `target` runs its steps in **reverse** via
`inverse_transform_target`, and `output` runs last:

```python
model_input = recipe.transform_features(test_features)

prediction = model(model_input)

prediction = recipe.inverse_transform_target(prediction)
prediction = recipe.transform_output(prediction)
```

## The `target` phase

`target` is two-directional and lives entirely in the recipe:

- **Forward (training).** `fit_transform_target` fits the steps on the labels
  and maps them into the space the model predicts — for example scaling
  regression targets with `StandardScale`, or applying `Power`/`Quantile`.
- **Inverse (inference).** `inverse_transform_target` runs those same fitted
  steps in reverse to map predictions back to the original space.

Because the inverse is part of the contract, **every target step must mix in
`InvertibleMixin`** or `inverse_transform_target` raises a `TypeError`.

## Available processors

`StandardScale`, `Clip`, `Power`, `Quantile`, and `SoftmaxTemperature` ship in
`sdm.processing`. See {doc}`api/processing` for the full list, signatures, and
which steps are invertible.
