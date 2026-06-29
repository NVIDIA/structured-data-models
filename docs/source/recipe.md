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

- A **`Recipe`** groups three named pipelines, called **phases**, around the
  point where you call your model:

  | Phase         | Runs             | Purpose                                                       |
  | ------------- | ---------------- | ------------------------------------------------------------- |
  | `preprocess`  | before the model | Prepare features for the model.                               |
  | `target`      | after the model  | Invert target-side transforms (model space → original space). |
  | `postprocess` | after `target`   | Shape-preserving cleanup of model output.                     |

The recipe executes them in a fixed order around inference:

```text
preprocess  ->  model  ->  target (inverse)  ->  postprocess
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

Fit `preprocess` on training data, then apply the recipe around the model.
`target` runs its steps in **reverse** via `inverse_transform_target`, and
`postprocess` runs last.

```python
# Fit on train, then transform any split with the same fitted state.
training_table = recipe.fit_transform_preprocess(train_data)
model_input = recipe.transform_preprocess(test_data)

prediction = model(model_input)

prediction = recipe.inverse_transform_target(prediction)
prediction = recipe.transform_postprocess(prediction)
```

## The `target` phase

`target` holds **pre-fitted, invertible** processors — for example the same
`StandardScale`/`Power`/`Quantile` instance you fitted on your labels. The
recipe runs **only their inverse**, after the model, to map predictions back to
the original space.

The forward direction is yours to own: you fit these processors on the labels
and transform your training targets yourself, *before* building the recipe.
By design `Recipe` has **no `fit_target` method**, and every target step must
mix in `InvertibleMixin` or `inverse_transform_target` raises a `TypeError`.

## Available processors

`StandardScale`, `Clip`, `Power`, `Quantile`, and `SoftmaxTemperature` ship in
`sdm.processing`. See {doc}`api/processing` for the full list, signatures, and
which steps are invertible.
