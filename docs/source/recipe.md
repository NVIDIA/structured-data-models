# Recipes

A `Recipe` is an inspectable definition of how data crosses a model
boundary: it transforms inputs into the space your model expects and maps
the model's outputs back to the original space. Defining it once keeps the
same auditable transforms on both sides of the model.

## Mental model

- A **`Pipeline`** is an ordered list of `steps` (each a `Processor`) applied to
  the **numerical block** of a `TableTensor`. Categorical columns pass through
  unchanged.

- A **`Recipe`** groups three pipelines — **phases** — named after the data they
  act on. Every phase method takes a `TableTensor` and returns a `TableTensor`.

  | Phase      | Acts on      | Before the model  | After the model                                  |
  | ---------- | ------------ | ----------------- | ------------------------------------------------ |
  | `features` | model inputs | forward transform | —                                                |
  | `target`   | labels       | forward transform | inverse transform (predictions back to original) |
  | `output`   | model output | —                 | forward transform (e.g. logits to probabilities) |

```text
inputs:   features.forward + target.forward  ->  model
outputs:  model  ->  target.inverse  ->  output.forward
```

## Building a recipe

Pass each phase a list of `Processor` steps; omit any you don't need (it
defaults to identity). Stateful steps are fitted on your **labeled data only**,
so no held-out statistics leak in.

```python
from sdm.processing import Recipe, StandardScale

recipe = Recipe(
    features=[StandardScale()],
    target=[StandardScale()],  # target steps must be invertible
)
```

## Model inputs

Fit the recipe on your labeled data and transform it in one call with
`fit_transform`; transform later inputs with `transform_features` (no re-fit).
All inputs are `TableTensor`s.

```python
# fit on labeled data, then hand to the model
model_features, model_target = recipe.fit_transform(labeled_features, labels)

# transform new inputs with the already-fitted recipe
model_input = recipe.transform_features(new_features)
```

## Model outputs

Run the model, then map its predictions back to the original space:

```python
prediction = model(model_input)
prediction = recipe.inverse_transform_target(prediction)
prediction = recipe.transform_output(prediction)  # identity if `output` is empty
```

`target` steps run in reverse during the inverse, and every one must mix in
`InvertibleMixin` — otherwise `inverse_transform_target` raises `TypeError`.

## Available processors

`StandardScale`, `Clip`, `Power`, `Quantile`, and `SoftmaxTemperature` ship in
`sdm.processing`. See {doc}`api/processing` for signatures and which steps are
invertible.
