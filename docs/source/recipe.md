# Recipes

A `Recipe` is an inspectable definition of how data crosses a model
boundary: it transforms inputs into the space your model expects and turns
the model's outputs back into the original space. Defining it once means
training and inference share the same auditable transforms.

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
train:      features.forward + target.forward   ->   model
inference:  features.forward  ->  model  ->  target.inverse  ->  output.forward
```

## Building a recipe

Pass each phase a list of `Processor` steps; omit any you don't need (it
defaults to identity). Stateful steps are fitted on the **training split only**,
so no validation/test statistics leak in.

```python
from sdm.processing import Recipe, StandardScale

recipe = Recipe(
    features=[StandardScale()],
    target=[StandardScale()],  # target steps must be invertible
)
```

## Training

`fit_transform` fits the `features` and `target` phases on the training tables
and returns the transformed `(features, target)`. Both `train_features` and
`train_labels` are `TableTensor`s.

```python
model_features, model_target = recipe.fit_transform(train_features, train_labels)
train_your_model(model_features, model_target)  # your model and training loop
```

## Inference

Reuse the fitted recipe — transform features (no re-fit), call the model, then
invert the target to return predictions to their original units.

```python
model_input = recipe.transform_features(test_features)  # TableTensor in and out
prediction = your_model(model_input)                     # predictions as a TableTensor
prediction = recipe.inverse_transform_target(prediction)
prediction = recipe.transform_output(prediction)         # identity if `output` is empty
```

`target` steps run in reverse during the inverse, and every one must mix in
`InvertibleMixin` — otherwise `inverse_transform_target` raises `TypeError`.

## Available processors

`StandardScale`, `Clip`, `Power`, `Quantile`, and `SoftmaxTemperature` ship in
`sdm.processing`. See {doc}`api/processing` for signatures and which steps are
invertible.
