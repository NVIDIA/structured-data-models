# Recipes

A {py:class}`~sdm.processing.Recipe` defines how data crosses a model boundary:
it transforms inputs into the space your model expects and maps the model's
outputs back to the original space, keeping the same auditable transforms on
both sides of the model.

## Concepts

- A **step** is a {py:class}`~sdm.processing.Processor` that transforms the
  **numerical block** of a {py:class}`~sdm.tensor.TableTensor`; categorical
  columns pass through unchanged. A *stateful* step learns parameters when you
  call `fit` (for example {py:class}`~sdm.processing.StandardScale` learns each
  column's mean and standard deviation); a stateless one does not (for example
  {py:class}`~sdm.processing.SoftmaxTemperature`).

- A {py:class}`~sdm.processing.Pipeline` is an ordered list of steps.

- A {py:class}`~sdm.processing.Recipe` bundles three pipelines, reached as
  attributes:

  - `features` — model inputs, transformed before the model.
  - `target` — labels, transformed before the model and inverted after it to
    map predictions back to the original space; every step must be invertible.
  - `output` — the model output, an optional forward-only cleanup after the
    target inverse (for example turning logits into probabilities).

## Usage

```python
from sdm.processing import Recipe, StandardScale

recipe = Recipe(features=[StandardScale()], target=[StandardScale()])
```

Fit the {py:class}`~sdm.processing.Recipe` on your labeled data and transform it
in one call with {py:meth}`~sdm.processing.Recipe.fit_transform`; transform
later inputs with `recipe.features.transform` (no re-fit). All values are
{py:class}`~sdm.tensor.TableTensor`s.

```python
model_features, model_target = recipe.fit_transform(labeled_features, labels)
model_input = recipe.features.transform(new_features)
```

The model returns a {py:class}`~sdm.tensor.TableTensor`; map its predictions
back to the original space:

```python
prediction = recipe.target.inverse_transform(model(model_input))
prediction = recipe.output.transform(prediction)  # identity if `output` is empty
```

Target steps run in reverse during the inverse, and every one must mix in
{py:class}`~sdm.processing.InvertibleMixin`.

## Available processors

{py:class}`~sdm.processing.StandardScale`, {py:class}`~sdm.processing.Clip`,
{py:class}`~sdm.processing.Power`, {py:class}`~sdm.processing.Quantile`,
{py:class}`~sdm.processing.SigmaClip`, {py:class}`~sdm.processing.MeanImpute`,
and {py:class}`~sdm.processing.SoftmaxTemperature` ship in `sdm.processing`. See
{doc}`api/processing` for signatures and which steps are invertible.
