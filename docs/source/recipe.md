# Recipes

A {py:class}`~sdm.processing.Recipe` defines how data crosses a model boundary:
it transforms inputs into the space your model expects and maps the model's
outputs back to the original space, keeping the same auditable transforms on
both sides of the model.

## Concepts

- A **step** is a {py:class}`~sdm.processing.Processor` that transforms a
  {py:class}`~sdm.tensor.TableTensor` and returns a
  {py:class}`~sdm.tensor.TableTensor`. A *stateful* step learns parameters
  when you call `fit` (for example
  {py:class}`~sdm.processing.StandardScale` learns each column's mean and
  standard deviation); a stateless one does not (for example
  {py:class}`~sdm.processing.SoftmaxTemperature`).

- A {py:class}`~sdm.processing.Sequential` is an ordered list of steps.

- A {py:class}`~sdm.processing.Recipe` bundles three pipelines, reached as
  attributes:

  - `features` — model inputs, transformed before the model.
  - `target` — labels, transformed before the model and inverted after it to
    map predictions back to the original space; every step must be invertible.
  - `output` — the model output, an optional forward-only cleanup after the
    target inverse (for example turning logits into probabilities).

## Usage

```python
import torch

from sdm import TableTensor
from sdm.processing import Recipe, StandardScale

recipe = Recipe(features=[StandardScale()], target=[StandardScale()])
```

Fit and transform labeled data in one pass with `recipe.fit_transform`. Use
`recipe.preprocess` to transform validation or test data without refitting.
Recipe pipelines accept and return {py:class}`~sdm.tensor.TableTensor`s.

```python
model_features, model_target = recipe.fit_transform(
    features=labeled_features,
    target=labels,
)
model_input = recipe.preprocess(features=new_features)
```

The target pipeline runs first so its final semantic type can select any
task-dependent processor configured in `recipe.output`.

The model returns a {py:class}`~sdm.tensor.TableTensor`; map its predictions
back to the original space:

```python
prediction = recipe.target.inverse_transform(model(model_input))
prediction = recipe.output.transform(prediction)  # identity if `output` is empty
```

Target steps run in reverse during the inverse, and every one must mix in
{py:class}`~sdm.processing.InvertibleMixin`.
