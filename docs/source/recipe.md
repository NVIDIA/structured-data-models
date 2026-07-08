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

{py:class}`~sdm.processing.ConstantFilter` learns which numerical,
categorical, and datetime columns have too few unique values and removes them
while keeping table blocks and column metadata aligned. Put it early in the
features pipeline, before transforms that should fit the filtered feature
space.

{py:meth}`~sdm.processing.Processor.resolve` returns the concrete processor
for a processing context. Plain processors return themselves; processors that
depend on a view or estimator can override it. For example,
{py:class}`~sdm.processing.FeaturePermute` preserves the single-estimator
behavior when used directly and becomes a concrete non-identity view after
`resolve(estimator=...)`. It accepts numerical-only tables; place
{py:class}`~sdm.processing.ToNumerical` before it when features include
categorical columns:

```python
from sdm.processing import FeaturePermute, LabelShuffle

feature_view = FeaturePermute(method="shift").resolve(estimator=1)
target_view = LabelShuffle(method="shift").resolve(estimator=1)
```

{py:class}`~sdm.processing.LabelShuffle` maps integer-encoded numerical target
labels in the target pipeline. Use
{py:meth}`~sdm.processing.LabelShuffle.correct_output` to restore class-score
outputs to the original class order before ensemble averaging; target-pipeline
`inverse_transform` maps label ids, not class-score tensors.

Fit the recipe pipelines on your labeled data and transform them in one call
with `fit_transform`; transform later inputs with `recipe.features.transform`
(no re-fit). Recipe pipelines accept and return
{py:class}`~sdm.tensor.TableTensor`s.

```python
model_features = recipe.features.fit_transform(labeled_features)
model_target = recipe.target.fit_transform(labels)
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
