# Recipes

A {py:class}`~sdm.processing.Recipe` is an inspectable definition of how data
crosses a model boundary: it transforms inputs into the space your model
expects and maps the model's outputs back to the original space. Defining it
once keeps the same auditable transforms on both sides of the model.

## The building blocks

- A **step** is a single {py:class}`~sdm.processing.Processor` that transforms
  the **numerical block** of a {py:class}`~sdm.tensor.TableTensor`. A step can
  be *stateful* — it learns parameters from data when you call `fit` (for
  example {py:class}`~sdm.processing.StandardScale` learns each column's mean
  and standard deviation) — or *stateless*, needing no fitting (for example
  {py:class}`~sdm.processing.SoftmaxTemperature`).

- A {py:class}`~sdm.processing.Pipeline` is an ordered list of steps applied to
  the numerical block; categorical columns pass through unchanged. **Order
  matters**: each step is fitted on the previous step's output, and most
  transforms do not commute (see [Ordering](#ordering)).

- A {py:class}`~sdm.processing.Recipe` groups three pipelines by the role their
  data plays around the model. You reach each one as an attribute
  (`recipe.features`, `recipe.target`, `recipe.output`) and call its
  `fit`/`transform`/`fit_transform`/`inverse_transform` methods directly.

## The three pipelines

- **`features`** — your model inputs. Transformed once, on the way *into* the
  model.
- **`target`** — your labels. Transformed forward into model space on the way
  in, then inverted on the way out to map predictions back to the original
  label space. Every step must be invertible.
- **`output`** — the model's raw output. An optional, forward-only cleanup
  applied after the target inverse (for example turning logits into
  probabilities). It is the identity if you leave it empty.

```text
into the model:   recipe.features.transform(x)   +   recipe.target.fit_transform(y)
out of the model: model output -> recipe.target.inverse_transform -> recipe.output.transform
```

## Building a recipe

Pass each pipeline a list of {py:class}`~sdm.processing.Processor` steps; omit
any you don't need (it defaults to the identity). Stateful steps are fitted on
your **labeled data only**, so no held-out statistics leak in.

```python
from sdm.processing import Recipe, StandardScale, SoftmaxTemperature

recipe = Recipe(
    features=[StandardScale()],
    target=[StandardScale()],       # target steps must be invertible
    output=[SoftmaxTemperature()],  # optional cleanup of the model output
)
```

## Model inputs

Fit the {py:class}`~sdm.processing.Recipe` on your labeled data and transform it
in one call with {py:meth}`~sdm.processing.Recipe.fit_transform`; transform
later inputs with `recipe.features.transform` (no re-fit). All inputs and
outputs are {py:class}`~sdm.tensor.TableTensor`s.

```python
# fit on labeled data, then hand to the model
model_features, model_target = recipe.fit_transform(labeled_features, labels)

# transform new inputs with the already-fitted recipe
model_input = recipe.features.transform(new_features)
```

## Model outputs

Run the model, then map its predictions back to the original space. Because the
target inverse reads and rewrites the numerical block, the model is expected to
return a {py:class}`~sdm.tensor.TableTensor` (wrap a raw tensor with, e.g.,
{py:meth}`~sdm.tensor.TableTensor.from_tensor` first):

```python
prediction = model(model_input)                       # a TableTensor
prediction = recipe.target.inverse_transform(prediction)
prediction = recipe.output.transform(prediction)      # identity if `output` is empty
```

`recipe.target.inverse_transform` runs the target steps in reverse, and every
one must mix in {py:class}`~sdm.processing.InvertibleMixin` — otherwise it
raises `TypeError`.

## Ordering

Steps are not interchangeable. Each stateful step is fitted on the output of
the steps before it, and transforms such as
{py:class}`~sdm.processing.Clip`, {py:class}`~sdm.processing.Power`, and
{py:class}`~sdm.processing.Quantile` are nonlinear, so reordering changes the
result. A typical feature pipeline runs from cleanup to shaping:

```python
from sdm.processing import Clip, MeanImpute, StandardScale

features = [MeanImpute(), Clip(q_low=0.01, q_high=0.99), StandardScale()]
```

Impute missing values first, clip outliers next, then scale — so the scale is
not driven by missing entries or outliers.

## Available processors

{py:class}`~sdm.processing.StandardScale`, {py:class}`~sdm.processing.Clip`,
{py:class}`~sdm.processing.Power`, {py:class}`~sdm.processing.Quantile`,
{py:class}`~sdm.processing.SigmaClip`, {py:class}`~sdm.processing.MeanImpute`,
and {py:class}`~sdm.processing.SoftmaxTemperature` ship in `sdm.processing`. See
{doc}`api/processing` for signatures and which steps are invertible.
