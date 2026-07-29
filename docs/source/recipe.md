# Recipes

A {py:class}`~sdm.processing.recipe.Recipe` defines how data crosses a model boundary: it transforms inputs into the space a model expects and maps the model's outputs back to the original space, keeping the same auditable transforms on both sides of the model.

## Concepts

The basic unit of a recipe is a {py:class}`~sdm.processing.base.Processor`.
A list of all available processors grouped by their domain and semantic type is outlined in the [API reference](api/processing).

- A {py:class}`~sdm.processing.base.Processor` transforms a {py:class}`~sdm.tensor.TableTensor` and returns a new {py:class}`~sdm.tensor.TableTensor`.
- A **stateful** {py:class}`~sdm.processing.base.Processor` learns state when you call {py:meth}`~sdm.processing.base.Processor.fit` (*e.g.*, {py:class}`~sdm.processing.numerical.Standardize` learns each column's mean and standard deviation); a **stateless** one does not (*e.g.*, {py:class}`~sdm.processing.output.Softmax`).

A {py:class}`~sdm.processing.base.Processor` is fully composable:

- A {py:class}`~sdm.processing.common.Sequential` processor applies a sequence of processors.
- A {py:class}`~sdm.processing.common.StypeDispatch` processor applies a processor per [semantic type](api/generated/sdm.Stype).
- A {py:class}`~sdm.processing.common.TaskDispatch` processor applies a processor per task (*e.g.*, classification or regression).

Processors let you define powerful recipes that manage the full pre-processing pipeline of input features and targets, as well as post-processing pipelines of model outputs.
In particular:

- [`Recipe.features`](api/generated/sdm.processing.recipe.Recipe): a {py:class}`~sdm.processing.base.Processor` that operates on model inputs, transformed before the model.
- [`Recipe.target`](api/generated/sdm.processing.recipe.Recipe): a {py:class}`~sdm.processing.base.Processor` that operates on targets, transformed before the model. For regression tasks, it is also used to invert model outputs back to their original space.
- [`Recipe.output`](api/generated/sdm.processing.recipe.Recipe): a {py:class}`~sdm.processing.base.Processor` that operates on model outputs (after the target inverse in regression tasks), *e.g.*, to reduce outputs from multiple estimators or to turn logits into probabilities.

## Usage

Each model defines a default recipe that closely mimics pre- and postprocessing routines of its official implementation (*e.g.*, take a look at the default recipe of {py:class}`~sdm.models.TabICLv2`).

Recipes are plain Python objects, so they can be inspected, copied and modified.
This makes it easy to keep the default model contract while changing one part of the pipeline.
For example, {py:class}`~sdm.models.TabICLv2` does not consume raw {py:attr}`~sdm.Stype.datetime` columns directly.
To support {py:attr}`~sdm.Stype.datetime` inputs, you can, *e.g.*, add a {py:attr}`~sdm.Stype.datetime` branch to the recipe that expands timestamps into numerical calendar features before running the rest of the default feature pipeline:

```python
from sdm.models import TabICLv2
from sdm.processing import AddCalendarFields, StypeDispatch

recipe = TabICLv2.default_recipe()
recipe.features = StypeDispatch(
    datetime=AddCalendarFields(
        fields=("minute", "hour", "weekday", "day_of_month", "month"),
    )
) + recipe.features
```

You can also define a recipe from scratch when you want full control over the
transformations applied to features, targets, and outputs:

```python
from sdm.processing import *

recipe = Recipe(
    # First impute missing values, then standardize:
    features=[ImputeMean(), Standardize()],

    target=StypeDispatch(
        # Align and shuffle classes for classification:
        categorical=[
            AlignCategories(),
            ShuffleCategories(),
        ],
        # Standardize the targets for regression:
        numerical=Standardize(),
    ),

    output=TaskDispatch(
        # Convert logits to probabilities for classification tasks:
        classification=Softmax(temperature=0.9),
    ),
)
```

When a custom recipe is passed to an {py:class}`~sdm.models.ICLModel`, the model applies the feature, target, and output pipelines automatically at the appropriate points in its execution.

```python
model = TabICLv2(device="cuda")
model(..., recipe=recipe)
```

You can also call the individual processors directly to inspect intermediate representations:

```python
recipe.features.fit(table)
table = recipe.features.transform(table)
```

## Ensembling
