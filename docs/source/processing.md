# Data Processing

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
To support {py:attr}`~sdm.Stype.datetime` inputs, you can, *e.g.*, prepend a {py:attr}`~sdm.Stype.datetime` branch to the recipe that expands timestamps into numerical calendar features before running the rest of the default feature pipeline:

```python
import sdm
import sdm.processing as sp

recipe = sdm.models.TabICLv2.default_recipe().prepend_features(
    sp.StypeDispatch(
        datetime=sp.AddCalendarFields(
            fields=("minute", "hour", "weekday", "day_of_month", "month"),
        )
    )
)
```

You can also define a recipe from scratch when you want full control over the
transformations applied to features, targets, and outputs:

```python
import sdm
import sdm.processing as sp

recipe = sp.Recipe(
    # First impute missing values, then standardize:
    features=[sp.ImputeMean(), sp.Standardize()],

    target=sp.StypeDispatch(
        # Align and shuffle classes for classification:
        categorical=[
            sp.AlignCategories(),
            sp.ShuffleCategories(),
        ],
        # Standardize the targets for regression:
        numerical=sp.Standardize(),
    ),

    output=sp.TaskDispatch(
        # Convert logits to probabilities for classification tasks:
        classification=sp.Softmax(temperature=0.9),
    ),
)
```

When a custom recipe is passed to an {py:class}`~sdm.models.ICLModel`, the model applies the feature, target, and output pipelines automatically at the appropriate points in its execution.

```python
model = sdm.models.TabICLv2(device="cuda")
model(..., recipe=recipe)
```

You can also call the individual processors directly to inspect intermediate representations:

```python
recipe.features.fit(table)
table = recipe.features.transform(table)
```

## Ensembling

A model can run several ensemble members over the same task.
Most preprocessing steps do not need to distinguish those members: imputing a mean, standardizing a column, or converting categories to numbers often produces the same transformed table for every estimator.
Other steps intentionally create member-specific views to add variance to the input data, such as drawing a different column permutation, choosing a different numerical transform, or shuffling categorical values.

The `structured-data-models` package takes advantage of this to avoid duplicating work: common transformations can run once over shared member groups, while only estimator-specific variations split the ensemble into a tree of member views as the {py:class}`~sdm.processing.recipe.Recipe` is applied.

```{figure} images/ensemble_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/ensemble_dark.svg
:figclass: dark-only
:width: 100%
```

This shared-by-default, split-when-needed behavior is captured by {py:class}`~sdm.processing.ensemble.EnsembleProcessor`.
An {py:class}`~sdm.processing.ensemble.EnsembleProcessor` is a regular {py:class}`~sdm.processing.base.Processor` that operates on an {py:class}`~sdm.EnsembleTable`, allowing a processing step to split ensemble members into separate groups when their transformed views diverge.
This lets a {py:class}`~sdm.processing.recipe.Recipe` stay shared by default and branch only at steps that actually introduce member-specific behavior.

The underlying {py:class}`~sdm.EnsembleTable` stores members by layout rather than by estimator.
Members that see the same table share storage, while compatible member tables are stacked into one leading dimension of a single {py:class}`~sdm.tensor.TableTensor`.
Regular processors can therefore operate on whole groups, leveraging PyTorch vectorization and GPU parallelism, and only fall back to separate groups when storage layout diverges.

In short, a plain {py:class}`~sdm.processing.base.Processor` describes a transformation for one table or one group, and an {py:class}`~sdm.processing.ensemble.EnsembleProcessor` describes a transformation over the collection of estimator views.
Most {py:class}`~sdm.processing.recipe.Recipe` steps therefore remain ordinary processors.

For most users, ensemble processing is an internal (but cool) detail of model execution.
Recipes can be composed from ordinary processors, and the {py:class}`~sdm.processing.recipe.Recipe` execution handles ensemble grouping, sharing, and branching when a model runs with multiple estimators.
You only need to reason about {py:class}`~sdm.processing.ensemble.EnsembleProcessor` directly when writing a processor whose behavior can change table layout.
Examples of those include {py:class}`~sdm.processing.common.Choice`, {py:class}`~sdm.processing.common.ShuffleColumns`, and {py:class}`~sdm.processing.categorical.ShuffleCategories`.
