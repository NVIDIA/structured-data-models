# Quick Tour

`structured-data-models` is a GPU-native library of foundation models,
tensor subclasses, and data processors for structured data.

Foundation models via in-context learning are becoming a central direction for learning over structured data.
A foundation model in this domain needs more than the neural network itself: typed data representation, flexible preprocessing pipelines, estimator ensembling, and caching.
`structured-data-models` keeps these concerns explicit while moving the full modeling workflow into GPU-accelerated PyTorch.
In particular, this library provides:

- [**Models**](icl): Reference implementations of structured data foundation models such as the tabular {py:class}`~sdm.models.TabICLv2` and relational {py:class}`~sdm.models.KumoRFM`, built on a unified interface with room for future model families.
- [**Tensor semantics**](tensor): PyTorch-compatible tensor types for numerical, categorical, datetime, text, and relational data.
- [**Data processing**](processing): Composable, extensible, and GPU-accelerated preprocessing and postprocessing for structured data workflows.

Together, these layers provide a common foundation for building, studying, and serving structured data models.

```{figure} images/pipeline_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/pipeline_dark.svg
:figclass: dark-only
:width: 100%
```

## Quick Tour

The typical workflow starts by converting a data frame into a {py:class}`~sdm.tensor.TableTensor`.
Semantic types ({py:class}`~sdm.Stype`) describe the role of each column, such that the model can distinguish numerical features from categorical, datetime, or text:

```python
import sdm

table = sdm.TableTensor.from_pandas(
    df,
    stypes = sdm.infer_stypes(df),
    device="cuda",
)
```

Models then consume these tensor containers through a shared in-context learning interface.
The same model object supports direct one-shot calls and cached fit/predict execution for efficient context re-use:

```python
model = sdm.models.TabICLv2(device="cuda")

out = model(
    x_context=table[:300].drop_columns("target"),
    y_context=table[:300, "target"],
    x_query=table[300:].drop_columns("target"),
    num_estimators=8,
)

model.fit(
    x=table[:300].drop_columns("target"),
    y=table[:300, "target"],
    num_estimators=8,
)
model.predict(table[300:].drop_columns("target"))
model.clear()
```

Preprocessing and postprocessing are defined by an explicit {py:class}`~sdm.processing.recipe.Recipe`.
A {py:class}`~sdm.processing.recipe.Recipe` denotes how features and targets are transformed before model execution, and how outputs are postprocessed afterwards, making model-specific data handling inspectable, extensible and replaceable.
The resulting {py:class}`~sdm.processing.recipe.Recipe` can be passed to the model's {py:meth}`~sdm.models.ICLModel.forward` or {py:meth}`~sdm.models.ICLModel.fit` calls:

```python
import sdm.processing as sp

recipe = sp.Recipe(
    features=[
        sp.StypeDispatch(
            numerical=[
                sp.ImputeMean(),
                sp.Standardize(),
            ],
            categorical=[
                sp.AlignCategories(),
            ],
        ),
    ],
    target=[
        sp.StypeDispatch(
            categorical=[
                sp.AlignCategories(),
                sp.ShuffleCategories(),
            ],
            numerical=sp.Standardize(),
        ),
    ],
    output=[
        sp.ReduceEstimators(method="mean"),
        sp.TaskDispatch(
            classification=sp.Softmax(temperature=0.9),
        ),
    ],
)

model.fit(..., recipe=recipe)
```

Recipes gives foundation models a unified way to handle a wide range of semantic column types.
For example, datetime columns can be expanded into calendar features, and text columns can be embedded via an LLM.
These components are reusable across model families rather than tied to a single model implementation, and designed for ensemble-aware, GPU accelerated execution across the entire processing stack.

For relational tasks, {py:class}`~sdm.relational.RelatedTables` allows foundation models to attach to surrounding tables and relationships.
Models that support relational context, such as {py:class}`~sdm.models.KumoRFM`, can process these {py:class}`~sdm.relational.RelatedTables` through the same {py:meth}`~sdm.models.ICLModel.fit`, {py:meth}`~sdm.models.ICLModel.predict`, {py:class}`~sdm.processing.recipe.Recipe`, and {py:class}`~sdm.tensor.TableTensor` interfaces.
