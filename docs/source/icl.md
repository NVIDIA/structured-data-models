# In-Context Learning

In-Context Learning (ICL) treats predictions on structured data (*e.g.*, tabular, relational, time series) as an inference-time task: a model receives labeled context rows together with unlabeled query rows, and predicts the query targets without updating its weights.
The `structured-data-models` package groups and unifies such foundation models behind a single interface:

1. Convert dataframe-like data into a {py:class}`~sdm.tensor.TableTensor` (see [here](tensor) for the accompanying tutorial).
2. Optionally define pre- and post-processing routines through a {py:class}`~sdm.processing.recipe.Recipe` (see [here](processing) for the accompanying tutorial).
3. Instantiate and run an in-context model from [`sdm.models`](api/models).
4. Receive predictions as a {py:class}`~sdm.tensor.TableTensor`.

```{figure} images/pipeline_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/pipeline_dark.svg
:figclass: dark-only
:width: 100%
```

## The ICL Interface

A In-Context Learning task is described by **context features**, **context targets**, and **query features**, while model-specific details are handled by the model implementation and its pre- and post-processing {py:class}`~sdm.processing.recipe.Recipe`.
As a result, single-table and multi-table models can share the same prediction workflow while consuming different forms of context.

Specifically, an in-context learning task has three core inputs, as defined in the {py:class}`~sdm.models.ICLModel` base class:

- `x_context` ({py:class}`~sdm.tensor.TableTensor` | {py:class}`torch.Tensor`): feature rows with observed targets.
- `y_context` ({py:class}`~sdm.tensor.TableTensor` | {py:class}`torch.Tensor`): the target column for those context rows.
- `x_query` ({py:class}`~sdm.tensor.TableTensor` | {py:class}`torch.Tensor`): feature rows whose targets should be predicted.

The context rows are not used to update model weights.
They are examples supplied at inference time, and the model predicts query rows by attending to that labeled context:

```python
from sklearn.datasets import load_breast_cancer

import sdm

df = load_breast_cancer(as_frame=True).frame

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df),
    device="cuda",
)

model = sdm.models.TabICLv2(device="cuda")
out = model(
    x_context=table[:300].drop_columns("target"),
    y_context=table[:300, "target"],
    x_query=table[300:].drop_columns("target"),
)
```

This one-shot {py:meth}`~sdm.models.ICLModel.forward` call is the most direct form of the interface.
When the same context is reused for many query batches, call {py:meth}`~sdm.models.ICLModel.fit` once and then call {py:meth}`~sdm.models.ICLModel.predict` for each query batch.
This records reusable model state, including key/value projections, and avoids recomputing the context side of the model for every prediction:

```python
model.fit(
    x=table[:300].drop_columns("target"),
    y=table[:300, "target"],
)
for batch in table[300:].drop_columns("target").split(batch_size):
    out = model.predict(batch)
model.clear()
```

The cached interface has the same prediction contract as the one-shot call.
Use one-shot {py:meth}`~sdm.models.ICLModel.forward` calls for one-time calls when tasks change frequently, and use the {py:meth}`~sdm.models.ICLModel.fit`+{py:meth}`~sdm.models.ICLModel.predict` flow for large batch predictions over a single fixed task.

## Model Concepts

Structured data foundation models are not bound to a specific task type.
The same {py:class}`~sdm.models.ICLModel` instance can, *e.g.*, consume both classification and regression targets, depending on the input data it receives.
The {py:attr}`~sdm.models.ICLModel.supported_target_stypes` attribute of an {py:class}`~sdm.models.ICLModel` denotes which target semantic types the model accepts.
For example, a {py:attr}`~sdm.Stype.categorical` target is treated as classification, while a {py:attr}`~sdm.Stype.numerical` target is treated as regression.

Similarly, there are no task-specific constructor arguments when creating an {py:class}`~sdm.models.ICLModel`.
Instead, task behavior is determined at the data boundary.
A single {py:class}`~sdm.processing.recipe.Recipe` can express task-specific pre- and post-processing through {py:class}`~sdm.processing.common.StypeDispatch` and {py:class}`~sdm.processing.common.TaskDispatch`.
While each model provides a default recipe, users can fully control or modify how inputs and outputs are processed (see [here](processing) for more information).

Lastly, the {py:attr}`~sdm.models.ICLModel.supported_feature_stypes` attribute denotes which feature semantic types a model accepts after pre-processing.
Semantic types outside this set need to be converted, dropped, or otherwise handled by the recipe before they reach the model.
For example, {py:class}`~sdm.models.TabICLv2` can only consume numerical features internally, so it is the recipe's job to convert any other semantic type into a numerical representation before it reaches the model, *e.g.*, via {py:class}`~sdm.processing.common.ToNumerical` on categorical columns.

## Ensembling

The interface of an {py:class}`~sdm.models.ICLModel` additionally supports estimator ensembling through the `num_estimators` argument in {py:meth}`~sdm.models.ICLModel.forward` and {py:meth}`~sdm.models.ICLModel.fit`.
When a recipe contains stochastic processors, such as {py:class}`~sdm.processing.common.ShuffleColumns`, pre-processing produces different transformed views of the same task, and model outputs are stacked for post-processing.

## Model Outputs

Predictions are returned as a general {py:class}`~sdm.tensor.TableTensor`, where the output schema depends on the task, model, and post-processing routine of the {py:class}`~sdm.processing.recipe.Recipe`:

- Classification predictions are generally returned as numerical probabilities, where each target category corresponds to one column in the output.
- Regression predictions are generally model-dependent.
  For example, {py:class}`~sdm.models.TabICLv2` outputs numerical quantiles named `"q001"` through `"q999"`, from which the (approximate) mean prediction can be derived via `out.numerical.mean(dim=-1)` and the median prediction is available via `out["q500"].numerical`.

Since predictions are returned as {py:class}`~sdm.tensor.TableTensor`, you can zero-copy them to [`pandas`](https://pandas.pydata.org/docs), [`arrow`](https://arrow.apache.org/docs), or [`cudf`](https://docs.rapids.ai/api/cudf) via {py:meth}`~sdm.tensor.TableTensor.to_pandas`, {py:meth}`~sdm.tensor.TableTensor.to_arrow`, {py:meth}`~sdm.tensor.TableTensor.to_cudf` for further downstream processing.

In order to simplify metric calculation (*e.g.*, via [`torchmetrics`](https://lightning.ai/docs/torchmetrics)), we provide helper functions in the [`sdm.evaluation`](api/evaluation) package to align and convert prediction columns back to class indices (see {py:func}`~sdm.evaluation.to_class_indices` and {py:func}`~sdm.evaluation.to_binary_class`).
