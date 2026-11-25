# Explainability

The [`sdm.explain`](api/explain) package explains the predictions of an {py:class}`~sdm.models.ICLModel`.
Every explainer is a class with the same entry point: its options go to the constructor, and `explain(model, x_query, ...)` returns the explanation.
Explainers that need more data take it by keyword: `y_query` for the labels of the query rows, `background` for reference rows, `x_context` and `y_context` for in-context examples, and `generator` for randomness.

```{note}
Shapley values are computed and plotted by [`shapiq`](https://shapiq.readthedocs.io), which requires Python 3.12 or later.
Install it via `pip install "structured-data-models[explain]"`.
The other explainers need no additional packages.
```

## Quick Tour

Fit a model, explain the prediction of one row, and plot the contribution of each column:

```python
import sdm
import shapiq
import torch

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df),
    device="cuda",
)
train, test = table[:400], table[400:]
x_train, y_train = train.drop_columns("target"), train["target"]
x_test, y_test = test.drop_columns("target"), test["target"]

model = sdm.models.KumoTabular(task="classification", device="cuda")
model.fit(
    x=x_train,
    y=y_train,
    generator=torch.Generator("cuda").manual_seed(0),
)

explainer = sdm.explain.ShapleyExplainer(
    output="yes",  # The predicted column to explain.
    budget=2**8,  # The number of column subsets per row.
)
values = explainer.explain(model, x_test[:1], background=x_train)

names = [*x_test.numerical_columns, *x_test.categorical_columns]
shapiq.plot.waterfall_plot(values[0], feature_names=names, show=True)
```

The examples below reuse this model and its tables.

## Choosing an Explainer

**Prediction explanations** explain the predictions of individual query rows:

| Explainer                                                 | Question                                                          | Needs besides the model and query rows |
| --------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------- |
| {py:class}`~sdm.explain.GradientExplainer`                | How sensitive is the prediction to a small change of an input?    | Nothing                                |
| {py:class}`~sdm.explain.ShapleyExplainer`                 | How much does each column contribute to the prediction of a row?  | Reference rows                         |
| {py:class}`~sdm.explain.RecontextualizedShapleyExplainer` | The same, for a model that never sees the removed columns.        | In-context examples                    |
| {py:class}`~sdm.explain.ICEExplainer`                     | How does the prediction of a row respond when one column changes? | Nothing                                |
| {py:class}`~sdm.explain.PartialDependenceExplainer`       | How does the mean prediction respond when one column changes?     | Nothing                                |

**Feature importances** measure how much the quality of the predictions of labelled query rows depends on a column:

| Explainer                                     | Question                                                 | Needs besides the model and query rows |
| --------------------------------------------- | -------------------------------------------------------- | -------------------------------------- |
| {py:class}`~sdm.explain.PermutationExplainer` | How much does the fitted model rely on a column?         | Query labels                           |
| {py:class}`~sdm.explain.LOCOExplainer`        | How much does a column improve a model that is given it? | Query labels, in-context examples      |

All explainers except {py:class}`~sdm.explain.GradientExplainer` explain the numerical and categorical feature columns of a {py:class}`~sdm.tensor.TableTensor`, in this order.
See {ref}`explain-inputs` for the tables they accept, and {ref}`explain-cost` for the number of model calls of each explainer.

## Shapley Values

{py:class}`~sdm.explain.ShapleyExplainer` distributes the prediction of each row over its columns.
It evaluates the fitted model on subsets of the columns of a row, and the `imputer` decides how the columns outside a subset are removed:

| `imputer`              | Removed columns are replaced by                     | Base value                               |
| ---------------------- | --------------------------------------------------- | ---------------------------------------- |
| `"baseline"` (default) | The mean or most frequent category of `background`. | The prediction of this reference row.    |
| `"marginal"`           | The values of sampled `background` rows.            | The mean prediction of the sampled rows. |
| `"missing"`            | Missing values, which the model handles itself.     | The prediction of a row without values.  |

```python
explainer = sdm.explain.ShapleyExplainer(output="yes", budget=2**8)
values = explainer.explain(
    model,
    x_test[:64],
    background=x_train,  # Not needed for `imputer="missing"`.
    generator=torch.Generator("cuda").manual_seed(0),
)
```

The Shapley values of a row and its base value sum to its prediction.
They are exact once the `budget` covers all `2 ** D` subsets of the `D` columns, and approximated from sampled subsets below.
The `generator` seeds the sampled subsets and the sampled `background` rows of `"marginal"`.
The `output` names one column of the {py:meth}`~sdm.models.ICLModel.predict` output, *e.g.*, a class for classification or the median `"q500"` for regression.

Each row is returned as `shapiq.InteractionValues`.
Passing `max_order=2` returns k-Shapley interaction values (k-SII) instead, which split the prediction into main effects and pairwise interactions, keyed by the positions of the columns:

```python
explainer = sdm.explain.ShapleyExplainer(
    output="yes",
    budget=2**8,
    max_order=2,
)
value = explainer.explain(model, x_test[:1], background=x_train)[0]
value[(0, 1)]  # The interaction of the first two columns.
```

The `index` selects another interaction index of `shapiq`, and further keyword arguments are passed to `shapiq.TabularExplainer`, *e.g.*, the `sample_size` of `"marginal"`.

### Plotting

The values of each row are `shapiq.InteractionValues`, which `shapiq` plots itself.
Its plots take the names of the columns in the order of their positions, and the plots of several rows also take their feature values as numbers:

```python
shapiq.plot.waterfall_plot(values[0], feature_names=names, show=True)  # One row.

features = sdm.processing.ToNumerical().transform(x_test[:64]).numerical
shapiq.plot.beeswarm_plot(  # All rows.
    values,
    features.cpu().numpy(),  # Categories as codes.
    feature_names=names,
)

shapiq.plot.network_plot(value, feature_names=names, show=True)  # Interactions.
```

## Recontextualization

An in-context model can also remove a column by never seeing it.
{py:class}`~sdm.explain.RecontextualizedShapleyExplainer` drops the columns outside a subset from both the in-context and the query examples, which avoids predictions for imputed rows that the model has no examples for.
It takes the in-context examples instead of a fitted model:

```python
explainer = sdm.explain.RecontextualizedShapleyExplainer(
    output="yes",
    budget=2**8,
)
values = explainer.explain(
    model,
    x_test[:64],
    x_context=x_train,
    y_context=y_train,
    generator=torch.Generator("cuda").manual_seed(0),
)
```

Every subset takes one model call, which predicts all rows.
The base value is the prediction without any column, which all rows share.
Always pass a `generator`: it seeds the sampled subsets and lets all of these calls start from the same state, so that they only differ in their columns.
Further keyword arguments are passed to the model call, *e.g.*, `num_estimators`.
With the arguments and the seed of the fit above, the values of a row and the base value sum to the prediction of the fitted model.

Both explainers return the same kind of values, but answer different questions: {py:class}`~sdm.explain.ShapleyExplainer` explains how the fitted model uses the value of a column, and {py:class}`~sdm.explain.RecontextualizedShapleyExplainer` what the model gains from having the column at all.
Choose between them as follows:

- **Dependent columns:** Imputing one of several dependent columns gives rows whose values do not occur together. Recontextualization never predicts such rows, and lets the model fall back on the remaining columns instead.
- **Cost:** Recontextualization takes `budget` model calls, and each call reads all in-context examples again. {py:class}`~sdm.explain.ShapleyExplainer` takes one model call per row, which reuses the in-context examples of the fitted model, and is therefore much faster for a few rows. The model calls of recontextualization are shared by all rows, so explain all rows of interest at once.

## ICE and Partial Dependence

{py:class}`~sdm.explain.ICEExplainer` sets a column to each value of a grid in all rows.
The predictions of a single row trace its individual conditional expectation (ICE) curve:

```python
explainer = sdm.explain.ICEExplainer(
    output="yes",
    features="age",  # All columns by default.
)
grid, curves = explainer.explain(model, x_test)["age"]  # [G, 1], [G, R]
```

{py:class}`~sdm.explain.PartialDependenceExplainer` takes the same options and returns the mean of these curves over rows, which is the partial dependence of the column:

```python
explainer = sdm.explain.PartialDependenceExplainer(
    output="yes",
    features="age",
)
grid, average = explainer.explain(model, x_test)["age"]  # [G, 1], [G]
```

The grid of a numerical column holds `grid_resolution` evenly spaced values between its 5% and 95% quantiles in the query rows, and the grid of a categorical column holds the categories of the query rows.
Plot the curves against the grid, *e.g.*, with `matplotlib`:

```python
import matplotlib.pyplot as plt

plt.plot(grid.to_pandas()["age"], curves.cpu(), color="gray")  # ICE.
plt.plot(grid.to_pandas()["age"], average.cpu(), color="black")  # Mean.
```

## Feature Importance

{py:class}`~sdm.explain.PermutationExplainer` shuffles a column across rows and reports how much the score of the fitted model decreases.
{py:class}`~sdm.explain.LOCOExplainer` instead leaves the column out of both the in-context and the query examples (leave-one-covariate-out, LOCO), so that the model can fall back on related columns:

```python
permutation = sdm.explain.PermutationExplainer(num_repeats=10).explain(
    model,
    x_test,
    y_query=y_test,
    generator=torch.Generator("cuda").manual_seed(0),
)  # [10, D]
loco = sdm.explain.LOCOExplainer().explain(
    model,
    x_test,
    y_query=y_test,
    x_context=x_train,
    y_context=y_train,
    generator=torch.Generator("cuda").manual_seed(0),
)  # [1, D]
permutation.to_pandas().mean()  # The mean importance of each column.
```

Both return a {py:class}`~sdm.tensor.TableTensor` of score decreases, named by column.
A column that is important under permutation but not when left out shares its information with other columns.
A value at or below zero of {py:class}`~sdm.explain.LOCOExplainer` means that the model predicts these rows no worse without the column.
LOCO is also known as leave-one-feature-out (LOFO) importance, *e.g.*, in the book [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/lofo.html) and in the [`lofo-importance`](https://github.com/aerdem4/lofo-importance) package.
There, the model is retrained without the column, while an in-context model only needs to be called without it.

The `score` is a function that maps the predictions and the targets of the query rows to one number, for which higher is better, and the importance of a column is the decrease of this number, as in [`sklearn.inspection.permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).
By default, it is the negative log loss for a categorical target, and the negative squared error of the mean over all predicted columns otherwise, so that the importances are loss increases.
Both explainers take another `score`, *e.g.*, the accuracy:

```python
def accuracy(pred: sdm.TableTensor, target: sdm.TableTensor) -> torch.Tensor:
    prob, index = sdm.evaluation.to_class_indices(pred, target)  # [R, C], [R]
    return (prob.argmax(dim=-1) == index).float().mean()


explainer = sdm.explain.PermutationExplainer(num_repeats=10, score=accuracy)
permutation = explainer.explain(
    model,
    x_test,
    y_query=y_test,
    generator=torch.Generator("cuda").manual_seed(0),
)  # The decrease of the accuracy.
```

For an error or a loss, for which lower is better, return its negative.

(explain-cost)=

## Budget and Cost

The cost of an explanation is its number of model calls and the number of query rows of each call, for `R` query rows and `D` explained columns:

| Explainer                                                                                  | Model calls                                                    | Query rows per call                            |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ---------------------------------------------- |
| {py:class}`~sdm.explain.GradientExplainer`                                                 | One `predict` call, and one backward pass per predicted column | `R`                                            |
| {py:class}`~sdm.explain.ShapleyExplainer`                                                  | `R + 2` `predict` calls                                        | `budget`, times `sample_size` for `"marginal"` |
| {py:class}`~sdm.explain.RecontextualizedShapleyExplainer`                                  | `budget` one-shot calls                                        | `R`                                            |
| {py:class}`~sdm.explain.ICEExplainer`, {py:class}`~sdm.explain.PartialDependenceExplainer` | One `predict` call per explained column                        | `G * R` for `G` grid values                    |
| {py:class}`~sdm.explain.PermutationExplainer`                                              | `D + 1` `predict` calls                                        | `num_repeats * R`                              |
| {py:class}`~sdm.explain.LOCOExplainer`                                                     | `D + 1` one-shot calls                                         | `R`                                            |

A {py:meth}`~sdm.models.ICLModel.predict` call reuses the in-context examples that the fitted model has cached, while a one-shot {py:meth}`~sdm.models.ICLModel.forward` call reads all of them again.
The `"marginal"` imputer predicts `sample_size` rows (100 by default) for every subset, so lower it or prefer `"baseline"` when the explanation takes too long.

The `budget` of both Shapley explainers trades accuracy for cost:

- A `budget` of `2 ** D` evaluates every subset and gives exact values, and larger budgets evaluate no further subsets. For {py:class}`~sdm.explain.ShapleyExplainer`, only the sampled `background` rows of `"marginal"` then still depend on the `generator`, while the model calls of {py:class}`~sdm.explain.RecontextualizedShapleyExplainer` always do.
- Below `2 ** D`, the values are estimated and get more accurate with the `budget`. Explain a few rows with two different `generator` seeds, and increase the `budget` until their values agree closely enough.
- The values of a row and its base value sum to its prediction for every `budget`.

(explain-inputs)=

## Supported Inputs

- **Tables:** All explainers except {py:class}`~sdm.explain.GradientExplainer` take feature tables as {py:class}`~sdm.tensor.TableTensor`, and expect tables without columns of other semantic types than numerical and categorical, *e.g.*, datetime columns. They report the columns as numerical columns followed by categorical columns, which may differ from the order of the dataframe.
- **Missing values:** Query rows may hold missing values. {py:class}`~sdm.explain.ShapleyExplainer` treats categories that do not occur in `background` as missing.
- **Outputs:** Each explainer with an `output` explains one predicted column, so explain several classes or quantiles with one explainer each.
- **Estimators:** {py:class}`~sdm.explain.GradientExplainer` requires a model with exactly one estimator, and returns gradients with respect to the preprocessed numerical inputs rather than the original columns.

## Fitted Models and In-Context Examples

{py:class}`~sdm.explain.ShapleyExplainer`, {py:class}`~sdm.explain.ICEExplainer`, {py:class}`~sdm.explain.PartialDependenceExplainer` and {py:class}`~sdm.explain.PermutationExplainer` explain a fitted model.

{py:class}`~sdm.explain.RecontextualizedShapleyExplainer` and {py:class}`~sdm.explain.LOCOExplainer` call the model once for every set of columns they keep.
They neither use nor change a fitted model, so pass them the arguments of its fit, *e.g.*, `num_estimators` and a `generator` with the same seed, to explain the same predictions.

{py:class}`~sdm.explain.GradientExplainer` builds on {py:class}`~sdm.explain.ICLExplainer`.
It explains a fitted model as well, and additionally accepts related tables, and `x_context` and `y_context`, which fit the model for the call and clear it afterwards.
