# Time Series Forecasting

Time series forecasting predicts future values of a series from its observed history.
The `structured-data-models` package allows forecasting through the same in-context learning interface as all other models (see the [ICL guide](icl)): the observed history is the context, and the future time steps are the queries.
This guide starts with a single series and adds one ingredient at a time: past-only covariates, known future covariates, multiple targets, ensembling, and finally batches of series.

## The Forecasting Task

A time series is a sequence of observations ordered in time.
Its variables are either **targets**, which are forecast, or **covariates**, which inform the forecast.
Covariates differ in when they are known:

- *Past-only covariates* are only observed in the past (*e.g.*, the measured temperature).
- *Known future covariates* are available in the past and the future (*e.g.*, planned promotions or weather forecasts).

The number of future steps to forecast is the **forecast horizon** `H`.

Throughout this guide, we forecast the daily sales of ice cream and cold drinks (the targets) for the next three days, *i.e.*, `H = 3`.
The measured temperature is a past-only covariate, and the promotion calendar is a known future covariate.
We start with ice cream alone and add the other variables one at a time.

## A Single Series

We begin with a single target, the daily sales of ice cream, observed for five days and forecast for the next three:

| day | ice_cream |
| --- | --------- |
| 1   | 40        |
| 2   | 55        |
| 3   | 80        |
| 4   | 75        |
| 5   | 50        |
| 6   | ?         |
| 7   | ?         |
| 8   | ?         |

A time series is a {py:class}`~sdm.tensor.TableTensor` in which each row is one time step, sorted in time, and each column is one variable (see the [tensor guide](tensor)):

```python
import sdm

history = sdm.TableTensor.from_columns(
    {"ice_cream": [40.0, 55.0, 80.0, 75.0, 50.0]},
    stypes={"ice_cream": "numerical"},
    device="cuda",
)
```

Forecasting models implement the {py:class}`~sdm.models.ICLModel` interface, where rows are interpreted as ordered time steps.
Here, we use {py:class}`~sdm.models.TimesFM3`, a time series foundation model that forecasts new series zero-shot, *i.e.*, without training on the series at hand:

```python
x_context = sdm.TableTensor(size=(5,), device="cuda")  # [5, 0]
y_context = history  # [5, 1]
x_query = sdm.TableTensor(size=(3,), device="cuda")  # [3, 0]

model = sdm.models.TimesFM3(device="cuda")
out = model(x_context, y_context, x_query)
```

The model takes three inputs, each a {py:class}`~sdm.tensor.TableTensor` or {py:class}`torch.Tensor`:

- `x_context`: the covariates over the observed history, with shape `[R_context, D]`.
- `y_context`: the past values of the targets, with shape `[R_context, Y]`.
- `x_query`: the known future covariates, with shape `[R_query, D_query]`. The number of rows in `x_query` is the forecast horizon, so no further argument is needed to request more or fewer future steps.

`x_context` and `x_query` carry the covariates.
We have none yet, so both are tables with zero columns, and only the number of rows of `x_query` matters.

The forecast is returned as a {py:class}`~sdm.tensor.TableTensor` with one row per future step.
Its columns depend on the model: {py:class}`~sdm.models.TimesFM3`, for example, outputs one column per target and quantile, from the 10th to the 90th percentile, named `"<target>__q<level>"`:

```python
out["ice_cream__q50"].numerical  # median forecast, [3, 1]
out["ice_cream__q10"].numerical  # lower end of an 80% prediction interval
out["ice_cream__q90"].numerical  # upper end of an 80% prediction interval
```

## Past-Only Covariates

We add the measured temperature, a past-only covariate, to the history.
It is unknown for the forecast horizon:

```python
history = sdm.TableTensor.from_columns(
    {
        "ice_cream": [40.0, 55.0, 80.0, 75.0, 50.0],
        "temperature": [21.0, 24.0, 28.0, 27.0, 22.0],
    },
    stypes={"ice_cream": "numerical", "temperature": "numerical"},
    device="cuda",
)
```

Past-only covariates need no declaration.
They are the columns of `x_context` that are absent from `x_query`, and are passed to the model as **context-only columns**:

```python
x_context = history.drop_columns("ice_cream")  # [5, 1]
y_context = history.select_columns("ice_cream")  # [5, 1]
x_query = sdm.TableTensor(size=(3,), device="cuda")  # [3, 0]

out = model(x_context, y_context, x_query)
```

Models declare support for context-only columns via {py:attr}`~sdm.models.ICLModel.supports_context_only_columns` (`True` for {py:class}`~sdm.models.TimesFM3`).

## Known Future Covariates

We add the promotion calendar, a known future covariate.
Unlike the temperature, it is known for the next three days:

| day | ice_cream | temperature | promotion |
| --- | --------- | ----------- | --------- |
| 1   | 40        | 21          | 0         |
| 2   | 55        | 24          | 0         |
| 3   | 80        | 28          | 1         |
| 4   | 75        | 27          | 1         |
| 5   | 50        | 22          | 0         |
| 6   | ?         | ?           | 1         |
| 7   | ?         | ?           | 0         |
| 8   | ?         | ?           | 0         |

The history gains a `promotion` column, and the future is a table that only holds the known future covariates:

```python
history = sdm.TableTensor.from_columns(
    {
        "ice_cream": [40.0, 55.0, 80.0, 75.0, 50.0],
        "temperature": [21.0, 24.0, 28.0, 27.0, 22.0],
        "promotion": [0.0, 0.0, 1.0, 1.0, 0.0],
    },
    stypes=dict.fromkeys(["ice_cream", "temperature", "promotion"], "numerical"),
    device="cuda",
)
future = sdm.TableTensor.from_columns(
    {"promotion": [1.0, 0.0, 0.0]},
    stypes={"promotion": "numerical"},
    device="cuda",
)
```

Known future covariates are the columns that appear in both `x_context` and `x_query`, in the same order:

```python
x_context = history.drop_columns("ice_cream")  # [5, 2]
y_context = history.select_columns("ice_cream")  # [5, 1]
x_query = future  # [3, 1]

out = model(x_context, y_context, x_query)
```

In summary, the columns that are passed determine which covariates the model sees:

- **Known future covariates** appear in both `x_context` and `x_query`.
- **Past-only covariates** appear in `x_context` only.
- **Without any covariates**, `x_context` and `x_query` have zero columns.

To compare several futures for the same history, *e.g.*, alternative promotion plans, call {py:meth}`~sdm.models.ICLModel.fit` once and {py:meth}`~sdm.models.ICLModel.predict` for each plan.
Since there is no query to infer the context-only columns from, they are given explicitly:

```python
model.fit(x=x_context, y=y_context, context_only_columns=["temperature"])
for plan in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]):
    x_query = sdm.TableTensor.from_columns(
        {"promotion": plan},
        stypes={"promotion": "numerical"},
        device="cuda",
    )
    out = model.predict(x_query)
model.clear()  # release the cached context
```

Passing `context_only_columns` to {py:meth}`~sdm.models.ICLModel.fit` raises an error for models that do not support them.

## Multiple Targets

So far, we forecast a single target.
Models that support multiple targets forecast them jointly, so they can exploit dependencies between them.
Each model declares this via {py:attr}`~sdm.models.ICLModel.supports_multi_target` (`True` for {py:class}`~sdm.models.TimesFM3`), while other models accept a single target column only.

We add the daily sales of cold drinks as a second target, which completes the example:

| day | ice_cream | cold_drinks | temperature | promotion |
| --- | --------- | ----------- | ----------- | --------- |
| 1   | 40        | 25          | 21          | 0         |
| 2   | 55        | 30          | 24          | 0         |
| 3   | 80        | 45          | 28          | 1         |
| 4   | 75        | 41          | 27          | 1         |
| 5   | 50        | 28          | 22          | 0         |
| 6   | ?         | ?           | ?           | 1         |
| 7   | ?         | ?           | ?           | 0         |
| 8   | ?         | ?           | ?           | 0         |

All targets are passed as columns of `y_context`:

```python
history = sdm.TableTensor.from_columns(
    {
        "ice_cream": [40.0, 55.0, 80.0, 75.0, 50.0],
        "cold_drinks": [25.0, 30.0, 45.0, 41.0, 28.0],
        "temperature": [21.0, 24.0, 28.0, 27.0, 22.0],
        "promotion": [0.0, 0.0, 1.0, 1.0, 0.0],
    },
    stypes=dict.fromkeys(["ice_cream", "cold_drinks", "temperature", "promotion"], "numerical"),
    device="cuda",
)

x_context = history.drop_columns(["ice_cream", "cold_drinks"])  # [5, 2]
y_context = history.select_columns(["ice_cream", "cold_drinks"])  # [5, 2]
x_query = future  # [3, 1]

out = model(x_context, y_context, x_query)
```

The forecast contains one column per target and quantile:

```python
out["ice_cream__q50"].numerical  # median forecast, [3, 1]
out["cold_drinks__q10"].numerical  # lower end of an 80% prediction interval
out["cold_drinks__q90"].numerical  # upper end of an 80% prediction interval
```

## Ensembling

A forecast can be made by several ensemble members, whose forecasts are averaged.
The `num_estimators` argument sets the number of members, and omitting it is the same as `num_estimators=1`.
More members cost proportionally more.
Members may see differently transformed views of the series, so their forecasts can differ (see the [ensembling section](icl) of the ICL guide):

```python
out = model(x_context, y_context, x_query, num_estimators=4)
```

## Batches of Series

Forecasting often involves many series, *e.g.*, the sales of every store.
To forecast a batch of `B` series in one call, stack them along a leading batch dimension, so that `x_context`, `y_context` and `x_query` have shapes `[B, R_context, D]`, `[B, R_context, Y]` and `[B, R_query, D_query]`.
All series in a batch share the same horizon and number of context rows, and shorter series are left-padded with `NaN`.

Pass `num_estimators` so that the leading dimension is read as a batch of independent series, each forecast by `num_estimators` ensemble members.

```{note}
Without `num_estimators`, the leading dimension is read as the ensemble members of a single series, and their forecasts are combined into one.
A batch of series is then averaged into a single forecast, and the batch dimension is lost.
```

For example, to forecast the next 24 steps of a batch of `B` series without covariates:

```python
# values: [B, R_context] tensor of observations, left-padded with NaN
targets = sdm.TableTensor.from_tensor(values.unsqueeze(-1), columns=["target"])  # [B, R_context, 1]

out = model(
    x_context=targets.drop_columns("target"),  # [B, R_context, 0]
    y_context=targets,  # [B, R_context, 1]
    x_query=sdm.TableTensor(size=(B, 24), device="cuda"),  # [B, 24, 0]
    num_estimators=1,
)
```

The forecasts keep the batch dimension:

```python
out["target__q50"].numerical  # median forecast, [B, 24, 1]
out["target__q10"].numerical  # lower end of an 80% prediction interval, [B, 24, 1]
out["target__q90"].numerical  # upper end of an 80% prediction interval, [B, 24, 1]
```

See [`examples/timeseries/quickstart.py`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/timeseries/quickstart.py) for runnable examples on real datasets, which cover batches of series as well as known future and past-only covariates.
