# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples
- [**`timeseries/`**](timeseries/): Time series foundation model examples

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `KumoTabular`.

[`timeseries/forecast_missingness.py`](timeseries/forecast_missingness.py) shows `KumoForecastMissingness` with incomplete histories and named historical covariates. It uses the separate 96-step missingness-aware checkpoint and returns point forecasts in the original target units.
