# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples
- [**`timeseries/`**](timeseries/): Time series foundation model examples

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.

See [`tabular/context_sharding.py`](tabular/context_sharding.py) for experimental multi-GPU KumoTabular inference with optional FP16 execution. The example includes launch instructions and requires pretrained weights and a finite external job timeout.
