# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples

Inference-serving benchmark drivers (the dataset benchmark suites live under [`benchmark/`](../benchmark/)):

- [**`benchmark_tabiclv2.py`**](benchmark_tabiclv2.py): inference acceleration benchmark for `sdm.models.TabICLv2`, sweeping NVIDIA-recommended serving recipes. Its module docstring doubles as the serving notes (bucketed shapes, regional compilation and its raised `recompile_limit`, compile-cache artifacts).

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.
