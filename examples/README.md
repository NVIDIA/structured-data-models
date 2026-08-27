# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabiclv2/`**](tabiclv2/): `sdm.models.TabICLv2`
- [**`kumo/relational/`**](kumo/relational/): `sdm.models.KumoRelational`

Inference-serving benchmark drivers (the dataset benchmark suites live under [`benchmark/`](../benchmark/)):

- [**`benchmark_tabiclv2.py`**](benchmark_tabiclv2.py): inference acceleration benchmark for `sdm.models.TabICLv2`, sweeping NVIDIA-recommended serving recipes. Its module docstring doubles as the serving notes (bucketed shapes, regional compilation and its raised `recompile_limit`, compile-cache artifacts).

See [`tabiclv2/quickstart.py`](tabiclv2/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.
