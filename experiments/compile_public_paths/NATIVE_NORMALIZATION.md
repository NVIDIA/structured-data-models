# Same-dtype numerical diagnostic

This branch combines the dynamic-neighborhood fixes with two isolated numerical experiments. It is not a recommendation to enable them by default.

1. Rank-three compiled `Linear` uses functional `bmm` when the eager buffered operation uses `bmm`, instead of substituting `matmul`. This preserves the eager operation choice.
2. `native_layer_norm_patch.py` routes compiled inference LayerNorm through a custom operation that calls the original native kernel. Eager execution and gradient-enabled execution retain their original method. Importing this experiment module activates the override only in that process.

The second operation prevents Inductor from replacing the normalization arithmetic. It may also prevent useful fusion, so numerical agreement alone does not establish a speed or memory benefit. FP32 inputs, weights, and comparison tolerances stay unchanged.

Reproduce the ten real neighborhoods with:

```sh
PYTHONPATH=.:experiments/compile_public_paths OMP_NUM_THREADS=1 \
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python -c 'import native_layer_norm_patch, runpy; runpy.run_path("experiments/compile_public_paths/probe.py", run_name="__main__")' \
  --model relational --entry predict --arm-index 1 --fullgraph \
  --capture-dynamic-outputs \
  --relational-query-indices 3 4 5 6 7 8 9 10 11 3 \
  --include-predictions --data /path/to/driver-dnf_bundle.pt \
  --checkpoint /path/to/Kumo-Relational/classifier.pt \
  --output /tmp/native-normalization.json
```

The public integration branch does not include either numerical change. Its default arithmetic results remain the comparison baseline in `dynamic-neighborhoods.json`.

## Result

Actual CPU Inductor on PyTorch 2.14, identical ten RelBench neighborhoods, FP32 inputs/weights, `atol=1e-5, rtol=1e-4`:

| Arithmetic | Strict comparison | Maximum absolute difference | Captured full graphs |
|---|---:|---:|---:|
| Default compiled operations | 7/10 query batches pass | `6.25e-5` | 2 |
| Functional `bmm` + native LayerNorm diagnostic | 10/10 pass | `2.38e-7` | 2 |

All 160 predicted classes agree in both cases. The diagnostic source is `d8e17e00e`; results are in [native-normalization-results.json](native-normalization-results.json). The first query captures one graph, the second captures another, and the remaining queries reuse them. Each run uses an isolated cache directory with cache reuse disabled. No compilation limit or tolerance was increased.

This identifies a way to preserve close numerical agreement while compiling the full public prediction path. It does not establish GPU performance, memory usage, broader dataset quality, or the tradeoff from preventing normalization fusion. The public integration branch remains unchanged.
