# Kumo preprocessing benchmark

This runs the actual `KumoTabular.default_recipe()` through `RecipeExecution`. It does not construct a model or time inference. `prediction_parity.py` separately validates real Kumo-Small predictions with non-constant, reproducible random weights; it does not require downloaded checkpoints.

## Configuration

- The benchmark adapter `SDMKumoTabularModel` defaults to 16 estimators. The direct model API instead defaults to one estimator for a two-dimensional input. The matrix uses the adapter's actual default, not the direct API's implicit default.
- Numerical preprocessing is Cast(float64), DropConstantColumns, Standardize(eps=1e-6), Clip(-100, 100), round-robin Choice(Identity, PowerTransform, RobustScale + saturating Clip, RankGaussian(max_knots=8192)), logarithmic ClipSigma, and FlipSign.
- Categorical preprocessing is AlignCategories(sort_by="value"), AddCategoryCounts(min_cardinality=50), ToNumerical, and the same numerical pipeline without FlipSign. The combined feature pipeline then shuffles columns with the Latin method, selects at most 500 per stype, and casts numerical values to float32.
- Text is passthrough in this recipe. The requested 1% text columns are present and validated, but no text embedding or TF-IDF work is invented for the benchmark. The Kumo model does not consume passthrough text.
- Classification targets contain five classes and use AlignCategories plus shifted ShuffleCategories. Output reduction and inference are outside the latency scope.
- Every dataset has exactly 89% numerical, 10% categorical and 1% text columns. Fit and query have the specified row count. Numerical fixtures include outliers, skew, missing values and a constant column; categorical dictionaries have 12/64 entries, missing codes, and reversed query ordering.

## Reproduce

Use the same environment, thread count and otherwise idle GPU for both versions. The harness refuses normal latency measurements when another process occupies the selected GPU. For a remapped `CUDA_VISIBLE_DEVICES`, pass a device whose index also matches `nvidia-smi`'s physical index.

```bash
uv sync --locked --no-default-groups --group test
task_repo=$(pwd)
task_results=/tmp/sdm-797-results
git worktree add --detach /tmp/sdm-797-baseline 12804f2e187b4e78321eabb89294a1b45c3de865
(
    cd /tmp/sdm-797-baseline
    PYTHONPATH=. "$task_repo/.venv/bin/python" \
        "$task_repo/benchmark/ensemble/kumo_preprocessing.py" \
        --suite --phase baseline --artifacts "$task_results"
)
PYTHONPATH=. .venv/bin/python -m benchmark.ensemble.kumo_preprocessing \
    --suite --phase after --artifacts "$task_results"
PYTHONPATH=. .venv/bin/python -m benchmark.ensemble.report \
    --artifacts "$task_results"
```

The baseline commit includes main `3d24f3e299d8412d7156cb07883de0d859bca1af`, PR #1344 at `76909cd2e44f68dc13d303e4d6bf1642aed0e30c`, and PR #1361 at `bfef597007f8c0aab43c539c8ccca697cadf0931`. The new harness lives in the refactored checkout but can execute against the unchanged baseline source with `PYTHONPATH` as above. It refuses to overwrite existing baseline measurements.

Defaults are two warmups, seven repetitions and four CPU threads. The six matrix cases run at E16; 10,000 × 500 also runs at E1/E4/E8. Each case runs in a fresh process. Only 1,000 × 100 and 10,000 × 500 at E16 receive full CPU/CUDA profiles. Individual cases accept `--rows`, `--columns`, `--members`, `--profile` and `--device`.

To collect just those two selective profiles, use `--suite --profile-only` with the appropriate phase and checkout. These traces are separate from latency samples. CUDA profiling must precede any CPU-only profiler in a process: this PyTorch build otherwise silently omits CUDA kernels. The harness enforces that order and rejects a requested CUDA profile with no device activity.

Collect per-processor memory separately with `--suite --memory-only`, once against each checkout with `--phase baseline|after`. This writes `memory/<phase>/` and rechecks preprocessing parity. One measured pipeline follows two warmups; it produces peaks, not latency samples. GPU allocator peaks are reset at callback boundaries, preserving each enclosing callback's high-water mark before a nested reset. One RSS sampler tracks active callbacks every 2 ms, supplemented by boundary samples. Peaks include live inputs, fitted state and enclosing work, and are not additive. `processors.csv` includes both absolute peaks and increases relative to each callback's entry. Unlike the repeated latency harness, this pass does not retain previous returned tables.

For untimed correctness and operation/allocation diagnostics when the GPU is shared:

```bash
PYTHONPATH=. .venv/bin/python -m benchmark.ensemble.kumo_preprocessing \
    --suite --phase after --parity-only --artifacts "$task_results"
PYTHONPATH=. .venv/bin/python -m benchmark.ensemble.kumo_preprocessing \
    --suite --phase after --counters-only --artifacts "$task_results"
```

These modes compare saved baseline values, schemas, categories and text, but deliberately do not produce latency claims. `--counters-only` profiles CPU-side ATen dispatch/allocation events, not expensive CUDA traces. `prediction_parity.py` accepts `--phase baseline|after --artifacts ... --device cpu|cuda`; run the same script with the corresponding source checkout on `PYTHONPATH`.

## Interpretation

- Total preprocessing is synchronized wall time for recipe fit-transform plus query transform. Each repetition's two times are added before calculating median/p95, so the total median need not equal the sum of separate medians. Feature fit without requesting its returned transform is measured separately. Individual `fit` and `transform` callbacks invoked by the real pipeline are recorded alongside fused callbacks; a lifecycle not invoked by the configured recipe is not assigned an invented timing.
- Per-processor CPU and CUDA-event timings come from separate instrumented runs. Median/p95 are per callback, not per complete pipeline. See callback counts and exclusive CPU time per pipeline in `processors.csv`; never sum nested inclusive times. Seven repetitions make p95 descriptive, not a statistically precise tail estimate.
- Per-processor `*_delta_percent` columns use positive values for regressions and negative values for improvements. The total `speedup_percent` column uses positive values for speedups. Per-callback deltas must be read alongside callback counts when physical batching changes.
- Process RSS is sampled every 2 ms, not an exact CPU allocator high-water mark. GPU peaks use PyTorch allocated memory and include live inputs/outputs. The repeated-call harness retains the previous returned tables until their replacements are assigned, identically before and after; these are repeated-workflow peaks, not isolated-call working-set minima. Temporary cumulative allocated bytes from profiler events are not peak live memory.
- `operations.csv` lists every processor/configuration/lifecycle/ATen operation, including copying, indexing, views and materialization. ATen dispatch counts include decomposition and views; they are not unique CUDA kernel counts. Positive self-memory events are allocation-producing events, not raw malloc calls.
- The baseline's ordinary processor-body execution counts and the refactor's ensemble-aware body counts describe different boundaries. They must not be compared as physical execution counts. Compare ATen operations and selective CUDA traces instead. Reused input storage alone does not prove redundant arithmetic was removed.
- Raw fixtures, parity references, JSON measurements, traces and source/fixture hashes remain outside the repository. `report.py` exports every processor to CSV, preserves missing measurements as empty fields and never fills in speedups for missing runs.

See [PERFORMANCE.md](PERFORMANCE.md) for this run's status and remaining measurement limitations.
