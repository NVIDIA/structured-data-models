# PR #421 ensemble-aware processing revalidation

## Outcome

**Parity: PASS.** The latest implementation matches pinned TabICLv2 v2.0.0
at every requested semantic boundary for classification and regression, in
both vectorized and sequential Recipe execution. Current-API KumoRFM
vectorized/sequential and cached execution also match.

- Latest `main`: `0d3a0b8aaec16d6ff503119a4ea3ebc64c46e046`
- Final PR #516 commit: `6727c1026be72ad9b8deeaa4b28817f3472d1ffa`
- TabICLv2 reference: `f719c886a586ed4a29236345e319ac1ea596c478`
  (`tabicl==2.0.0`)
- Checkpoints: classifier SHA-256
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`;
  regressor SHA-256
  `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`
- Updated PR: https://github.com/NVIDIA/structured-data-models/pull/421

The final change set keeps PR #516's redesigned `EnsembleTable` and Recipe
execution architecture. It does not restore the historical `TargetDecode`,
old Processor tree, canonical RFM fixture, or duplicated implementations from
the original PR #421.

## Parity

The parity fixture has eight context rows, two query rows, four retained
skewed float32 features, three classes with a deliberately non-canonical SDM
vocabulary, and a skewed regression target. Both references use eight
estimators, seed 42, `none`/`power` normalization, Latin feature shuffling,
and shifted class shuffling.

Discrete member configurations, feature permutations, class permutations,
column mappings, and final class columns are compared exactly. Floating-point
tolerances account only for SDM float32 versus the reference's sklearn
float64 preprocessing:

- identity-normalized features: `atol=2e-7`, `rtol=2e-7`;
- power-normalized features: `atol=2e-4`, `rtol=4e-4`;
- regression target preprocessing: `atol=rtol=2e-6`;
- per-estimator classification outputs: `atol=3e-4`, `rtol=4e-4`;
- per-estimator regression outputs: `atol=2e-4`, `rtol=8e-4`;
- final public classification/regression outputs: `atol=3e-5` /
  `1e-4`.

The strict suite compares feature and target preprocessing, every materialized
member, model inputs, raw per-estimator checkpoint outputs, canonical output
mapping, estimator reduction, final processing, direct public forward, and
cached fit/predict. The result is **6 passed**. The complete affected suite is
**101 passed, 2 expected compile skips**.

The first divergences found during revalidation were:

1. member 0 was the only matching classification member, and only one of
   eight regression members matched, because the latest generic shufflers did
   not reproduce TabICLv2's coupled Latin feature/class plan;
2. sequential execution restarted member-local choices at zero, because
   one-member Recipe binds did not know the global estimator id;
3. final classification probabilities had equal values in a cyclically
   shifted class order, because reduction used the first shuffled member's
   output schema.

All three are fixed by a private TabICLv2 ensemble plan, global member ids
through the existing bind path, callback hooks on the current shufflers, and
one final canonical class reorder.

## Performance method

The main workload is the original 40,000-context + 10,000-query dataset with
100 features (90 numerical, 10 categorical), vocabulary 4,096, missing values,
a constant feature, outliers, seed 7 for data, seed 42 for the eight-member
ensemble, float32, five warmups, and 20 repetitions. Times are synchronized
wall-clock median / nearest-rank p95. Inputs are device-resident. Peak GPU
memory is incremental allocated memory above the pre-call baseline; CPU memory
is sampled RSS.

Hardware is an NVIDIA L4 with 23,659,151,360 bytes device memory and an AMD
EPYC 7R13 host. Software is PyTorch 2.13.0+cu130 and CUDA 13.0.

Cells below are classification / regression median / p95 milliseconds.
The historical and speed-of-light runs did not isolate reduction and final
processing from their output aggregate.

| Stage                          | Original E2E | Latest implementation | TabICLv2 | Speed-of-light | Difference |
| ------------------------------ | -----------: | --------------------: | -------: | -------------: | ---------: |
| Feature + target preprocessing | 93.90 / 94.70; 86.80 / 90.60 | 83.95 / 96.85; 73.47 / 77.18 | PASS | 79.75 / 82.24; 73.07 / 73.83 | −10.6% / −15.4% vs old median; +5.3% / +0.5% vs lower bound |
| Canonical output mapping       | included in 0.66 / 0.76; 9.94 / 10.00 output | 1.58 / 1.68; 9.99 / 10.12 | PASS | 0.070 / 0.083; 2.17 / 2.27 | +1.51 / +7.82 ms vs lower bound |
| Estimator reduction            | not isolated | 0.182 / 0.223; 1.61 / 1.62 | PASS | included in output | no direct historical split |
| Final output processing        | not isolated | 0.722 / 0.911; 1.07 / 1.51 | PASS | output aggregate 0.232 / 0.259; 0.496 / 0.594 | latest output aggregate is +1.70 / +1.65 ms vs old |
| Total processing overhead      | 128.60 / 129.80; 123.90 / 126.00 | 108.32 / 112.13; 108.85 / 148.83 | PASS | 78.95 / 79.49; 78.04 / 80.60 | −15.8% / −12.1% vs old median; +37.2% / +39.5% vs lower bound |

The speed-of-light Power Recipe is a practical lower-bound experiment, not a
drop-in TabICLv2 implementation. Its analytic compiled Power fitter changes
the fitting algorithm, so matching its number is not a parity-preserving
localized change.

## Original versus latest 50k Recipe

| Task / device | Original median / p95 | Latest median / p95 | Latest peak delta | Median change |
| ------------- | --------------------: | ------------------: | ----------------: | ------------: |
| Classification / CPU | 2265.4 / 2304.7 ms | 2244.96 / 2344.61 ms | 196.3 MiB RSS | −0.9% |
| Regression / CPU | 2618.8 / 3057.2 ms | 2596.57 / 2747.77 ms | 1035.2 MiB RSS | −0.8% |
| Classification / L4 | 128.6 / 129.8 ms | 108.32 / 112.13 ms | 292.7 MiB | −15.8% |
| Regression / L4 | 123.9 / 126.0 ms | 108.85 / 148.83 ms | 1035.9 MiB | −12.1% |

The current public zero-core processing boundary measures 106.55 / 108.82 ms
for vectorized classification and 111.13 / 192.18 ms for vectorized
regression. Sequential Recipe execution measures 361.96 / 414.21 ms and
394.08 / 473.15 ms while lowering peak allocation to 146.7 MiB and 668.1 MiB,
respectively. Sequential execution now intentionally fits/transforms one
member at a time; it is a low-memory mode, not the historical shared-
preprocessing model schedule.

Regression's p95 tail is orchestration rather than a slow Processor. Its
independently measured bind, query, mapping, reduction, and final stages have
p95s of 77.18, 25.02, 10.12, 1.62, and 1.51 ms. A separate 40-sample
diagnostic reproduced one 182.6 ms outlier exactly when Python generation-2
garbage collection ran over repeated deep-copied Recipe/module graphs; median
was 107.6 ms and p95 122.6 ms in that diagnostic. Disabling automatic GC
reduced diagnostic p95 to 119.4 ms but is not an appropriate library-side
semantic change.

Regression Recipe peak allocation is 1035.9 MiB versus 914.9 MiB originally
(+121.0 MiB, +13.2%). This is the remaining material processing-memory
regression: current grouped output mapping keeps packed raw and decoded
representations alive through reduction. The real-model result below remains
far below the historical parallel peak.

## Processor timings

| Processor operation | CPU median / p95 | L4 median / p95 | L4 speed-of-light |
| ------------------- | ----------------: | ---------------: | ----------------: |
| `ImputeMean.fit_transform` | 8.17 / 9.47 ms | 0.325 / 0.379 ms | current-scale |
| `DropConstantColumns.fit_transform` | 16.43 / 17.91 ms | 0.635 / 0.805 ms | current-scale |
| `Standardize.fit_transform` | 12.31 / 12.46 ms | 0.303 / 0.344 ms | 0.165 / 0.193 ms affine candidate |
| `PowerTransform.fit` | 1602.22 / 1625.69 ms | 32.88 / 33.59 ms | 3.381 / 3.416 ms compiled analytic Newton |
| `PowerTransform.fit_transform` | 1633.67 / 1664.94 ms | 33.92 / 34.49 ms | 3.411 / 3.497 ms compiled analytic Newton |
| `PowerTransform.transform` | 7.06 / 8.52 ms | 0.411 / 0.440 ms | 0.376 / 0.394 ms |
| `PowerTransform.inverse_transform` | 139.93 / 143.81 ms | 1.864 / 1.876 ms | 0.278 / 0.308 ms compiled |
| `ClipSigma.fit_transform` | 29.69 / 31.87 ms | 0.987 / 1.137 ms | 1.254 / 1.281 ms |
| `ShuffleColumns.fit_transform` | 3.72 / 4.14 ms | 1.485 / 1.732 ms | 0.074 / 0.101 ms direct index-select |
| `AlignCategories.fit_transform` | 3.60 / 3.89 ms | 5.65 / 7.41 ms | 0.362 / 0.394 ms dense lookup |
| `AlignCategories.transform` | 3.10 / 3.33 ms | 5.15 / 6.57 ms | 0.194 / 0.226 ms direct lookup |

No timed numerical Processor fell back to CPU or transferred data between
host and device. `PowerTransform.fit` remains the dominant latency. The
~1.1 ms `ShuffleColumns` regression versus the old 0.41 ms comes from the
current generic grouped-schema/materialization orchestration; it is too small
to justify a second specialized shuffle implementation. Dense category
lookups remain a medium-risk future optimization because they need explicit
dtype, unseen-category, and bounded-domain validation.

## Large representative real-model run

The largest real-checkpoint workload that was rerun uses 2,400 context + 600
query rows, 100 features, eight estimators, float32, five warmups, and 20
repetitions on the L4.

| Task / execution | Original median / p95 | Latest median / p95 | Original / latest peak | Median change |
| ---------------- | --------------------: | ------------------: | ---------------------: | ------------: |
| Classification / vectorized | 2456.50 / 2466.29 ms | 2328.45 / 2345.48 ms | 13221.4 / 1663.4 MiB | −5.2% |
| Classification / sequential | 2478.10 / 2494.18 ms | 2534.06 / 2578.31 ms | 1661.2 / 1655.8 MiB | +2.3% |
| Regression / vectorized | 2517.52 / 2551.87 ms | 2426.94 / 2485.48 ms | 13221.5 / 1679.5 MiB | −3.6% |
| Regression / sequential | 2482.94 / 2499.95 ms | 2590.17 / 2609.79 ms | 1676.4 / 1671.6 MiB | +4.3% |

The default vectorized path is faster than the original and uses about 87%
less peak GPU allocation because PR #516 keeps Recipe preprocessing
vectorized while running the model once per estimator. Sequential's 2-4%
runtime regression is explained by repeated one-member fitting and
transformation; its low-memory behavior is preserved.

## Smallest fixes and remaining gap

The only performance fix added after parity was restored is a CUDA-only packed
regression-output path. The first latest-code measurement was 16.71 ms for
output transformation and 128.89 ms for the public vectorized boundary.
Wrapping one already-stacked `EnsembleTable` group avoids the second
materialization and reuses the current inverse/output processors. Remeasured
values are 11.59 ms and 111.13 ms. The same strategy was 6.9% slower on CPU,
so CPU deliberately retains member-wise inversion.

The remaining speed-of-light gap is 29.37 ms classification and 30.81 ms
regression. Almost all of that is the 32.9 ms Power fit versus the 3.4 ms
compiled analytic-Newton prototype. Adopting it would change the fitting
algorithm and requires model-quality validation beyond execution parity;
therefore it is not included as a minimal PR #421 fix. The categorical and
shuffle lower bounds offer smaller absolute gains and would add broader
generic-Processor risk.

## Limitations

- The speed-of-light rows were measured on an earlier main-state commit and
  are lower bounds, not pinned-reference timings.
- The 50k Recipe p95 includes Python orchestration/GC by the original
  methodology. Real-model p95 remains within about 1-3% of its median.
- KumoRFM functional behavior was rerun on CPU and CUDA using the current
  relational fixture. CUDA string sorts and joins emitted the expected CPU
  fallback warning because cuDF is not installed. The historical canonical
  RFM benchmark fixture/API no longer exists on current main, so its old
  non-equivalent performance harness was not restored.
- CPU peak RSS for short individual Processor calls is sampler-limited; total
  Recipe/public-boundary RSS is reported.

## Artifacts and reproduction

- `results/pr421_tabiclv2_cpu_processing.json`
- `results/pr421_tabiclv2_gpu_processing.json`
- `results/pr421_tabiclv2_gpu_model_3k.json`

```bash
env SDM_RUN_TABICLV2_GPU_PARITY=1 uv run pytest -q \
  test/integration/tabiclv2_parity/test_strict_ensemble.py

uv run python benchmark/tabiclv2_ensemble_revalidation.py \
  --devices cuda --executions vectorized sequential --include-processors \
  --output benchmark/results/pr421_tabiclv2_gpu_processing.json

uv run python benchmark/tabiclv2_ensemble_revalidation.py \
  --devices cuda --executions vectorized sequential \
  --context-rows 2400 --query-rows 600 --only-model \
  --output benchmark/results/pr421_tabiclv2_gpu_model_3k.json
```
