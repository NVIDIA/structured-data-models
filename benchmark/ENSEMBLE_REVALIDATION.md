# PR #421 ensemble-aware processing revalidation

## Outcome

**Parity: PASS.** The latest implementation matches pinned TabICLv2 v2.0.0
at every requested semantic boundary for classification and regression, in
both vectorized and sequential Recipe execution. Current-API KumoRFM
vectorized/sequential and cached execution also match.

- Latest `main`: `d2ade8961e941536d8c0ede2c044f276295eb4fa`
- Final PR #516 commit: `8b3eaca8eb0bccecfdd9379bf7279437779a83e8`
- Pre-EnsembleProcessor baseline: `82dd59560796533d4e8142f4c770eda366af398a`
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
cached fit/predict. The result is **6 passed**; final targeted coverage is
**117 passed, 4 expected skips**. The complete repository pre-push suite is **584 passed, 105 skipped**.

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

The latest implementation artifacts were captured at source head `1b742180`,
immediately before latest main `d2ade896` was merged. That three-commit main
delta changes only documentation/example layout, git-blame metadata,
full-column no-op slice compatibility, and composite `pin_memory` dispatch.
The benchmark invokes none of those branches; strict parity, the affected
suite, and the full repository gate were rerun after the merge. The exact
pre-EnsembleProcessor artifacts were captured from detached commit `82dd5956`,
the parent of `123807c7` (`Introduce EnsembleProcessor base class (#497)`), in
the same PyTorch/CUDA environment and on the same host and L4.

Cells below are classification / regression median / p95 milliseconds.
The historical and speed-of-light runs did not isolate reduction and final
processing from their output aggregate.

Here, **Original E2E** means the already optimized ensemble-aware
implementation originally benchmarked in PR #421. It does **not** mean the
older implementation that deep-copied the Recipe eight times. That exact
pre-EnsembleProcessor baseline is reported separately below.

| Stage                          |                                 Original E2E |            Latest implementation | TabICLv2 |                                Speed-of-light |                                                   Difference |
| ------------------------------ | -------------------------------------------: | -------------------------------: | -------: | --------------------------------------------: | -----------------------------------------------------------: |
| Feature + target preprocessing |                 93.90 / 94.70; 86.80 / 90.60 |     84.36 / 85.42; 75.29 / 78.21 |     PASS |                  79.75 / 82.24; 73.07 / 73.83 |  −10.2% / −13.3% vs old median; +5.8% / +3.0% vs lower bound |
| Canonical output mapping       | included in 0.66 / 0.76; 9.94 / 10.00 output |       1.60 / 1.80; 10.00 / 10.13 |     PASS |                    0.070 / 0.083; 2.17 / 2.27 |                              +1.53 / +7.83 ms vs lower bound |
| Estimator reduction            |                                 not isolated |       0.187 / 0.207; 1.60 / 1.63 |     PASS |                            included in output |                                   no direct historical split |
| Final output processing        |                                 not isolated |       0.720 / 0.844; 1.05 / 1.22 |     PASS | output aggregate 0.232 / 0.259; 0.496 / 0.594 |           latest output aggregate is +1.78 / +1.63 ms vs old |
| Total processing overhead      |             128.60 / 129.80; 123.90 / 126.00 | 108.82 / 109.89; 112.26 / 123.19 |     PASS |                  78.95 / 79.49; 78.04 / 80.60 | −15.4% / −9.4% vs old median; +37.8% / +43.8% vs lower bound |

The speed-of-light Power Recipe is a practical lower-bound experiment, not a
drop-in TabICLv2 implementation. Its analytic compiled Power fitter changes
the fitting algorithm, so matching its number is not a parity-preserving
localized change.

## Before EnsembleProcessor: exact eight-copy baseline

Commit `82dd5956` is the last main commit before `EnsembleProcessor` existed.
Its public `ICLModel.forward` constructs
`[copy.deepcopy(recipe) for _ in range(num_estimators)]` and then executes all
feature fit/transform, target fit/transform, model-boundary, inverse mapping,
and output work estimator by estimator in a Python loop. The table below
compares that exact code with the latest public vectorized zero-core boundary;
the zero core deliberately removes model-forward time from both sides.

| Task / device        | Pre-EnsembleProcessor median / p95 | Latest vectorized median / p95 | Median speedup | Old / latest Power members | Old / latest peak delta |
| -------------------- | ---------------------------------: | -----------------------------: | -------------: | -------------------------: | ----------------------: |
| Classification / CPU |               4935.28 / 5034.29 ms |           2225.29 / 2267.11 ms |          2.22× |                      2 / 4 |       121.2 / 271.8 MiB |
| Regression / CPU     |               3944.45 / 4021.63 ms |           2693.64 / 2791.63 ms |          1.46× |                      1 / 4 |      654.3 / 1034.9 MiB |
| Classification / L4  |                 341.05 / 387.21 ms |             105.50 / 111.08 ms |          3.23× |                      4 / 4 |       149.8 / 291.8 MiB |
| Regression / L4      |                 394.39 / 469.74 ms |             112.62 / 194.75 ms |          3.50× |                      5 / 4 |      681.3 / 1067.1 MiB |

The L4 result is the meaningful historical architecture comparison: the old
seed-42 device RNG selected four Power members for classification and five for
regression, while the latest parity-aware plan deterministically selects four
for both. The current path calls the adapted `PowerTransform` once for the
selected member group instead of fitting four or five copied processors. It
is 3.23-3.50× faster, close to the expected 4× but not exactly 4× because the
remaining processors still handle member-sized tensors, and canonical
mapping, reduction, output processing, grouping, and materialization remain.
The extra fifth historical regression Power fit also makes that row a slightly
easier speedup target.

The raw CPU ratio is not a controlled four-Power comparison. The historical
CPU RNG happened to select only two Power members for classification and one
for regression at the required seed, whereas the latest TabICLv2-parity plan
uses four. For a semantically controlled 4/4 comparison, latest sequential
Recipe execution measures 8739.04 / 10456.31 ms and 9064.93 / 9172.80 ms,
which makes vectorized execution 3.93× and 3.37× faster. The exact historical
L4 results also validate that this current sequential proxy is close to the
old eight-copy architecture: it differs by +5.0% for classification and
−3.6% for regression at the median.

The higher latest peak allocation is the intentional batching tradeoff: the
old loop retained one estimator's processing intermediates at a time, while
the vectorized path packs compatible members to remove repeated calls and
launches. Production model execution remains estimator-sequential after PR
#516, so it does not recreate the old eight-model activation peak.

## Original versus latest 50k Recipe

| Task / device        | Original median / p95 |  Latest median / p95 | Latest peak delta | Median change |
| -------------------- | --------------------: | -------------------: | ----------------: | ------------: |
| Classification / CPU |    2265.4 / 2304.7 ms | 2224.91 / 2320.28 ms |     211.4 MiB RSS |         −1.8% |
| Regression / CPU     |    2618.8 / 3057.2 ms | 2571.84 / 2614.88 ms |    1124.8 MiB RSS |         −1.8% |
| Classification / L4  |      128.6 / 129.8 ms |   108.82 / 109.89 ms |         292.7 MiB |        −15.4% |
| Regression / L4      |      123.9 / 126.0 ms |   112.26 / 123.19 ms |        1035.9 MiB |         −9.4% |

The current public zero-core processing boundary measures 105.50 / 111.08 ms
for vectorized classification and 112.62 / 194.75 ms for vectorized
regression. Current sequential Recipe execution measures 358.03 / 370.36 ms
and 380.13 / 463.93 ms while lowering peak allocation to 146.1 MiB and
745.2 MiB, respectively. It is the semantically controlled 4/4 low-memory
comparison; the exact pre-EnsembleProcessor measurements are reported above.

Regression's p95 tail is orchestration rather than a slow Processor. Its
independently measured bind, query, mapping, reduction, and final stages have
p95s of 78.21, 23.11, 10.13, 1.63, and 1.22 ms. A separate 40-sample
diagnostic reproduced one 182.6 ms outlier exactly when Python generation-2
garbage collection ran over repeated deep-copied Recipe/module graphs; median
was 107.6 ms and p95 122.6 ms in that diagnostic. Disabling automatic GC
reduced diagnostic p95 to 119.4 ms but is not an appropriate library-side
semantic change.

Regression Recipe peak allocation is 1035.9 MiB versus 914.9 MiB originally
(+121.0 MiB, +13.2%). This is the remaining material processing-memory
regression: current grouped output mapping keeps packed raw and decoded
representations alive through reduction. The production-model result remains
far below the historical parallel peak.

## Processor timings

| Processor operation                 |     CPU median / p95 |  L4 median / p95 |                         L4 speed-of-light |
| ----------------------------------- | -------------------: | ---------------: | ----------------------------------------: |
| `ImputeMean.fit_transform`          |       8.11 / 9.95 ms | 0.320 / 0.357 ms |                             current-scale |
| `DropConstantColumns.fit_transform` |     15.82 / 16.29 ms | 0.637 / 0.681 ms |                             current-scale |
| `Standardize.fit_transform`         |     12.21 / 12.38 ms | 0.292 / 0.312 ms |         0.165 / 0.193 ms affine candidate |
| `PowerTransform.fit`                | 1566.84 / 1619.00 ms | 32.93 / 33.68 ms | 3.381 / 3.416 ms compiled analytic Newton |
| `PowerTransform.fit_transform`      | 1590.12 / 1612.12 ms | 33.95 / 34.76 ms | 3.411 / 3.497 ms compiled analytic Newton |
| `PowerTransform.transform`          |       7.15 / 7.69 ms | 0.415 / 0.517 ms |                          0.376 / 0.394 ms |
| `PowerTransform.inverse_transform`  |   144.95 / 159.52 ms | 1.895 / 1.913 ms |                 0.278 / 0.308 ms compiled |
| `ClipSigma.fit_transform`           |     31.74 / 40.41 ms | 1.010 / 1.079 ms |                          1.254 / 1.281 ms |
| `ShuffleColumns.fit_transform`      |       3.65 / 3.83 ms | 1.432 / 1.556 ms |      0.074 / 0.101 ms direct index-select |
| `AlignCategories.fit_transform`     |       3.74 / 4.20 ms |   3.90 / 4.13 ms |             0.362 / 0.394 ms dense lookup |
| `AlignCategories.transform`         |       3.15 / 3.62 ms |   3.53 / 3.72 ms |            0.194 / 0.226 ms direct lookup |

No timed numerical Processor fell back to CPU or transferred data between
host and device. `PowerTransform.fit` remains the dominant latency. The
~1.15 ms `ShuffleColumns` regression versus the old 0.41 ms comes from the
current generic grouped-schema/materialization orchestration; it is too small
to justify a second specialized shuffle implementation. Dense category
lookups remain a medium-risk future optimization because they need explicit
dtype, unseen-category, and bounded-domain validation.

## Large representative production-model run

The largest production-architecture workload that was rerun uses 2,400
context + 600 query rows, 100 features, eight estimators, float32, five
warmups, and 20 repetitions on the L4.

| Task / execution            | Original median / p95 |  Latest median / p95 | Original / latest peak | Median change |
| --------------------------- | --------------------: | -------------------: | ---------------------: | ------------: |
| Classification / vectorized |  2456.50 / 2466.29 ms | 2321.29 / 2339.66 ms |   13221.4 / 1663.4 MiB |         −5.5% |
| Classification / sequential |  2478.10 / 2494.18 ms | 2514.62 / 2542.08 ms |    1661.2 / 1654.8 MiB |         +1.5% |
| Regression / vectorized     |  2517.52 / 2551.87 ms | 2398.20 / 2411.41 ms |   13221.5 / 1679.5 MiB |         −4.7% |
| Regression / sequential     |  2482.94 / 2499.95 ms | 2529.12 / 2605.63 ms |    1676.4 / 1672.8 MiB |         +1.9% |

The default vectorized path is faster than the original and uses about 87%
less peak GPU allocation because PR #516 keeps Recipe preprocessing
vectorized while running the model once per estimator. Sequential's 1-2%
runtime regression is explained by repeated one-member fitting and
transformation; its low-memory behavior is preserved.

## Smallest fixes and remaining gap

The only performance fix added after parity was restored is a packed
regression-output path. The first latest-code measurement was 16.71 ms for
output transformation and 128.89 ms for the public vectorized boundary.
Wrapping one already-stacked `EnsembleTable` group avoids the second
materialization and reuses the current inverse/output processors. Remeasured
GPU values are 11.57 ms and 112.62 ms. On CPU, this preserves final PR #516's
ensemble-aware inverse semantics while reducing the public boundary from
2734.26 to 2693.64 ms and peak RSS from 1264.0 to 1034.9 MiB.

The remaining speed-of-light gap is 29.87 ms classification and 34.22 ms
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
  methodology. Production-model p95 remains within about 1-3% of its median.
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
- `results/pr421_pre_ensemble_cpu_processing.json`
- `results/pr421_pre_ensemble_gpu_processing.json`

```bash
env SDM_RUN_TABICLV2_GPU_PARITY=1 uv run pytest -q \
  test/integration/tabiclv2_parity/test_strict_ensemble.py

uv run python benchmark/tabiclv2_ensemble_revalidation.py \
  --devices cpu --executions vectorized sequential --include-processors \
  --output benchmark/results/pr421_tabiclv2_cpu_processing.json

uv run python benchmark/tabiclv2_ensemble_revalidation.py \
  --devices cuda --executions vectorized sequential --include-processors \
  --output benchmark/results/pr421_tabiclv2_gpu_processing.json

uv run python benchmark/tabiclv2_ensemble_revalidation.py \
  --devices cuda --executions vectorized sequential \
  --context-rows 2400 --query-rows 600 --only-model \
  --output benchmark/results/pr421_tabiclv2_gpu_model_3k.json
```
