# Chronos-2 parity and runtime

This port targets the released `amazon/chronos-2` checkpoint. It supports numerical univariate and multivariate targets, past-only and known-future numerical covariates, NaN observations, context truncation, and forecasts longer than 1,024 steps. It deliberately excludes categorical preprocessing, fine-tuning, and upstream's optional cross-learning between independent tasks. No upstream runtime dependency is needed by SDM.

- Upstream source: `amazon-science/chronos-forecasting@1224fafa91e1ecd6674f5204066b159e40b5e4a1`.
- Checkpoint: `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`.
- Source and weights: Apache-2.0; source attribution is in `sdm/models/chronos2/NOTICE`.

## Use

```python
import torch

from sdm.models.chronos2 import Chronos2

model = Chronos2(device="cuda")
t = torch.arange(256, device="cuda").float()
y = (t / 10).sin().unsqueeze(-1)
out = model(
    x_context=y.new_empty(256, 0),
    y_context=y,
    x_query=y.new_empty(48, 0),
)
# [48, 21], ordered from q1 to q99; float32 for these float32 inputs.
```

The SDM recipe uses `Standardize()` followed by the invertible `Asinh()` independently for features and targets, then averages estimator outputs. Context is capped before recipe fitting in both `forward` and `fit`; future query rows are preserved. The recipe performs initial standardization and the asinh transform; target inversion applies sinh followed by inverse standardization. The model recomputes normalization only after extending the context during long-horizon rollout. Rollout statistics and quantile mixtures are computed on the linear scale, then outputs are returned in asinh space for recipe inversion. `Standardize.inverse_transform` broadcasts fitted target statistics across consecutive quantile outputs. For batched benchmark inputs, an explicit leading estimator axis separates independent batch entries from estimators. Both `Chronos2` and `_Chronos2` live in `model.py`; the package is intentionally not exported from `sdm.models`.

## Reproduce

Install the optional reference library into an existing GPU-compatible SDM environment:

```bash
uv pip install --python .venv/bin/python \
  "chronos-forecasting @ git+https://github.com/amazon-science/chronos-forecasting.git@1224fafa91e1ecd6674f5204066b159e40b5e4a1" \
  "transformers==5.14.1"
.venv/bin/python -m benchmark.timeseries.chronos2.parity \
  --output /tmp/chronos2-results.json --warmup 5 --repeats 15
```

The standalone benchmark creates deterministic sinusoidal inputs without seeds. It compares upstream `Chronos2Pipeline.predict` with SDM's public `forward`, including checkpoint loading and output column order. It separately compares CUDA-resident numerical execution. Public upstream prediction includes CPU dictionary preparation and copying results to CPU; SDM accepts tensors already on CUDA and returns CUDA tensors. Core timings remove that boundary difference and include standardization, asinh, and their inverses for the SDM numerical call. Core timings cover one chunk even when public forecasts require multiple chunks.

All calls run without gradients, using float32 or float16 parameters with float32 scaling and forecast restoration. Models are constructed outside `inference_mode`; only prediction runs inside it. Constructing parameters inside inference mode selected slower execution paths in this environment and is not representative of ordinary model initialization. Float32 matmul precision is `highest` (TF32 disabled). Five warmups precede fifteen alternating paired repetitions, with CUDA events and explicit synchronization around wall timing. JSON output records median latency and wall p10/p90. No compilation or CUDA graphs are used.

## Measurements

The full matrix below was measured after moving standardization into the recipe, before extracting Asinh. The current Asinh recipe follow-up is reported separately below.

Measured on 2026-10-29: NVIDIA GB10, driver 580.178.04, PyTorch 2.14.0+cu130, Python 3.14.7, Transformers 5.14.1, Chronos reference 2.3.3.dev0 at the source revision above. These are latency measurements on synthetic inputs, not forecasting quality benchmark scores.

These results include the recipe standardization. Its extra fitting and inversion add overhead to small public calls, so the initial port's general speedup does not apply uniformly.

Every timing is **upstream → SDM**, in milliseconds, using median synchronized wall time.

| Case              | Float32 public |  Float32 core | Float16 public |  Float16 core |
| ----------------- | -------------: | ------------: | -------------: | ------------: |
| univariate        |    7.12 → 7.20 |   6.60 → 6.13 |    7.45 → 7.58 |   7.16 → 6.73 |
| batch univariate  |  22.26 → 19.61 | 21.93 → 19.07 |   10.03 → 9.57 |   9.52 → 9.00 |
| multivariate      |  12.26 → 11.54 | 11.96 → 11.01 |    7.85 → 7.68 |   7.47 → 6.77 |
| covariates        |  12.48 → 12.54 | 11.89 → 11.02 |    8.08 → 8.58 |   7.55 → 6.82 |
| missing           |   9.60 → 10.32 |   9.02 → 8.78 |    8.00 → 8.46 |   7.48 → 6.77 |
| all missing       |    7.29 → 7.08 |   6.79 → 6.10 |    7.64 → 7.49 |   7.27 → 6.65 |
| constant          |    8.44 → 8.30 |   8.03 → 7.76 |    7.75 → 7.56 |   7.43 → 6.76 |
| one step          |    7.07 → 7.09 |   6.61 → 6.15 |    7.37 → 7.53 |   7.10 → 6.66 |
| truncated context |  33.14 → 32.07 | 32.17 → 30.88 |  14.38 → 14.80 | 13.96 → 14.05 |
| long horizon      |  70.35 → 68.91 | 11.92 → 11.09 |  33.33 → 32.85 |   7.48 → 6.81 |
| three chunks      |  81.38 → 78.47 |   8.69 → 8.45 |  43.59 → 39.73 |   7.79 → 7.11 |

The cases include a sixteen-item univariate batch; two independent three-target groups; mixed target/past/future covariates; irregular patch lengths; internal and leading missing observations; an entirely missing target; entirely missing histories; constant and one-point histories; truncation of 8,301 context steps; and horizons of 1,041 and 2,051 steps. The case shapes are recorded in `parity.py` and the result JSON.

## Asinh recipe follow-up

Moving asinh and its inverse into a processor was checked on twelve comparisons: six shapes in float32 and float16, with five warmups and fifteen alternating paired repetitions. Short-horizon prediction differences were unchanged; long-horizon differences changed only by small rounding effects. The accepted constant-history limitation remains. The command above can reproduce this subset with `--cases univariate covariates constant one_step long_horizon three_chunks`.

Every value is upstream → SDM public synchronized wall latency in milliseconds.

| Case         |       Float32 |       Float16 |
| ------------ | ------------: | ------------: |
| univariate   |   6.60 → 6.83 |   8.17 → 8.29 |
| covariates   | 12.62 → 13.08 |   8.95 → 9.85 |
| constant     |   8.49 → 8.48 |   8.57 → 8.41 |
| one step     |   7.14 → 7.38 |   8.15 → 8.30 |
| long horizon | 70.71 → 69.10 | 34.01 → 34.56 |
| three chunks | 81.09 → 77.69 | 43.73 → 40.90 |

## Plain SDM RMSNorm comparison

Plain `sdm.nn.RMSNorm` is the chosen implementation to keep normalization shared across models. Relative RMSE below `1e-3` is accepted; float32 and ordinary float16 cases satisfy this threshold. The missing-history half-precision cases above that threshold are retained as documented exceptions, without a model-specific normalization fallback.

On 2026-10-30, the local T5-style LayerNorm was replaced with `sdm.nn.RMSNorm(eps=1e-6)` throughout the attention blocks, feed-forward layers, and final normalization. The comparison holds weights and the current Standardize/Asinh recipe fixed, isolating the normalization change from the already accepted constant-scale differences. The baseline substitutes the pinned upstream `Chronos2LayerNorm`, whose arithmetic matches the removed custom class, and preserves shared query/key normalization objects.

```bash
.venv/bin/python -m benchmark.timeseries.chronos2.norm \
  --output /tmp/chronos2-norm-results.json --warmup 3 --repeats 15
```

This covers eleven cases in each of three modes: float32 parameters, float16 parameters, and float32 parameters with CUDA float16 autocast. The last mode exercises SDM's optimized RMSNorm kernel. Models are constructed outside inference mode. All 33 public-API comparisons produced finite outputs. Wall and CUDA event timing use warmups, alternating paired repetitions, and explicit synchronization on the same GB10.

Differences below compare **plain RMSNorm against the previous custom normalization**, with identical weights and preprocessing. Speedup is the median percentage latency reduction across the eleven cases, not a forecasting accuracy metric.

| Mode             | Largest relative RMSE | Case with largest difference | Median latency reduction |
| ---------------- | --------------------: | ---------------------------- | -----------------------: |
| Float32          |              1.71e-06 | missing                      |                     5.1% |
| Float16 weights  |                1.454% | all missing                  |                    13.3% |
| Float16 autocast |                0.466% | missing                      |                    12.4% |

For ordinary univariate, batched, multivariate, covariate, and long-context inputs, float16-weight relative RMSE is 0.033–0.087%, and autocast relative RMSE is 0.026–0.053%. Entirely missing histories with float16 weights have the largest change: 1.45% relative RMSE and maximum absolute difference 694,210 because forecasts are already in the millions. Strict half-precision parity is not claimed. Float32 changes remain at floating-point rounding scale; the maximum relative RMSE is 1.71e-6.

The JSON also records old-versus-upstream and new-versus-upstream errors separately. The previously accepted Standardize constant-history and one-observation discrepancies remain; they must not be attributed to this norm swap.

## MLP ownership comparison

`Chronos2TransformerBlock` now accepts `mlp: torch.nn.Module | None`. Time blocks receive `None`; group-attention blocks own the normalized feed-forward network, as in TimesFM-3. The separate MLP list is removed, and checkpoint remapping targets `icl_block.group_blocks.<layer>.mlp`.

The 2026-10-30 comparison held SDM RMSNorm, weights, and preprocessing fixed and restored the saved pre-change ICL forward for the baseline. It covered eleven shapes in each of float32, float16-weight, and float16-autocast modes, with three warmups and ten alternating paired repetitions. Float32 results were bit-for-bit identical in all eleven cases. Ordinary half-precision cases, including long context and multi-chunk horizons, were also bit-for-bit identical.

Differences appeared only in missing-history and constant-history cases. Stored float16 weights had maximum aggregate relative RMSE 3.94e-5 (constant case, maximum absolute difference 0.0346). Float16 autocast reached 0.00382 relative RMSE on the mixed missing-history case. Constant-case aggregate errors are dominated by the 100-valued target: individual near-zero quantile outputs had larger relative errors, up to 15.6% with float16 weights and 8.92% under autocast. These are documented degenerate-input exceptions, not a claim that every half-precision output meets the 1e-3 tolerance.

Latency was effectively unchanged: median changes across cases were -0.49% for float32, -0.36% for float16 weights, and -0.75% for float16 autocast. All outputs remained finite. The standalone comparison, saved before-change source, and JSON results are in `/tmp/sdm-chronos2-mlp-20261030/` for this run.

## Numerical agreement

CPU float32 checks cover univariate, covariates, constant, and missing data. Cached `fit`/`predict` uses the same capped context and fitted recipe as direct prediction; empty horizons preserve batch and output column dimensions.

Public output shape and target/quantile column ordering matched in the original 22 comparisons and the twelve Asinh follow-up comparisons. The full-matrix precision measurements below refer to the Standardize-only recipe; the Asinh follow-up retained those precision characteristics. For nonconstant histories and entirely missing histories, float32 relative RMSE (`RMSE(reference - SDM) / RMS(reference)`) is at most **1.83e-6**. Ordinary multivariate and covariate cases have maximum absolute errors below **1.5e-6**. The 2,051-step rollout reaches **2.06e-4** maximum absolute error, with relative RMSE **8.07e-7**.

This is numerical agreement, not bitwise identity. Entirely missing histories can generate forecasts in the millions upstream: float32 differences reach **60** in those cases despite relative RMSE below **1.83e-6**. Fused QKV projections and different tensor layouts change floating-point rounding; inverse `sinh` amplifies it. These cases must not be judged using absolute error alone.

Float16 is less exact: ordinary batched/multivariate/covariate cases have relative RMSE **0.050–0.058%** and maximum absolute error below **0.0048**. Relative RMSE reaches **0.44%** across three rollout chunks and **0.83%** in the mixed missing-target case. The latter has **244,213** maximum absolute difference because of extremely large upstream forecasts. All measured outputs remain finite, but strict float16 parity is not claimed for these unstable inputs; use float32 when matching the reference is important.

**Accepted parity limitation:** the recipe deliberately uses unchanged `Standardize()` fitting semantics. For constant histories it uses scale 1 rather than upstream's 1e-5. Removing the previous duplicate normalization exposes this difference. The constant-series case has relative RMSE **1.32%** (float32) or **1.33%** (float16), with maximum absolute difference **5.98** or **6.01**. With only one context observation, output scale differs by roughly **100,000×**; the maximum absolute difference is **11.7 million**. These cases do not have upstream parity. The choice to retain Standardize's default scale was made explicitly after measuring these differences. They are prediction disagreements on synthetic inputs, not measured changes in forecasting quality on real datasets.

The core uses float32 NaN reductions instead of `RevIN.stats`' float64 reductions. The recipe retains SDM Standardize's finite-value and constant-column rules. Affine normalization reuses `RevIN`. The implementation now uses plain SDM RMSNorm; its measured precision tradeoffs relative to the previous T5-style normalization are recorded above. Attention uses scale 1.0 and upstream's finite additive masks, including its fully masked-row behavior. Long forecasts use nine quantile paths and the weighted empirical CDF rather than median-only feedback.
