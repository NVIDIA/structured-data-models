# Compiled numerical fitting: parity investigation

This branch is an experimental source-level fix for numerical fitting accuracy. It builds on preprocessing integration `8950a466c` plus immutable column-schema work. It is not a claim that every public preprocessing entry point or GPU workload compiles correctly.

## What failed

Compiling PowerTransform fitting changes the summation order in its objective. Small likelihood differences change the comparisons used by the 44-step golden-section search and produce different fitted lambdas. PyTorch 2.7 also produces slightly different `expm1` values in compiled code.

The failing computation can be isolated from unrelated table-schema routing:

```python
processor = PowerTransform()
compiled_fit = torch.compile(processor._fit_transform, fullgraph=True)
with torch.inference_mode():
    actual = compiled_fit(table)
```

This calls the actual processor fitting/transformation implementation. `_fit_transform` deliberately excludes the separate public applicability checks that still obstruct 2.7 fullgraph tracing; it does not replace table containers or mathematical operations with mocks.

On FP32 breast-cancer features (160 rows, 30 original columns plus constant/missing stress columns), the original compiled fitting differed from eager by approximately `0.00157` in transformed values. Default same-dtype `torch.testing.assert_close` failed. Compiling only transformation with eager fitted parameters did not exhibit this parameter drift.

## Source change

The patch keeps three sensitive primitive operations as native PyTorch calls inside a compiled graph:

- NaN-ignoring sums in PowerTransform fitting and Standardize fitting.
- NaN-ignoring means in ImputeMean fitting. Without this, sparse missing values change imputation means and can reintroduce PowerTransform fitting drift downstream.
- `expm1` during PowerTransform fitting, needed for the observed 2.7 pointwise discrepancy.

Small `torch.library.custom_op` boundaries preserve the original primitive implementations. The optimizer, comparisons, iteration count, selected dtype, and normalization formulas are unchanged. Other operations can still compile. Eager execution and computations requiring gradients use their existing implementations. Public metadata types are unchanged.

Cached PowerTransform transformation does not use the new fitting-only `expm1` boundary. The patch makes no blanket replacement of sums elsewhere in the model or compiler.

## Validation

All checks below use CPU Inductor, not the tracing-only eager backend. Tolerances were not relaxed and input dtypes were not promoted.

| Check | PyTorch 2.7.1 | PyTorch 2.14 |
|---|---|---|
| Synthetic fitting, FP32/FP64, both fullgraph settings, 11/7/21 rows | 12 passed | 12 passed |
| Constant, all-missing, individual missing and infinite values | Passed in those cases | Passed in those cases |
| Fitted lambdas, mean and scale | Exact in tested cases | Exact in tested cases |
| Breast-cancer feature fitting, FP32, 160/79 rows | Default output tolerance passed; maximum `3.55e-6` / `1.31e-6` | Exact outputs |
| Real-data reproduction with sparse missing values, 160 rows | Default tolerance passed; maximum `8.02e-6`, fitted parameters exact | Exact outputs and fitted parameters |
| Existing PowerTransform/Standardize/ImputeMean tests | 25 passed, 27 skipped | 25 passed, 27 skipped |
| Focused compiled fitting regression, both fullgraph settings | 2 passed | 2 passed |
| Native-reduction wrapper finite-gradient smoke check | Passed | Passed |

The separate recipe investigation also verified that preserving both sums and imputation means restores default tolerances on the real relational numerical recipe with sparse missing values. That experiment uses a larger integrated branch; see its report for the exact end-to-end scope.

## Performance and memory limits

Correct fitting does not guarantee a speedup. A diagnostic with 30 alternating warm eager/compiled fits on the same FP32 160-by-32 input observed approximately 2.13 ms eager versus 0.99 ms compiled on 2.14. The 2.7 run was slower under compilation (approximately 2.78 ms eager versus 3.42 ms compiled) and had substantial timing variability. These were shared-host CPU measurements, not production performance claims; raw samples are retained.

The native `expm1` returns an intermediate tensor that is copied into the existing workspace. That can increase temporary storage relative to a fully fused compiled expression. GPU peak memory and GPU throughput have not been measured. Validate both before recommending this path for GPU fitting. No GPU instance was launched for this investigation.

## Reproduce

From the repository root, using the desired PyTorch environment:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/numerical-fit-parity/validate.py --fullgraph

PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/numerical-fit-parity/validate.py --dtype float64

PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/numerical-fit-parity/validate.py \
  --dataset breast-cancer --fullgraph --benchmark
```

The real-data reproduction requires scikit-learn for its bundled dataset, or `--data-file PATH` pointing to a cached `(features, targets)` tensor tuple. The script asserts exact fitted parameters and unchanged default output tolerances. It does not call a custom compiler backend; the tested native boundaries are in source code.

## Deferred metadata prototype

A private primitive-string mirror of `handles_stypes` enabled 2.7 fullgraph Standardize fitting while preserving the public frozenset of Stype enums. It required class-construction caching, interception of instance assignments, and live properties for Sequential, TaskDispatch, TableDispatch and adapters. Thirty existing eager tests passed, but later class-level metadata mutation could leave the cache stale, and arbitrary custom processor properties needed additional handling. This prototype was rejected from the source branch rather than introducing those compatibility holes or a larger metaclass mechanism. The 2.7 public metadata blocker remains separate from the fitting arithmetic fixed here.

## Autocast operator contracts

The native fitting `expm1` helper disables autocast only within its operation.
The original fitting code uses `expm1_`, which preserves its input dtype under
autocast. CUDA registers the out-of-place `expm1` for FP32 promotion, so using
it without this guard could disagree with the fake operator's input-dtype
output and with the original fitting calculation. The guard adds no cast.

Installed PyTorch 2.7.1 and 2.14.0 dispatch registrations confirm that
`expm1` has an AutocastCUDA kernel, while `expm1_`, `nansum`, and `nanmean`
have fallthrough registrations. This is source/registration evidence, not a
CUDA execution result.

CPU validation on both versions passed all 18 combinations of three helpers,
FP32/BF16/FP16 inputs, and BF16/FP16 autocast. Eager custom operators and actual
Inductor with either full-graph setting match the original native operation
exactly (including NaNs), and schema/fake checks pass. This does not establish
full low-precision PowerTransform fitting parity or GPU performance.

```bash
PYTHONPATH=. python experiments/numerical-fit-parity/autocast.py --device cpu
# Run separately on a CUDA host; not validated by the CPU results above:
PYTHONPATH=. python experiments/numerical-fit-parity/autocast.py --device cuda
```
