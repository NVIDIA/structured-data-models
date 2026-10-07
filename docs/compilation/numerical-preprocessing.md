# Numerical preprocessing compilation investigation

Base: `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`. This branch includes the tensor-container protocol changes from `compile/tabletensor-preprocessing` and categorical support, followed by a small processor applicability fix. It is an integration experiment, not a claim that the complete model or recipe compiles.

## Change

`Processor.fit`, `transform`, and `fit_transform` decide whether a table contains a semantic type handled by that processor. Previously they constructed a `frozenset` intersection:

```python
if not table.active_stypes & self.handles_stypes:
    return table
```

PyTorch 2.14 fullgraph tracing failed with `SourcelessBuilder.create does not know how to wrap <class 'frozenset'>`. The branch instead checks the existing column metadata, without constructing a set or invoking a tensor property that returns one:

```python
if not any(
    len(columns) > 0 and stype in self.handles_stypes
    for stype, columns in table.columns.items()
):
    return table
```

This preserves the applicability decision. Numerical algorithms, dtypes, fitted state and public processor APIs are unchanged.

## Reproduce

Run from the repository root with each supported Python environment:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python docs/compilation/numerical_processors.py --dynamic

PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python docs/compilation/power_fit_parity.py
```

The first script calls the actual public `fit_transform` and `transform` methods using actual `TableTensor` objects and CPU Inductor. Each case checks 11, 7, and 15 rows; finite values, a constant column, a completely missing column and individual missing values; FP32 and FP64; and `fullgraph=False`/`True`. It asserts same-dtype default PyTorch tolerances, schema and fitted-state parity. `--dynamic` requests symbolic shapes; without it changing shapes may recompile. These scripts report failures without relaxing tolerances. They do not benchmark speed, GPU memory, or downstream model quality.

## Findings

- PyTorch 2.14 supports the tested public Standardize, ImputeMean, RobustScale, RankGaussian and ClipSigma calls after the container and applicability changes, including symbolic row counts and both graph modes.
- PyTorch 2.7.1 with graph breaks allowed passes the same five public processors with symbolic row counts and both dtypes. Fullgraph still fails on `handles_stypes` frozenset handling. Changing the intersection alone does not establish support on that version.
- Independently isolating actual processor arithmetic from table plumbing showed CPU Inductor support on both versions for Standardize, ImputeMean, RankGaussian and ClipSigma fitting/transformation, and PowerTransform/RobustScale cached transformation. These isolated checks are diagnostics, not public-API validation.
- RobustScale fitting has an additional 2.7.1 fullgraph issue: `chunk.nanquantile(q, dim=-2, keepdim=True)` fails with `Data dependent operator: aten.equal.default`. The same operation passed 2.14 Inductor. Replacing a statistics primitive just to bypass an older compiler is not part of this patch.
- DropConstantColumns fitting learns a Python column schema from `_keep_mask(...).tolist()`. This is a separate data-dependent metadata boundary. Cached transformation has an already fitted schema; supporting it does not imply compiling schema selection during fitting. Do not disable column dropping to obtain compilation.
- CUDA chunked numerical processors also use `split_size`, which queries device memory properties. CPU results do not validate that CUDA path. A prepared memory budget can preserve chunking while moving the device query outside tracing.

## PowerTransform needs fitting parity validation

A CPU PyTorch 2.14 fullgraph Inductor check separates compiled fitting from compiled transformation on the same inputs and dtype:

| Case | FP32 max absolute difference | FP64 max absolute difference |
|---|---:|---:|
| Compile transform, reuse eager fitted statistics | 0 | 0 |
| Compile fit_transform | 0.0002503693 | 1.11e-15 |
| Eager transform, reuse compiled fitted statistics | 0.0002503693 | 1.11e-15 |
| Fitted lambda difference | 0.0003452301 | 0 |

The FP32 optimizer selects slightly different lambdas after compiled arithmetic changes its comparisons. Eager transformation with those compiled statistics reproduces the discrepancy, locating it in fitting rather than cached transformation. This is not an FP32-versus-BF16 comparison. The FP32 result fails the default same-dtype tolerance and is reported as a failure. No prediction-quality conclusion follows from this small processor reproduction. Reusing eager fitted statistics while compiling transformation is the validated narrower option.

Existing Standardize, ImputeMean and Recipe tests pass on both CPU runtimes: 15 passed, 10 CUDA cases skipped per version.
