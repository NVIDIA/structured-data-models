# Preprocessing compilation investigation

Baseline: `842c408fe2a8711bdf2e7cff4bfbe54266d6b940` (2026-10-08).

This investigation targets the computations before KumoTabular and KumoRelational internal model calls. It separates compiler compatibility from speed claims and preserves preprocessing statistics, missing-value behavior, categorical mappings, and context-only fitting.

## Scope

- CPU PyTorch 2.7.1 and 2.14.0; same input dtype for eager/compiled comparisons.
- Actual Inductor execution with `fullgraph=False` and `fullgraph=True`. Capture-only diagnostics are labeled separately.
- Public processor fitting and fitted transformation, tensor container inputs/outputs, and public model fit/predict/forward calls.
- Changing row counts, empty/missing values, and noncontiguous tensors where supported by eager execution.
- No GPU performance claims from CPU checks. No cloud resources are required for reproducing the initial tracing failures.

## Independent compilation boundaries

1. Recipe construction: build processor/module trees and validate configuration.
2. Fitting: learn statistics and categorical mappings from context rows; some steps also choose output columns.
3. Fitted transformation: apply the learned state to query rows.
4. Relational graph preparation: table joins and task-row/edge mappings.
5. Internal model computation.
6. Output transformation and formatting.

Compiling `model.predict` traces the called preprocessing and internal model code, subject to graph breaks. Compiling the outer module traces its `forward`, not arbitrary sibling methods such as `fit` or `predict`.

## Acceptance criteria

A successful no-op wrapper is not evidence of compiled preprocessing. Passing results must execute tensor computation through the reported backend. Graph-break-allowed and fullgraph results are reported independently. A failure after an earlier fix is a remaining blocker, not a passing end-to-end result.

Data-dependent fitting and schema changes must retain eager behavior. Fixed schemas can be specialized by the compiler; a different schema may require recompilation. Returning a view must preserve its relationship to its source. A workaround that drops a missing-value mask, changes category identities, or alters learned statistics is not acceptable.

## Results and branch map

The branches below separate container support, recipe orchestration, numerical/categorical processors, and public model integration. Their reports contain exact commands and remaining errors; this document summarizes the combined evidence.

## Confirmed progress

| Boundary | PyTorch 2.7.1 CPU | PyTorch 2.14 CPU |
|---|---|---|
| Numerical TableTensor construction, replacement, slicing, numeric stacking | Inductor passes both graph policies | Inductor passes both graph policies |
| Standardize, ImputeMean, RobustScale, RankGaussian, ClipSigma public fit/transform | Graph-break-allowed matrix passes; fullgraph blocked by metadata | Both policies pass FP32/FP64 matrix |
| Default fitted feature recipe, real numerical data | Fullgraph still blocked | Fullgraph passes, including four tabular / two relational task-table members |
| PowerTransform fitting in FP32 | Not established as parity-safe | Fitted parameter drift exceeds default tolerance in one reproduction |
| PowerTransform transform with the same eager-fitted parameters | See component report | Exact in isolated reproduction |
| Entire public KumoTabular predict after eager fit | Further validation in integration report | Classification passes both policies, rows 4→7→3→4; 1/4 estimators; max differences 8.94e-7 / 3.28e-7 |
| Entire public KumoRelational predict | Still fails enum/context-manager tracing | Still fails categorical string metadata / variable-length concatenation |

A changed input row count being accepted does not imply one reusable graph: some categorical processors recompile at each tested size despite dynamic=True.

## Branches

| Branch | Purpose / dependencies |
|---|---|
| [compile/nested-tensor-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/nested-tensor-preprocessing) | Categorical tracing, fake-safe representations, empty string payloads; based on main. |
| [compile/tabletensor-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/tabletensor-preprocessing) | Table/columnar flatten-unflatten support; includes nested tensor prerequisites. |
| [compile/numerical-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/numerical-preprocessing) | Public processor metadata checks; includes container prerequisites and numeric validation. |
| [compile/categorical-preprocessing](https://github.com/NVIDIA/structured-data-models/tree/compile/categorical-preprocessing) | Focused numeric category lookup and tracing fixes, based on main; whole-processor validation also uses the container/processor integration. |
| [compile/recipe-preparation](https://github.com/NVIDIA/structured-data-models/tree/compile/recipe-preparation) | Focused recipe validation and ensemble packing changes; validated with container/applicability prerequisites. |
| [compile/public-preprocessing-investigation](https://github.com/NVIDIA/structured-data-models/tree/compile/public-preprocessing-investigation) | Combined experimental stack, public entry-point reproducer, schema validation and cached class-label metadata. |

These are investigation branches, not a set of independent patches to merge blindly. The component reports identify source commits and prerequisite changes. Public integration results require the full tested combination.

## Remaining work

- Preserve TableTensor mutation semantics during AOT functionalization: current wrappers lack aten.copy_ support for compiled input mutation.
- Handle nonempty string/ragged storage operations without tensor-value-driven Python slicing; preserve view offsets and aliasing rather than forcing contiguous layouts.
- Resolve older-runtime metadata guards (including frozenset of Stype values) and symbolic string-wrapper layout limitations.
- Separate fitting steps that select a schema/vocabulary from fitted tensor transformations. DropConstantColumns currently turns a learned tensor mask into Python column metadata.
- Isolate or implement graph-compatible alternatives for external Arrow joins. Wrapping Arrow in a graph break does not make the join itself compiled.
- Preserve numerical fitting parity for PowerTransform; compiled transformation using fixed learned state is a separate, better-supported case.
- Validate GPU execution, autocast dtypes, memory chunking, and performance only after the CPU execution boundary is stable. No speed or memory improvement is asserted here.

## Independent regression review

At container snapshot 14d16c583, the existing TableTensor, ColumnarTensor, and CategoricalTensor test suites passed 95 CPU tests on PyTorch 2.14; 27 CUDA tests were skipped. Later component branches ran their own updated checks. This does not constitute GPU validation.


## Practical support boundaries

The fitted numerical default recipe is substantially closer to support than compiling all fitting. On PyTorch 2.14, the default four-member tabular recipe and two-member relational task-table recipe transform real numerical query data with exact eager parity and unchanged fitted buffers. These checks do not include string alignment or related tables.

Public KumoTabular prediction passes the tested numerical classification cases after the combined patches. Query sizes 4→7→3→4 create three graphs and reuse the first graph on the final call. Broader validation exposed a sliced-table/FakeTensor-mode failure; commit 15917f89e fixes empty identifier view-leaf propagation, and the full public one/four-estimator classification and one-estimator regression checks subsequently pass with sliced inputs too. Mixed categorical/text inputs retain separate blockers. Regression fresh-input checks pass at atol=1e-5, rtol=1e-4 with maximum absolute difference 1.83e-4; this is numerical prediction parity within tolerance, not a task-metric or GPU claim.

Public fitting is harder for concrete reasons: processors learn data-dependent column sets and vocabulary sizes, some fitting uses Python lists/random permutation metadata, and compiled FP32 PowerTransform optimization can select different parameters. Keep fitting parity separate from applying existing fitted statistics.

For PyTorch 2.7, changing handles_stypes to a tuple or primitive-string frozenset gets past one error but then fails enum-key dictionary guarding. Those exploratory metadata changes were not retained. A separate dynamic-stride issue is reproduced with a small raw PyTorch wrapper, without importing SDM; the nested-container branch contains it.

## Suggested implementation order

1. Review the independent numerical category lookup fix and the tensor flatten/unflatten foundations, with explicit alias and mutation coverage.
2. Integrate recipe/applicability changes and target compiled fitted preprocessing on PyTorch 2.14 first. Retain the documented 2.7 limitations rather than claiming equivalent support.
3. Review sliced-table integration and cached output metadata, then extend full public prediction to mixed categorical/text data and GPU execution.
4. Address fitting separately: data-dependent schemas/vocabularies, numerical optimizer parity, and external joins require distinct decisions. Do not disable all preprocessing merely to obtain a successful compiler call.

No PRs were opened. No EC2 instances were launched: the decisive failures and initial passing paths were reproducible on CPU. Performance and GPU correctness remain explicit follow-up validation, not inferred from these results.


## Relational public prediction after integration

On the combined branch, including graph preparation and #1068, compiled public predict still fails before the internal model. On 2.14 fullgraph, AlignCategories accesses a tuple of StringTensor category dictionaries that Dynamo cannot source; with breaks allowed, variable-length concatenation raises a data-dependent scalar error. On 2.7 the first reported failures remain a generic-context graph-break restriction and enum dictionary guards. These are public preprocessing failures, distinct from previously passing internal-model cached prediction checks.


## Final public prediction validation

The combined branch is published at commit `8950a466ccbd501f7b1431b5b4e63a8e37d9de09`. Its [README](https://github.com/NVIDIA/structured-data-models/blob/compile/public-preprocessing-investigation/experiments/compile_public_paths/README.md) and [compact final records](https://github.com/NVIDIA/structured-data-models/blob/compile/public-preprocessing-investigation/experiments/compile_public_paths/prediction-final.json) preserve exact cases and errors.

| KumoTabular, CPU 2.14, eager fit then compiled public predict | Graph breaks allowed | Fullgraph | Maximum absolute difference |
|---|---|---|---|
| Classification, one estimator | Pass | Pass | 8.94e-7 |
| Classification, four estimators | Pass | Pass | 3.28e-7 |
| Regression, one estimator | Pass | Pass | 1.8311e-4 |

Fresh tables and sliced views passed row counts 4→7→3→4. Tolerance is atol=1e-5 and rtol=1e-4, comparing the same dtype. Three graphs were compiled for the three row counts; the repeated size reused its graph. This establishes the tested numerical workloads, not arbitrary schemas or all model sizes. Model/standardizer regression tests passed 49 cases on each runtime, with 14 CUDA skips.

The integration branch also includes the existing Fourier embedding fix (#1054), relational graph preparation, and relative-time fix (#1068). Component branches overlap; consult their dependency descriptions before extracting changes. No passing compiled public fit or outer-forward case is established.
