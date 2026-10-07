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

Results are being collected in focused branches for tensor containers, recipe orchestration, numerical/categorical processors, and public model integration. Each implementation branch records its own exact tested scope and remaining errors. This document will consolidate the final evidence.
