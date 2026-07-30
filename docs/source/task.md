Implement the design described in:

- `/home/rbendias/code/sdm-105/structured-data-models-ensemble-aware-design/docs/source/ensemble_aware_processing_design.md`
- `/home/rbendias/code/sdm-105/structured-data-models-ensemble-aware-design/docs/source/ensemble_aware_processing_problem.md`

Use the current branch branch (agent/ensemble-aware-processing-design) but update to latest `main` to implement the complete end-to-end implementation.

## 1. Follow `AGENTS.md`

Read the repository’s main `AGENTS.md` before planning or changing code.

Treat it as a binding implementation contract throughout the task. Re-check the final implementation against every relevant instruction before completion.

## 2. Tests first

Before implementing the new behaviour, add tests covering problematic and special Recipe compositions in ensembles.

Include nested combinations of:

- `Sequential`
- `Choice`
- TaskDispatch
- Schema changing Processors
- shared and estimator-specific processors
- processors before and after estimator reduction
- feature, target, and output pipelines
- branches that can be shared across estimators
- branches that must remain estimator-specific
- mixed classification and regression behaviour
- invalid or ambiguous compositions that must fail clearly

The tests should define the expected execution semantics, not mirror implementation details.

## 3. Implement ensemble-aware execution

Implement the design so that all tests pass.

The implementation must support:

- optimized parallel processing when memory permits
- sequential estimator processing as a fallback for limited GPU memory
- deterministic ensemble materialization
- round-robin `Choice` selection for normalization methods
- eight-estimator TabICLv2 execution matching the reference behaviour
- correct estimator-specific fitted state
- correct output remapping and estimator reduction
- `TargetDecode()` for regression instead of target.inverse transform
- nested Recipes without inconsistent execution behaviour

Keep the implementation minimal and consistent with the design documents.

## 4. Parity validation

Reuse the existing parity infrastructure from https://github.com/NVIDIA/structured-data-models/pull/271

Validate:

- TabICLv2 parity with eight estimators
- RFM parity

For RFM, create a representative dataset that triggers all relevant processors and additionally contains:

- ID columns
- datetime columns
- numerical and categorical features
- missing values
- constant features
- outliers
- unseen test categories

If parity fails, identify the first semantic divergence, implement the smallest root-cause fix, and rerun.

Create a canonical relational parity dataset specifically designed to exercise every supported processor. The dataset should include multiple related tables, primary/foreign keys, IDs, datetime columns, categorical and numerical features, missing values, unseen categories, constant features and outliers. Reuse this dataset for all RFM parity tests.

## 5. Performance benchmarks

Reuse the original benchmarks and run them for:

- 50,000 rows
- CPU
- GPU
- optimized parallel execution
- sequential fallback execution

Report:

- total processing time
- relevant stage and Processor timings
- median and p95 runtime
- peak CPU/GPU memory
- hardware and dtype
- speedup of optimized execution over sequential execution

Optimize the implementation based on measured bottlenecks while preserving correctness and parity.

## 6. L4 memory crash test

On an NVIDIA L4, determine the largest row count for which parallel TabICL processing completes successfully.

Use a controlled search to report:

- largest successful row count
- first failing row count
- peak GPU memory
- dataset shape and characteristics
- batch and estimator configuration
- whether sequential execution succeeds for the failing parallel case

The test must handle out-of-memory failures cleanly and release GPU memory between attempts.

## Completion criteria

The task is complete only when:

- all new and existing relevant tests pass
- TabICLv2 parity passes
- RFM parity passes
- round-robin normalization choice works for eight estimators
- target inverse transform is integrated into the recipe for regression
- CPU and GPU benchmarks are complete
- parallel and sequential execution are both supported
- the L4 memory limit is documented
- measured bottlenecks have been addressed where reasonably possible
- a final review of `AGENTS.md` against the implementation proposes no further required changes
- a draft PR is open

The final summary must include:

- draft PR link
- test and parity status
- main implementation decisions
- CPU and GPU performance
- parallel versus sequential comparison
- L4 memory threshold
- remaining limitations or follow-up work