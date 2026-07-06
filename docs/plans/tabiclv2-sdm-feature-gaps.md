# TabICLv2 SDM feature gaps

## Document metadata

| Field | Value |
|---|---|
| Status | Draft — gap inventory verified against the current branch |
| Working branch | `tabicl-benchmarking` |
| SDM baseline commit | `1c8e017cabfbeda12b599100fcbfc663a1b4529d` |
| Last verified | 2026-07-06 |
| Parent plan | [TabICLv2 TabArena benchmarking plan](tabiclv2-tabarena-benchmarking.md) |

This document is the implementation-oriented inventory of SDM features and
behavior needed to evaluate the SDM TabICLv2 implementation against the
original implementation and TabArena. It distinguishes:

- behavior that is completely missing;
- primitives that exist but are not wired into TabICLv2;
- behavior that should initially live in the benchmark predictor/adapter; and
- broader reusable APIs that can wait until parity is proven.

## Priority definitions

- `P0 — parity gate`: required before claiming numerical or prediction parity.
- `P1 — benchmark gate`: required for reliable end-to-end TabArena metric,
  timing, and artifact comparison.
- `Later — productization`: useful generic/public functionality that should not
  block the first validated evaluation pipeline.

## Readiness verdict

The current SDM implementation is suitable for low-level architecture
development, but it is not yet a drop-in replacement for the original
TabICLv2 estimator.

It can currently:

- instantiate classifier and regressor neural networks;
- remap and load the published checkpoint weights;
- execute direct in-context forward passes;
- build and reuse an SDM key/value cache;
- return raw 10-class logits or 999 regression quantiles; and
- run part of the original single-estimator regression preprocessing pipeline.

It cannot currently reproduce the complete benchmarked estimator behavior
because it lacks a high-level task-aware predictor, dataframe/categorical
encoding, complete preprocessing, member generation, output decoding, an
AutoGluon adapter, and the required parity tests.

## Current public interface and constraints

### Model construction

[`TabICLv2`](../../sdm/models/tabiclv2/model.py) currently accepts:

```python
TabICLv2(
    pretrained: bool = True,
    device: torch.device | str | None = None,
)
```

It creates both a fixed 10-class classifier and a 999-quantile regressor. With
`pretrained=True`, it loads both published checkpoint files. There is no public
task selector, checkpoint path, Hugging Face revision, offline-only flag,
precision, batch size, seed, or estimator count.

### Task dispatch

The model chooses the task from `y.dtype`:

- floating target -> regression;
- integer target -> classification.

This is unsafe at a dataframe/benchmark boundary. Integer-valued regression
targets can be misclassified, and numeric class labels represented as floating
values can be treated as regression. The integration layer needs an explicit
problem type.

### Input boundary

[`BaseModel._preprocess`](../../sdm/models/base.py) passes only numerical
`TableTensor` features into the network. Categorical feature blocks are warned
about and ignored. Every feature intended for the benchmark must therefore be
encoded into the numerical block before calling the model.

### Output boundary

- Classification returns raw logits with final size 10, even for binary or
  smaller multiclass problems.
- Regression returns 999 raw quantiles.
- There is no `classes_`, `predict_proba`, class decoding, target inverse,
  ensemble aggregation, or NumPy/pandas boundary.

### Cache API

[`BaseModel.fit`](../../sdm/models/base.py) builds a cache and returns `None`.
[`BaseModel.predict`](../../sdm/models/base.py) returns a device tensor.
[`BaseModel.clear`](../../sdm/models/base.py) discards the cache.

Current tests verify that direct and cached calls agree for small random
tensors, but do not verify chunked prediction, reference parity, cache/device
movement, or timing boundaries.

## Existing primitives that are not fully wired

These should be reused where they match the reference. Do not reimplement them
inside the adapter without a demonstrated mismatch.

| Primitive | Current state | Required action |
|---|---|---|
| [`Identity`](../../sdm/processing/identity.py) | Implemented, exported, and tested | Remove it from stale “not implemented” comments; use only where an explicit no-op is useful |
| [`SoftmaxTemperature`](../../sdm/processing/postprocess.py) | Implemented and tested | Wire classification temperature `0.9` after confirming aggregation order |
| [`Power`](../../sdm/processing/power.py) | Tensor-native Yeo–Johnson transform exists | Add original/sklearn parity fixtures before using it in ensemble members |
| [`Quantile`](../../sdm/processing/quantile.py) | Implemented with deterministic seed support | Validate against the reference before enabling non-default normalization options |
| [`MeanImpute`](../../sdm/processing/impute.py) | Implemented and used by the regression recipe | Add reference cases for all-missing, mixed-missing, and constant columns |
| [`StandardScale`](../../sdm/processing/standard_scale.py) | Implemented and used | Verify exact epsilon, degrees-of-freedom, and constant-column semantics |
| [`SigmaClip`](../../sdm/processing/sigma_clip.py) | Implemented, used, and has a TabICL reference-value test | Keep as the pattern for golden processor tests |
| [`Clip`](../../sdm/processing/clip.py) | Fits quantile-derived bounds | Do not use it as the missing stateless `[-100, 100]` clamp; semantics differ |
| [`Pipeline`](../../sdm/processing/pipeline.py) | Supports shape-preserving transforms of a table's numerical block | Use independent pipelines per member initially; it cannot yet remove/relabel columns safely |
| [`Recipe`](../../sdm/processing/recipe.py) | Bundles feature, target, and output pipelines | The caller must invoke it; the model does not apply it automatically |
| [`Cache`](../../sdm/cache.py) and model `fit`/`predict` | Cache-backed inference exists | Add chunking, device, cleanup, and controlled timing tests |

Important documentation correction: the current
[`tabiclv2/recipe.py`](../../sdm/models/tabiclv2/recipe.py) comments list
`Identity` among unimplemented processors, but it already exists. `Clip`,
`Power`, `Quantile`, and `SoftmaxTemperature` also exist, although they do not
yet provide the complete recipe behavior required here.

## Summary gap matrix

| ID | Priority | Gap | Initial implementation location | Blocks |
|---|---|---|---|---|
| `GAP-01` | P0 | Frozen reference/checkpoint contract | benchmark manifest/config | every parity claim |
| `GAP-02` | P0 | Real-checkpoint raw network parity | tests plus existing model | preprocessing work |
| `GAP-03` | P0 | Task/checkpoint selection | predictor or core constructor | clean task-specific execution |
| `GAP-04` | P0 | Dataframe and categorical encoding | predictor boundary | all mixed-type datasets |
| `GAP-05` | P0 | Fixed `[-100, 100]` clamp | processing primitive | exact preprocessing |
| `GAP-06` | P0 | Constant-column filtering | predictor-local mask first | exact preprocessing |
| `GAP-07` | P0 | Feature permutation | predictor/member orchestration | ensemble parity |
| `GAP-08` | P0 | Class encoding/permutation/inverse | predictor | classification parity |
| `GAP-09` | P0 | Classification output decoding | predictor | `predict_proba` |
| `GAP-10` | P0 | Regression output decoding | predictor | regression `predict` |
| `GAP-11` | P0 | Member normalization and aggregation | predictor | default 8-member parity |
| `GAP-12` | P1 | AutoGluon/TabArena adapter | optional integration module | TabArena execution |
| `GAP-13` | P1 | Prediction chunking and memory controls | predictor/adapter | scalable inference |
| `GAP-14` | P1 | Controlled timing instrumentation | benchmark CLI | inference-time claim |
| `GAP-15` | P1 | Prediction/result artifacts and comparator | benchmark CLI/analysis | diagnosable parity |
| `GAP-16` | P1 | Optional dependency/environment story | packaging/runbook | reproducible setup |
| `GAP-17` | P1 | Resource cleanup and metadata | adapter | repeatable jobs |
| `GAP-18` | P1 | Documentation/API consistency | README/docs | usable public guidance |

## P0 — parity gates

### GAP-01 — Frozen reference and checkpoint contract

Current gap:

- checkpoint filenames and repository are hardcoded;
- Hugging Face revision and blob hashes are not part of the public API;
- installed dependency versions are unpinned; and
- benchmark inputs are not described by a machine-readable manifest.

Required behavior:

- record original implementation version/source;
- record classifier and regressor checkpoint paths, repository revision, and
  SHA-256;
- record dataset, split, problem type, preprocessing options, estimator count,
  seeds, dtype/autocast, device, and prediction chunk size;
- download all artifacts before timed regions; and
- support offline reruns from the frozen artifacts.

Required tests/evidence:

- manifest schema test;
- wrong/missing hash failure;
- offline checkpoint reload; and
- environment and task-grid fingerprints.

Definition of done:

- a run artifact uniquely identifies every input needed to reproduce it;
- no moving remote reference is resolved during a timed or certified run.

### GAP-02 — Raw architecture and checkpoint parity

Current gap:

- [`test/models/test_tabiclv2.py`](../../test/models/test_tabiclv2.py) uses
  `pretrained=False`;
- `_remap_ckpt` has no direct golden test; and
- no test compares original and SDM outputs.

Required behavior:

- load the same frozen checkpoint into original and SDM implementations;
- compare already-preprocessed inputs before any high-level estimator logic;
- test classifier logits and regression quantiles;
- test direct and cached inference; and
- debug mismatches through intermediate activation comparisons.

Required tests:

- binary and multiclass fixtures;
- regression fixture;
- CPU FP32, CUDA FP32, and intended BF16/AMP mode;
- strict state-dict load/remap; and
- deterministic repeat.

Definition of done:

- FP32 outputs pass committed tolerances;
- reduced-precision behavior has separately declared tolerances;
- failure blocks all TabArena runs.

### GAP-03 — Explicit task and checkpoint selection

Current gap:

- model construction always creates and loads both variants;
- task selection relies on target dtype;
- there is no local path, revision, or offline-only control.

Minimum initial behavior:

- the benchmark predictor/adapter accepts explicit `problem_type` and a pinned
  checkpoint per problem type;
- it validates that the loaded checkpoint hash matches the manifest.

Preferred reusable core behavior after design review:

```python
TabICLv2(
    task="classification" | "regression" | "both",
    checkpoint_path=...,
    revision=...,
    local_files_only=True,
    device=...,
)
```

Required tests:

- classifier-only and regressor-only construction;
- missing or wrong checkpoint;
- strict load;
- offline-only behavior; and
- integer regression target routed correctly through explicit task.

Definition of done:

- the selected task and exact checkpoint are explicit and logged;
- timed runs make no network calls;
- resource reports state which model variants are resident.

### GAP-04 — Dataframe and categorical feature encoding

Current gap:

- `TableTensor.from_pandas` needs explicit semantic types;
- the low-level model ignores categorical feature blocks;
- no fitted train-time category mapping exists at the predictor boundary.

Required reference-compatible behavior:

- identify string, object, category, and boolean columns;
- ordinal-encode categoricals;
- represent missing and unknown categories exactly as the reference does;
- mean-impute numerical columns;
- preserve feature order and names for fit/predict validation;
- apply only train-fitted maps to test data; and
- convert the final matrix to the model's device/dtype without silently dropping
  columns.

Required tests:

- mixed pandas dtypes;
- missing numerical and categorical values;
- unseen test category;
- all-missing category;
- changed/missing/reordered test columns;
- boolean features; and
- float32/float64 input.

Definition of done:

- original and SDM encoded matrices match on golden fixtures;
- every intended benchmark feature reaches the numerical model input.

### GAP-05 — Stateless fixed-value clamp

Current gap:

- the active recipe explicitly omits the reference `[-100, 100]` clamp;
- existing `Clip` learns quantile bounds and is not equivalent.

Required behavior:

- add an unambiguous stateless processor such as
  `FixedClip(min_value=-100.0, max_value=100.0)`, or extend `Clip` only if its
  two modes remain clear;
- preserve NaNs, device, and floating dtype;
- insert it at the exact reference position after standard scaling.

Required tests:

- values inside, at, and outside bounds;
- positive/negative infinity;
- NaNs;
- CPU/CUDA and float32/float64; and
- golden reference pipeline values.

Definition of done:

- processor and pipeline intermediates match the frozen reference.

### GAP-06 — Constant-column filtering

Current gap:

- `ConstantFilter` is absent;
- existing `Pipeline` assumes a shape-preserving numerical block and reuses
  existing column metadata.

Initial low-risk behavior:

- fit an adapter/predictor-local feature mask on training rows;
- apply the same mask and order to every test batch and ensemble member;
- match the reference treatment of constant and all-missing columns;
- define behavior if no feature survives.

Do not force this into `Pipeline` until schema-changing processor semantics are
designed.

Required tests:

- no constants;
- one and multiple constants;
- all-NaN columns;
- train constant/test varying;
- column order preservation; and
- zero-survivor case.

Definition of done:

- the same columns survive as in the original reference, in the same order;
- test data never changes the mask.

### GAP-07 — Deterministic feature permutation

Current gap:

- no `FeaturePermute(method="latin")` behavior is wired into the model.

Required behavior:

- reproduce the original Latin permutation schedule from estimator/member index
  and seed;
- apply identical permutations to training and test rows;
- retain permutation provenance for diagnostics;
- handle the one-feature and very-wide-feature cases.

Initial location:

- predictor/member orchestration, not a generic processor.

Required tests:

- known golden permutations;
- fixed-seed reproducibility;
- distinct member coverage;
- train/test consistency; and
- one-feature behavior.

Definition of done:

- each member receives the same feature order as its original counterpart.

### GAP-08 — Class encoding, permutation, and inverse alignment

Current gap:

- no label encoder or `classes_` metadata;
- no class-shift member logic;
- no inverse probability-column alignment;
- model always emits ten logits.

Required behavior:

- encode arbitrary labels to contiguous `[0, K)` indices;
- preserve original class order;
- reject missing training labels;
- validate the supported `K <= 10` limit;
- apply each member's deterministic class permutation;
- inverse-permute output columns before member aggregation; and
- map final predicted indices back to original labels.

Required tests:

- binary and multiclass;
- string and numeric labels;
- non-sorted source class order;
- exactly ten and more than ten classes;
- missing labels; and
- round-trip probability-column alignment.

Definition of done:

- every member's output columns are in canonical class order before
  aggregation;
- predicted labels round-trip to the original label domain.

### GAP-09 — Classification output decoding

Current gap:

- raw ten-class logits are returned directly;
- no `predict_proba` exists;
- `SoftmaxTemperature` is not wired.

Required behavior for the default reference path:

1. select the actual `K` class logits;
2. inverse-align each member's class columns;
3. average member logits;
4. apply temperature `0.9` softmax in the confirmed reference order;
5. normalize defensively only if the reference does; and
6. return CPU probability arrays with shape `[n_test, K]` in canonical class
   order.

Required tests:

- shape and row order;
- probability sums and finiteness;
- exact member aggregation order;
- label decoding;
- binary and multiclass metric fixtures; and
- original high-level prediction parity.

Definition of done:

- `predict_proba` and `predict` match the original estimator on frozen fixtures.

### GAP-10 — Regression output decoding

Current gap:

- the model returns 999 raw quantiles;
- target inverse scaling and ensemble point prediction are caller-managed;
- no high-level regression `predict` exists.

Required behavior:

- standardize the training target using reference semantics;
- obtain the 999 quantiles;
- apply the reference monotonicity behavior when relevant;
- compute the reference default point prediction. In the currently inspected
  original estimator this is the mean of the monotonic quantiles, not quantile
  index 499;
- inverse-transform the target scale per member; and
- average point predictions across ensemble members.

The frozen historical version must be checked before treating the inspected
default as historical fact.

Required tests:

- target scaling/inverse round-trip;
- mean versus median distinction;
- quantile crossing case;
- integer-valued regression target with explicit task;
- one-member and multi-member output; and
- original high-level prediction parity.

Definition of done:

- regression predictions and per-task RMSE match the frozen reference within
  declared scale-aware tolerances.

### GAP-11 — Member normalization and ensemble orchestration

Current gap:

- no implemented `n_estimators` contract despite README examples;
- no independent per-member fitted recipes;
- no normalization-choice schedule or output aggregation.

Required default behavior:

- `n_estimators=8`, plus `1` for diagnostic runs;
- default normalization methods `none` and Yeo–Johnson `power` in the exact
  reference schedule;
- independent fitted state for each active normalization path;
- feature and class permutations tied to the same member identity;
- member batching/chunking that does not change output; and
- task-specific aggregation described in GAP-09 and GAP-10.

Initial location:

- `TabICLv2Predictor`, using explicit member objects/loops or a leading batch
  dimension. A generic `Choice` processor is not required for first parity.

Required tests:

- one member equals the single-estimator path;
- deterministic eight-member schedule;
- original intermediate member matrices;
- independent fitted state;
- loop versus stacked execution; and
- member order does not change correctly aligned aggregate output.

Definition of done:

- every member and final aggregate match the original estimator on frozen data.

## P1 — full benchmark gates

### GAP-12 — AutoGluon/TabArena adapter

Current gap:

- low-level `fit` returns `None` and `predict` returns a device tensor;
- no AutoGluon model metadata or resource lifecycle exists.

Required behavior:

- importable module-level adapter with unique `ag_key` and `ag_name`;
- binary, multiclass, and regression support;
- `_fit`, `_predict`, and `_predict_proba`;
- explicit problem type and class order;
- fixed resource/checkpoint/seed configuration;
- metadata and cleanup; and
- CPU conversion only at the framework boundary.

Dependency rule:

- keep AutoGluon and TabArena optional. Importing core `sdm` must not require
  benchmark packages.

Required tests:

- tiny task for every problem type;
- module import in a fresh worker process;
- predict-before-fit;
- repeated fit and cleanup;
- correct probability and prediction shapes; and
- missing optional dependency produces a clear actionable error.

Definition of done:

- the public TabArena experiment flow evaluates the adapter without modifying
  TabArena.

### GAP-13 — Prediction chunking and memory controls

Current gap:

- no prediction batch-size or member batch-size control;
- cached repeated chunk prediction is not tested;
- cache movement after model/device changes is unsafe.

Required behavior:

- explicit member batch size and test-row prediction chunk size;
- fit/cache training context once, then process ordered test chunks;
- concatenate outputs without row-order changes;
- reject unsafe device moves after cache creation or move cache explicitly;
- clean cache after a job.

Required tests:

- chunk sizes 1, uneven, and full;
- one-shot versus chunked equality;
- binary, multiclass, and regression;
- cache cleanup and repeated use; and
- memory-oriented CUDA smoke test.

Definition of done:

- chunk size changes resource use but not predictions beyond declared numeric
  tolerance.

### GAP-14 — Controlled timing instrumentation

Current gap:

- example code demonstrates autocast but not a repeatable timing protocol;
- GPU phase boundaries are not explicitly synchronized;
- checkpoint load, preprocessing, cache construction, and prediction are not
  separated.

Required behavior:

- preload/download outside timing;
- synchronize before and after GPU regions;
- separate cold initialization, preprocessing, fit/cache build, first predict,
  and warm predict;
- perform warm-ups and repeated measurements;
- record test rows, dtype, cache mode, member count, hardware, and memory;
- publish raw timing samples.

Required tests:

- unit/fake-clock phase accounting;
- CUDA smoke for synchronization path;
- assertion that download time is excluded; and
- original/SDM runs use identical settings.

Definition of done:

- same-hardware performance equivalence can be evaluated automatically using
  predeclared ratio margins and bootstrap intervals.

### GAP-15 — Prediction/result artifacts and comparator

Current gap:

- TabArena outer results discard predictions after metric calculation;
- no SDM-owned per-row parity artifact exists;
- no strict paired-coverage validator exists.

Required behavior:

- save predictions or stable prediction hashes/deltas for selected parity runs;
- save per-split metrics, phase times, memory, and effective configuration;
- join candidate/reference strictly by dataset and split;
- fail on missing, duplicate, failed, or imputed rows;
- aggregate splits within datasets before overall statistics;
- retain per-task diagnostics for failed gates.

Required tests:

- artifact schema validation;
- missing/duplicate row failures;
- class-column alignment;
- scale-aware regression comparison; and
- deterministic analysis output.

Definition of done:

- every metric/timing discrepancy can be traced to row-level or task-level
  evidence.

### GAP-16 — Optional dependency and environment setup

Current gap:

- [`pyproject.toml`](../../pyproject.toml) has no benchmark extra;
- core dependencies are intentionally light and currently unpinned;
- the two repositories have separate development environments.

Required behavior:

- keep core dependencies unchanged;
- add either a clean optional benchmark extra or a dedicated reproducible
  environment/lock file;
- install editable SDM and pinned TabArena together;
- freeze original `tabicl`, AutoGluon, PyTorch, and related versions;
- document the command and package-collision checks.

Required tests:

- core import without extras;
- guarded integration import without extras;
- shared benchmark environment smoke; and
- version/provenance manifest generation.

Definition of done:

- one documented setup produces a compatible, frozen environment without
  making heavy packages mandatory for SDM users.

### GAP-17 — Resource cleanup and run metadata

Current gap:

- no benchmark adapter owns lifecycle cleanup;
- effective model parameters are not emitted from SDM;
- both checkpoint variants may remain resident unnecessarily.

Required behavior:

- clear SDM caches after each job;
- release model references and optionally empty allocator caches at controlled
  boundaries;
- report checkpoint/task/model residency;
- report effective seed, preprocessing, precision, batch, and cache options;
- make repeated jobs stable.

Required tests:

- repeated job execution;
- cleanup after success and failure;
- metadata schema; and
- no stale fitted state across datasets.

Definition of done:

- each job starts from declared state and leaves no model-specific cache needed
  by a later job.

### GAP-18 — Documentation and API consistency

Current gap:

- [`README.md`](../../README.md) shows `recipe=` and `num_estimators=` arguments
  not accepted by the current model;
- its `TableTensor.from_pandas` example does not match the actual signature;
- recipe comments incorrectly classify some existing primitives as missing.

Required behavior:

- decide whether the high-level predictor becomes the documented API or the
  README is reduced to the actual low-level API;
- make every published example executable in CI;
- distinguish implemented, adapter-only, experimental, and future behavior;
- link the benchmark runbook and evidence after validation.

Definition of done:

- documentation and executable interfaces agree;
- no example advertises unsupported arguments.

## Later — reusable productization opportunities

These items should not block first parity unless a design review explicitly
promotes them.

| ID | Feature | Why it can wait | Definition of done |
|---|---|---|---|
| `LATER-01` | Generic deterministic `Choice` processor | Predictor can select independent member pipelines explicitly | Reusable outside TabICLv2 with explicit member/seed semantics |
| `LATER-02` | Generic `TaskDispatch` processor/recipe | Predictor already knows task; separate recipes are clearer initially | Task-keyed fit/transform contract covered for both tasks |
| `LATER-03` | Unified task-aware TabICLv2 recipe | Avoid freezing unverified reference behavior into public API | Public recipe exactly matches validated predictor behavior |
| `LATER-04` | Schema-changing processors in `Pipeline` | Predictor-local mask solves constant filtering with lower risk | Column removal/reorder preserves correct names and semantic types |
| `LATER-05` | Tensor-aware model-output pipeline | Predictor can decode raw logits/quantiles directly | Raw distribution/classification outputs no longer need fake table schemas |
| `LATER-06` | sklearn `TabICLv2Classifier`/`Regressor` | TabArena adapter is the immediate consumer; sklearn adds maintenance/dependencies | Relevant sklearn estimator checks pass |
| `LATER-07` | High-level options on `BaseModel` | Core should remain thin until reusable semantics are proven | `recipe`, members, task, seed, and batching have a coherent public contract |
| `LATER-08` | Task-only/lazy checkpoint residency | Valuable for memory, not required for initial numerical parity | Only requested variant is instantiated and loaded |
| `LATER-09` | Compile/dtype/determinism controls | Adapter can manage initial autocast/timing settings | Public runtime controls are documented and tested on CPU/CUDA |
| `LATER-10` | Robust and RTDL quantile normalizers | Default benchmark uses `none`/`power`; do not add unused behavior speculatively | Implemented only when a validated config requires them |

## Required test matrix

| Layer | Binary | Multiclass | Regression | Missing/categorical | CPU FP32 | CUDA FP32 | CUDA BF16 | Original oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint/remap | yes | yes | yes | n/a | yes | yes | optional | yes |
| Raw network | yes | yes | yes | preprocessed only | yes | yes | yes | yes |
| Preprocessing | yes | yes | yes | yes | yes | yes | as used | yes |
| Output decoding | yes | yes | yes | labels/target | yes | yes | yes | yes |
| Ensemble | yes | yes | yes | yes | yes | yes | yes | yes |
| Adapter | yes | yes | yes | yes | yes | yes | yes | paired |
| Chunking/cache | yes | yes | yes | representative | yes | yes | yes | paired |
| Three-task smoke | yes | yes | yes | anneal covers categoricals | optional | required | configured | paired |
| 36-task Track A | 21 tasks | 7 tasks | 8 tasks | suite coverage | no | required | configured | paired |
| Historical Track B | 30 datasets | 8 datasets | 13 datasets | suite coverage | no | required H200 target | historical | archived results |

## Recommended implementation order

1. Freeze the original implementation, checkpoints, and manifests (`GAP-01`).
2. Add real-checkpoint raw network parity tests (`GAP-02`).
3. Add an explicit task/checkpoint boundary (`GAP-03`).
4. Implement exact single-estimator dataframe preprocessing (`GAP-04` through
   `GAP-06`).
5. Implement feature/class permutations and task output decoders (`GAP-07`
   through `GAP-10`).
6. Wire existing normalization primitives and implement ensemble orchestration
   (`GAP-11`).
7. Add the optional AutoGluon adapter and its contract tests (`GAP-12`).
8. Add chunking, timing, artifacts, and comparator (`GAP-13` through `GAP-17`).
9. Run Track A smoke and 36-task gates.
10. Run Track B only after Track A passes.
11. Promote validated behavior into generic/public APIs only where it is
    demonstrably reusable.
12. Correct public documentation (`GAP-18`) before release.

## Feature-gate checklist

An agent may call the implementation “TabArena parity ready” only when all of
the following are true:

- [ ] frozen original implementation and checkpoint hashes are recorded;
- [ ] raw classifier logits match the oracle;
- [ ] raw regression quantiles match the oracle;
- [ ] direct and cached raw inference both pass;
- [ ] dataframe and categorical encodings match;
- [ ] constant filtering and fixed clipping match;
- [ ] feature/member permutations match;
- [ ] class encoding, shifts, and inverse column alignment match;
- [ ] classification probabilities match;
- [ ] regression point predictions and target inverse match;
- [ ] one-member and eight-member aggregates match;
- [ ] adapter tests pass for all three problem types;
- [ ] chunking does not change predictions;
- [ ] timing excludes downloads and uses synchronized repeated samples;
- [ ] result coverage checks reject missing/imputed rows;
- [ ] benchmark dependencies remain optional; and
- [ ] documentation matches the implemented API.

## Agent implementation note template

Use this block in a pull request, issue, or handoff for each gap:

```markdown
Gap ID:
Status:
Implementation location:
Reference behavior/source:
Files changed:
Tests added:
Commands and results:
Numerical tolerances:
Evidence artifacts:
Known deviations:
Follow-up gaps unblocked:
```
