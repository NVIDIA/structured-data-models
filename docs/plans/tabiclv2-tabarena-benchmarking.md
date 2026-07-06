# TabICLv2 TabArena benchmarking plan

## Document metadata

| Field | Value |
|---|---|
| Status | Draft — implementation has not started |
| Working branch | `tabicl-benchmarking` |
| Primary repository | `structured-data-models` |
| Reference repository | sibling `tabarena` checkout; treat as read-only |
| SDM baseline commit | `1c8e017cabfbeda12b599100fcbfc663a1b4529d` |
| TabArena reference commit | `6624d55c1902a24da83ca779dcaf3388cb4b1666` |
| Last verified | 2026-07-06 |
| Companion document | [TabICLv2 SDM feature gaps](tabiclv2-sdm-feature-gaps.md) |

This is the source of truth for planning, executing, and reviewing the
TabICLv2-on-TabArena evaluation pipeline owned by this repository. It is written
for multiple contributors or coding agents working in sequence.

## How agents should use this plan

1. Read the whole document, the companion gap document, and `AGENTS.md`.
2. Claim one dependency-ready work item from the status table.
3. Do not modify the sibling TabArena repository.
4. Do not reuse a result cache unless its code, configuration, environment, and
   checkpoint fingerprints match the current run.
5. Run the work item's acceptance checks before marking it done.
6. Update the status table and append an agent handoff entry before stopping.
7. Preserve failed-job manifests and diagnostic artifacts. Never hide a failure
   through result imputation or by silently relaxing a tolerance.

## Goal

Build an evaluation pipeline, contained in `structured-data-models`, that:

- imports TabArena as an external benchmark dependency;
- benchmarks the SDM implementation of TabICLv2 on the canonical TabArena task
  splits;
- compares it with the original `tabicl` implementation under controlled,
  paired conditions;
- checks prediction, metric, inference-time, and memory equivalence;
- compares SDM results with the archived official TabArena result set; and
- records enough provenance for another machine or agent to reproduce each
  result.

## Non-goals

- Do not use leaderboard rank or Elo as the primary correctness test.
- Do not claim historical timing equivalence without recovering the original
  hardware and software environment.
- Do not make TabArena, AutoGluon, pandas, or sklearn mandatory core SDM
  dependencies.
- Do not commit checkpoints, OpenML caches, raw result pickles, or other large
  generated artifacts.
- Do not optimize the SDM model before numerical and behavioral parity is
  established.
- Do not redesign the generic SDM model API merely to satisfy the first
  benchmark integration. Adapter-local behavior is acceptable where it avoids
  premature core abstractions.

## Current state

### SDM implementation

The current low-level model lives in
[`sdm/models/tabiclv2/model.py`](../../sdm/models/tabiclv2/model.py). It:

- is a PyTorch module rather than an sklearn/AutoGluon estimator;
- accepts only `pretrained` and `device`;
- loads both classifier and regressor checkpoints;
- dispatches classification versus regression from the target dtype;
- returns raw 10-class logits or 999 regression quantiles; and
- provides a cache-backed `fit`/`predict` path through
  [`BaseModel`](../../sdm/models/base.py).

The active
[`default_regression_recipe`](../../sdm/models/tabiclv2/recipe.py) implements
only part of the original preprocessing behavior. Recipes are not called by the
model's `forward`, `fit`, or `predict` methods.

The existing model tests use `pretrained=False` and do not compare the loaded
network against the original implementation.

### Reference scripts

The sibling TabArena repository contains two useful patterns:

- `examples/advanced/run_quickstart_tabarena_model_without_bagging.py`
  demonstrates fixed model configurations and the outer/full-data protocol.
- `examples/advanced/run_tabarena_tabpfn3_custom_checkpoint.py` demonstrates
  per-problem-type checkpoint configuration.

Both follow this control flow:

```text
ConfigGenerator
  -> TabArena experiment bundle
  -> model x task jobs
  -> cached per-job results
  -> in-memory registration
  -> comparison with official results
  -> CSV leaderboard and plots
```

The SDM runner should use this public flow, not private modifications to
TabArena.

## Reference benchmark facts

### Archived official baseline

| Property | Value |
|---|---|
| Method | `TabICLv2` |
| Suite | `tabarena-2026-02-16` |
| Default config | `TabICLv2_c1_BAG_L1` |
| Comparison name | `TABICLV2 (default)` |
| Classifier checkpoint | `tabicl-classifier-v2-20260212.ckpt` |
| Regressor checkpoint | `tabicl-regressor-v2-20260212.ckpt` |
| Result rows | 816 |
| Datasets | 51 |
| Task mix | 30 binary, 8 multiclass, 13 regression |
| Outer folds | 3 per dataset |
| Repeats | 34 datasets use 3; 17 datasets use 10 |
| Metrics | binary ROC-AUC error, multiclass log loss, regression RMSE |

The official execution protocol used eight inner folds, one bag set,
sequential fold fitting, a full-data refit, seed 0, and one refit child for test
inference. Training time therefore includes the inner folds and refit; it is not
comparable with an outer/no-bagging quickstart fit.

The historical log identifies an H200 partition and eight CPUs. It does not
preserve the exact run commit, `tabicl` package version, Python/PyTorch/CUDA
stack, checkpoint hashes, CPU model, GPU SKU, clocks, concurrency, or repeated
timing samples. Archived historical times are descriptive until those inputs
are recovered.

### Development quickstart scope

The reference no-bagging quickstart uses `subset=["small", "lite"]`. This is an
intersection of training rows at most 10,000 with split zero, producing 36 jobs
per configuration:

- 21 binary datasets;
- 7 multiclass datasets; and
- 8 regression datasets.

The current script runs a default eight-estimator configuration and an
`n_estimators=1` diagnostic configuration. This is the right development scope,
but it is not the historical bag/refit protocol.

## Claims and evaluation tracks

Each claim must be reported independently. Passing one level does not imply the
next level passes.

### Claim 1 — numerical implementation parity

The original and SDM networks, using identical tensors, weights, dtype, device,
and inference mode, produce equivalent logits and quantiles.

### Claim 2 — high-level prediction parity

The original and SDM high-level predictors use equivalent preprocessing,
ensembling, label/feature permutations, and output decoding, and produce
equivalent probabilities or regression predictions.

### Claim 3 — TabArena metric parity

SDM reproduces paired per-split TabArena metrics and dataset-level aggregate
results within predeclared equivalence margins.

### Claim 4 — controlled performance parity

The original and SDM implementations have equivalent inference time under
co-located, repeated, synchronized measurements on the same hardware and
frozen environment.

### Claim 5 — historical benchmark reproduction

SDM reproduces the archived official protocol and scores. Archived timing is
informational unless the historical environment is recovered.

### Track A — implementation parity under the outer protocol

Run original `tabicl` and SDM TabICLv2 as full-data outer models under identical
current conditions. Track A is authoritative for implementation correctness and
same-machine inference speed.

Stages:

1. three-dataset smoke;
2. 36-task `small + lite` run;
3. optional full 816-split outer run.

### Track B — historical protocol reproduction

Run SDM through the explicit eight-fold bag/refit protocol and compare directly
with the archived config:

```python
reference = context.load_model_results(
    methods=["TabICLv2"],
    configs=["TabICLv2_c1_BAG_L1"],
)
```

Stages:

1. three-dataset bag/refit smoke;
2. 36-task `small + lite` bag/refit run;
3. full 816-split run only after a compute/storage estimate and explicit
   approval.

## Proposed implementation architecture

```text
TabArena task and metric protocol
  -> SDMTabICLv2Model              optional AutoGluon adapter
  -> TabICLv2Predictor             preprocessing, members, output decoding
  -> TabICLv2                      existing tensor-native neural network
  -> prediction capture/comparator
  -> TabArena metric and reporting
```

Proposed files:

```text
sdm/models/tabiclv2/predictor.py
sdm/integrations/tabarena/__init__.py
sdm/integrations/tabarena/tabiclv2.py
scripts/benchmarking/run_tabiclv2_tabarena.py
test/models/test_tabiclv2_reference_parity.py
test/models/test_tabiclv2_predictor.py
test/integrations/test_tabarena_tabiclv2.py
```

Design constraints:

- Keep the model and predictor tensor/PyTorch-centric.
- Keep AutoGluon and TabArena imports optional and isolated in the integration
  layer.
- The adapter must be defined in an importable module, not in `__main__`, so
  Ray or other workers can deserialize it.
- Fit every preprocessing state only on training/context rows.
- Expose all stochastic behavior through an explicit seed or generator.
- Preserve device and dtype inside the model path; transfer to CPU only at the
  TabArena boundary.
- Do not import or call the original implementation from the SDM candidate
  path. It may be used only as the paired reference/oracle.

## Dependency graph

```text
DEC-01 -> ENV-01 -> REF-01 -> CORE-01
             |                  |
             |                  v
             +--------------> PRE-01 -> ENS-01 -> ADP-01 -> TST-01
                                                        |
                                                        v
                                                     PIPE-01
                                                        |
                                                        v
                                                     JOB-01
                                                      /    \
                                           RUN-A0 -> RUN-A1  RUN-B0 -> RUN-B1 -> RUN-B2
                                                |               |                 |
                                                v               +-------> ANA-B1 <-+
                                             ANA-A1
                                                \                 /
                                                 +----> DOC-01 <-+
                                                          |
                                                          v
                                                        CI-01
```

## Work-item status

Use only these statuses: `Not started`, `Ready`, `In progress`, `Blocked`, and
`Done`.

| ID | Work item | Status | Depends on | Required output |
|---|---|---|---|---|
| `DEC-01` | Resolve open integration decisions | Ready | — | Updated decision log |
| `ENV-01` | Create and pin the shared benchmark environment | Blocked | `DEC-01` | Lock/freeze plus environment manifest |
| `REF-01` | Freeze official result and checkpoint references | Blocked | `ENV-01` | Checksummed reference artifacts |
| `CORE-01` | Prove raw network/checkpoint parity | Blocked | `REF-01` | Logit/quantile parity tests |
| `PRE-01` | Implement exact reference preprocessing | Blocked | `CORE-01` | Golden intermediate-value tests |
| `ENS-01` | Implement reference member generation and aggregation | Blocked | `PRE-01` | Deterministic ensemble parity tests |
| `ADP-01` | Implement the AutoGluon/TabArena adapter | Blocked | `ENS-01` | Importable three-task adapter |
| `TST-01` | Add adapter and end-to-end contract tests | Blocked | `ADP-01` | Passing binary/multiclass/regression tests |
| `PIPE-01` | Build the repository-owned benchmark CLI | Blocked | `REF-01`, `TST-01` | Preflight/plan/run/analyze CLI |
| `JOB-01` | Validate task and job manifests | Blocked | `PIPE-01` | Exact 36-task manifest with no skips |
| `RUN-A0` | Run three-dataset outer smoke | Blocked | `JOB-01` | Paired predictions, metrics, timings |
| `RUN-A1` | Run 36-task outer comparison | Blocked | `RUN-A0` | Complete Track A artifacts |
| `ANA-A1` | Analyze Track A equivalence | Blocked | `RUN-A1` | Machine-readable summary and report |
| `RUN-B0` | Run three-dataset historical-protocol smoke | Blocked | `JOB-01` | Verified bag/refit protocol and results |
| `RUN-B1` | Run 36-task historical-protocol comparison | Blocked | `RUN-B0` | Complete paired lite results |
| `RUN-B2` | Run full 816-split historical protocol | Blocked | `RUN-B1`, cost approval | Complete full-suite results |
| `ANA-B1` | Analyze historical score reproduction | Blocked | `RUN-B1` or `RUN-B2` | Dataset-weighted reproduction report |
| `DOC-01` | Publish runbook and final evidence | Blocked | `ANA-A1`, `ANA-B1` | User-facing reproducibility guide |
| `CI-01` | Add fast non-network regression gates | Blocked | `DOC-01` | Stable CI checks |

## Detailed work items

### DEC-01 — Resolve open integration decisions

Status: Ready

Decide and record:

- whether `TabICLv2Predictor` is public or initially private;
- whether the optional integration lives below `sdm/integrations` or in an
  installable benchmark-only package;
- benchmark optional-extra versus a dedicated environment lock;
- the canonical output root and retention policy;
- whether task-specific checkpoint selection belongs in the core constructor or
  remains adapter-managed for the first implementation; and
- initial numerical and statistical equivalence margins.

Acceptance gate:

- every decision has an owner, rationale, and status in the decision log;
- no unresolved decision blocks `ENV-01` or `CORE-01`.

### ENV-01 — Pin a shared environment

Status: Blocked on `DEC-01`

Create one environment containing editable SDM and the pinned sibling TabArena
checkout. Capture:

- repository SHAs and dirty-state patches;
- Python and full package freeze;
- AutoGluon, TabArena, original `tabicl`, PyTorch, CUDA, cuDNN, and driver
  versions;
- GPU name, UUID, memory, clocks/power information when available;
- CPU model, physical/logical core count, RAM, and thread environment; and
- effective cache paths.

The core SDM installation must remain usable without benchmark dependencies.

Acceptance gate:

- both implementations import from the intended local/package paths;
- no top-level package collision silently redirects the SDM candidate path;
- the environment can be recreated from committed inputs.

### REF-01 — Freeze references

Status: Blocked on `ENV-01`

Freeze and checksum:

- classifier and regressor checkpoint blobs;
- Hugging Face repository revision;
- official `model_results` for `TabICLv2_c1_BAG_L1`;
- canonical TabArena-v0.1 task grid;
- historical processed prediction artifacts if available; and
- original implementation package source/version used for Track A.

If the exact historical package cannot be recovered, evaluate likely historical
versions separately and label the uncertainty. Never silently use a moving
latest release.

Acceptance gate:

- all references have hashes and provenance in `manifest.json`;
- timed runs perform no downloads.

### CORE-01 — Prove raw network parity

Status: Blocked on `REF-01`

Compare original and SDM models before dataframe preprocessing or ensembling:

- binary and multiclass logits;
- regression quantiles;
- direct/joint forward;
- cached fit/predict forward;
- CPU FP32 first, then CUDA FP32, then the intended AMP/BF16 mode; and
- intermediate row-embedding and ICL activations if final outputs differ.

Acceptance gate:

- committed FP32 tolerances pass for all fixtures;
- lower-precision tolerances are declared separately;
- strict checkpoint remapping/loading is tested;
- a parity failure blocks all later benchmark runs.

### PRE-01 — Implement reference preprocessing

Status: Blocked on `CORE-01`

Implement the exact operation order and train-only fit scope described in the
companion gap document. Required behavior includes dataframe type handling,
categorical encoding, numerical imputation, constant-column filtering, scaling,
fixed clipping, member normalization, and sigma clipping.

Acceptance gate:

- original and SDM intermediate tensors match after every active step;
- tests cover missing, categorical, constant, extreme, and unseen-category
  values;
- no test-row value can change fitted preprocessing state.

### ENS-01 — Implement member generation and output aggregation

Status: Blocked on `PRE-01`

Implement:

- the default eight members and one-member diagnostic;
- normalization schedule;
- Latin feature permutations;
- class shifts and inverse probability-column alignment;
- classifier logit averaging followed by temperature `0.9` softmax;
- reference-compatible regression quantile-to-point prediction and inverse
  target scaling;
- deterministic seed 0 behavior; and
- optional chunking without prediction changes.

Acceptance gate:

- one member matches the single-estimator reference path;
- each member's transformed data and output matches the oracle;
- member aggregation matches original high-level predictions.

### ADP-01 — Implement the TabArena adapter

Status: Blocked on `ENS-01`

The adapter must provide:

- unique `ag_key` and `ag_name`;
- binary, multiclass, and regression support;
- `_fit`, `_predict`, and `_predict_proba`;
- explicit task instead of target-dtype inference;
- class-order preservation;
- CPU/GPU resource handling;
- `n_jobs`, batch-size, precision, checkpoint, seed, and cache-mode controls;
- cleanup of caches and GPU memory; and
- metadata containing all effective model parameters.

Acceptance gate:

- module-level import succeeds in a worker process;
- outputs have correct shapes, class order, and finite values;
- probabilities sum to one within tolerance;
- fixed-seed runs are deterministic;
- no core import requires AutoGluon unless the integration is requested.

### TST-01 — Add integration contract tests

Status: Blocked on `ADP-01`

Add fast tests for:

- binary classification;
- multiclass classification;
- regression;
- mixed pandas feature types;
- unseen test categories;
- predict-before-fit failure;
- repeated fit and cleanup;
- one-shot versus chunked prediction; and
- joint versus cache-backed mode, where expected.

Acceptance gate:

- tests run without network access using small fixtures;
- GPU-only tests are marked and remain separately runnable;
- core tests still pass without benchmark extras.

### PIPE-01 — Build the benchmark CLI

Status: Blocked on `REF-01` and `TST-01`

The CLI must support four operations:

- `preflight`: validate imports, environment, references, caches, and hardware;
- `plan`: materialize and save the exact job/config manifest without running;
- `run`: execute selected track, implementation, dataset scope, and resources;
- `analyze`: validate coverage and generate paired comparisons.

The CLI must expose at least:

```text
--track outer|official-protocol
--implementations original,sdm
--datasets ...
--subset ...
--n-estimators 1|8
--precision fp32|bf16
--inference-mode joint|kv
--num-cpus
--num-gpus
--output-root
--resume
--cache-mode use|ignore|refresh
--save-predictions
```

Acceptance gate:

- every command writes or consumes a manifest;
- result directories are fingerprinted and immutable;
- resuming cannot load a result created by different code/config/checkpoints.

### JOB-01 — Validate job manifests

Status: Blocked on `PIPE-01`

Acceptance gate for `small + lite`:

- exactly 36 datasets and 36 jobs per implementation/config;
- expected 21/7/8 binary/multiclass/regression mix;
- no model-constraint skips;
- no duplicate cache keys;
- resources and checkpoints are explicit; and
- the job manifest is saved before execution.

### RUN-A0 / RUN-A1 — Track A runs

Status: Blocked on `JOB-01`

Smoke datasets:

- `blood-transfusion-service-center` — binary;
- `anneal` — multiclass with categorical features;
- `QSAR_fish_toxicity` — regression.

Run the original and SDM implementations under identical resources. Capture
row-level prediction comparisons outside TabArena's default outer result pickle,
because outer results discard predictions after scoring.

Acceptance gate:

- all tasks pair exactly;
- no fill/imputation of missing result rows;
- failures produce diagnostic artifacts;
- declared prediction and metric margins pass before scaling to 36 tasks.

### ANA-A1 — Analyze Track A

Status: Blocked on `RUN-A1`

Report:

- probability/logit or regression-prediction deltas;
- per-split metric deltas;
- class agreement;
- fit, first-predict, warm-predict, total-time, and memory ratios;
- geometric mean, median, p90, and outlier tasks; and
- dataset-cluster bootstrap confidence intervals.

The provisional timing equivalence interval is a 95% confidence interval for
the geometric-mean SDM/original ratio contained in `[0.90, 1.10]`. Confirm or
replace this margin during `DEC-01` based on measured harness noise.

### RUN-B0 / RUN-B1 / RUN-B2 — Track B runs

Status: Blocked on `JOB-01`

The generated protocol must explicitly confirm:

- eight folds;
- one bag set;
- sequential fold fitting;
- full-data refit;
- seed 0 and the historical fold-seed policy;
- matching preprocessing;
- fixed checkpoints; and
- fixed CPU/GPU resources.

`RUN-B2` requires a compute/storage estimate and approval before launching.

### ANA-B1 — Analyze historical reproduction

Status: Blocked on Track B results

Pair raw model results by dataset and split. Aggregate splits within a dataset
before aggregating across the 51 datasets so datasets with 30 splits do not
dominate those with nine. Use dataset-cluster bootstrap intervals and
scale-aware/normalized errors; do not use a universal raw RMSE margin.

Historical timing must be labeled informational unless the original environment
is recovered.

## Intended command interface

These commands describe the target CLI; they are not expected to work until
`PIPE-01` is complete.

```bash
python scripts/benchmarking/run_tabiclv2_tabarena.py preflight \
  --output-root outputs/tabiclv2-tabarena/<run-id>

python scripts/benchmarking/run_tabiclv2_tabarena.py plan \
  --subset small lite \
  --output-root outputs/tabiclv2-tabarena/<run-id>

python scripts/benchmarking/run_tabiclv2_tabarena.py run \
  --track outer \
  --implementations original sdm \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity \
  --num-gpus 1 \
  --num-cpus <N> \
  --cache-mode ignore \
  --output-root outputs/tabiclv2-tabarena/<run-id>

python scripts/benchmarking/run_tabiclv2_tabarena.py run \
  --track outer \
  --implementations original sdm \
  --subset small lite \
  --num-gpus 1 \
  --num-cpus <N> \
  --output-root outputs/tabiclv2-tabarena/<run-id>

python scripts/benchmarking/run_tabiclv2_tabarena.py run \
  --track official-protocol \
  --implementations sdm \
  --subset small lite \
  --output-root outputs/tabiclv2-tabarena/<run-id>

python scripts/benchmarking/run_tabiclv2_tabarena.py analyze \
  --run-dir outputs/tabiclv2-tabarena/<run-id>
```

## Artifact layout

```text
outputs/tabiclv2-tabarena/<run-id>/
├── manifest.json
├── environment.json
├── hardware.json
├── checkpoint_checksums.json
├── experiment_configs/
│   ├── track_a_original.yaml
│   ├── track_a_sdm.yaml
│   └── track_b_sdm.yaml
├── jobs/
│   ├── track_a_jobs.csv
│   └── track_b_jobs.csv
├── cache/
│   ├── track_a_original/data/...
│   ├── track_a_sdm/data/...
│   └── track_b_sdm/data/...
├── reference/
│   ├── model_results.parquet
│   └── checksum.sha256
├── track_a/
│   ├── raw_results.parquet
│   ├── prediction_checks.parquet
│   ├── timing_samples.parquet
│   ├── paired_results.parquet
│   └── summary.json
├── track_b/
│   ├── raw_results.parquet
│   ├── paired_results.parquet
│   └── summary.json
├── eval/
└── logs/
```

`<run-id>` must include code, configuration, environment, and checkpoint
fingerprints. Generated heavy output roots must be gitignored. Commit only
compact manifests and summaries when they are useful review evidence.

## Correctness acceptance rules

- Compare paired rows only; missing rows are failures, not values to impute.
- Set `fillna_method=None` for reproduction analysis.
- Validate exact dataset/split coverage before computing aggregates.
- Declare tolerances before looking at candidate results.
- Keep FP32 and reduced-precision tolerances separate.
- Normalize regression discrepancies by a declared scale; do not compare raw
  RMSE deltas across differently scaled targets.
- Aggregate repeated splits within each dataset before global aggregation.
- Report each problem type separately as well as overall.
- Preserve task-level diagnostics for every failed gate.
- Use leaderboard output only as a secondary presentation artifact.

## Timing protocol

TabArena's default timer measures one wall-clock predict call, with no explicit
warm-up or repeated trials. GPU work can be attributed to the wrong phase if an
implementation returns before synchronization. The controlled timing harness
must therefore:

1. run original and SDM on the same machine and environment;
2. pin CPU/GPU counts, thread settings, precision, batch size, and cache mode;
3. preload checkpoints and datasets outside measured regions;
4. randomize or interleave implementation order;
5. perform at least two warm-ups and five measured predictions per task;
6. call `torch.cuda.synchronize()` before and after GPU timing boundaries;
7. separate model initialization, preprocessing, cache construction/fit, first
   prediction, steady prediction, and total time;
8. record test-row count and report seconds per 1,000 rows;
9. record peak allocated/reserved GPU memory and CPU RSS; and
10. publish every timing sample, not only the median.

The current development machine is an NVIDIA L4; the archived run used an H200
partition. L4 measurements are a current same-machine comparison, not a
reproduction of archived H200 time.

## Risk register

| Risk | Effect | Mitigation |
|---|---|---|
| Unknown historical `tabicl` version | Defaults and timing may drift | Freeze Track A version; test likely historical releases separately |
| Missing historical environment | Absolute timing claim is invalid | Restrict hard timing gate to co-located Track A |
| Cache key lacks code/config hash | Stale results can appear valid | Fingerprinted immutable run roots; preflight validation |
| Outer results discard predictions | Metric parity is hard to diagnose | Add separate prediction capture/comparison artifacts |
| Candidate uses cache while reference does not | Timing phases are incomparable | Make `joint`/`kv` modes explicit and paired |
| Categorical SDM features are ignored | Incorrect metrics on mixed-type tasks | Add reference-compatible dataframe encoder before model |
| Preprocessing learns from test rows | Data leakage and optimistic scores | Golden train-only fit-scope tests |
| Full Track B is expensive | Wasted compute before correctness is known | Enforce smoke and 36-task gates plus cost approval |
| Core dependency growth | Violates repository design principles | Guard optional integration and keep core imports light |

## Decision log

Append decisions; do not rewrite history.

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-07-06 | Use separate Track A and Track B claims | Current outer runs are best for implementation/timing comparison; archived protocol is required for historical score reproduction | Accepted |
| 2026-07-06 | Treat sibling TabArena as read-only | The requested pipeline must be owned by SDM and survive TabArena updates through public APIs | Accepted |
| 2026-07-06 | Keep benchmark dependencies optional | Required by SDM's lightweight, tensor-centric design | Accepted |
| 2026-07-06 | Use adapter-local orchestration before generic `Choice`/`TaskDispatch` APIs | Establish correct behavior before committing to broader abstractions | Proposed |
| 2026-07-06 | Proposed CLI path is `scripts/benchmarking/run_tabiclv2_tabarena.py` | This is an operational evaluation pipeline rather than a minimal usage example | Proposed |

## Agent handoff template

Append one block per handoff.

```markdown
### Handoff YYYY-MM-DD — <work-item-id>

Work item:
Status:
Files changed:
Commands run and results:
Evidence/artifact paths:
Decisions made:
Open risks or blockers:
Next dependency-ready work items:
```

## Definition of complete

This plan is complete only when:

- the candidate path uses SDM TabICLv2 and does not call original model code;
- raw checkpoint/network parity passes;
- preprocessing and ensemble behavior have golden reference tests;
- the adapter supports binary, multiclass, and regression tasks;
- all requested jobs are present with no imputed result rows;
- Track A reports paired predictions, metrics, repeated synchronized timing, and
  memory;
- Track B reports paired official-protocol metrics at the approved scope;
- every run has a reproducible manifest and immutable fingerprint;
- benchmark dependencies remain optional; and
- documentation states precisely which of the five claims passed or failed.
