# TabICLv2 same-machine comparison runbook

This follow-on benchmark builds on the phase-one TabArena smoke adapter. It
measures original TabICLv2 and SDM TabICLv2 on the same local task split and
produces a report whose result classes cannot be mistaken for one another.

## What the report means

The report contains three separate tables:

1. **Matched parity — authoritative same-workload comparison.** Both
   implementations use one estimator, the same checkpoint and seed, equivalent
   deterministic preprocessing, caching, the same data split, and the same
   CPU/GPU allocation. Only this table contains speedup ratios.
2. **Local native defaults — same machine, different workloads.** Original
   TabICLv2 uses its native eight-estimator configuration; SDM uses its native
   one-estimator recipe. This answers what each implementation does out of the
   box on the current machine, but it is not a fair workload-parity ranking.
3. **Historical TabArena baseline — external context.** This is the archived
   TabArena default result for the selected dataset and split. It was not rerun
   on the current machine, so its timing is not compared with local timing.

A baseline-enriched report simply means that the archived TabArena row is
displayed next to the new local measurements. TabArena does not record the
local measurements. The runner may download TabArena's own cached result on a
cache miss, but it never uploads results or writes to the TabArena repository.

## Local timing protocol

Every implementation, profile, and trial runs in a fresh subprocess. The
parent alternates original-first and SDM-first execution to reduce order bias.
Data loading and split selection happen before timing.

Each child records:

- end-to-end model construction, checkpoint loading, preprocessing, and fit;
- the first prediction call;
- 20 subsequent warmed prediction calls by default; and
- RMSE on the canonical held-out split.

The runner uses time.perf_counter_ns and explicitly synchronizes CUDA before
and after every timed GPU region. One pair per profile is discarded as a
system warm-up, followed by 30 measured trials per implementation by default.
The summary reports medians and interquartile ranges across trial medians.

With the defaults, the run executes four discarded child processes and 120
measured child processes. The native-default profile is expected to take
longer because the original implementation runs eight estimators.

## Configuration contract

The matched original profile uses one estimator, no distribution
normalization, no feature shuffling, batch size one, KV caching, the
caller-supplied seed, and disabled automatic AMP, FlashAttention, and
offloading.

The matched SDM profile uses one estimator and an explicit benchmark recipe:
mean imputation, constant-column filtering, standard scaling with epsilon
1e-6, fixed clipping to [-100, 100], identity distribution normalization,
sigma clipping at four standard deviations, no feature permutation, and
target standard scaling.

The native original profile uses eight estimators, none/power normalization,
Latin feature shuffling, batch size eight, no KV cache, random state 42, and
automatic acceleration/offload selection. The native SDM profile uses the
existing SDM default recipe, one estimator, the caller-supplied seed, and
cache-backed inference.

All four local profiles pin the caller-supplied checkpoint path and SHA-256,
disable checkpoint downloads, and use the same requested machine resources.
The complete resolved configurations are written to the manifest.

## Running the comparison

Use the dedicated environment containing editable SDM, pinned TabArena,
AutoGluon, and original tabicl:

```console
python -m examples.benchmarking.run_tabiclv2_local_comparison \
    --checkpoint-path /path/to/tabicl-regressor-v2-20260212.ckpt \
    --checkpoint-sha256 0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a \
    --checkpoint-repository caller/supplied-source \
    --checkpoint-revision caller-supplied-revision \
    --output-dir /tmp/tabiclv2-local-comparison \
    --num-cpus 8 \
    --num-gpus 1
```

The default task is OpenML task 363698, QSAR_fish_toxicity, repeat/fold/sample
0/0/0. Task coordinates, warm-up count, trial count, inference repetitions,
seed, and worker timeout have explicit CLI overrides. Use a new or empty output
directory; the runner refuses to mix artifacts with an existing run.

## Generated artifacts

The runner creates local artifacts only:

| Artifact                        | Contents                                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| tabiclv2_local_trials.csv       | One measured row per local profile, implementation, and trial                                               |
| tabiclv2_local_inference.csv    | Every warmed inference-call duration                                                                        |
| tabiclv2_comparison_summary.csv | Local median/IQR rows plus one classified historical row                                                    |
| tabiclv2_comparison_report.md   | Three clearly separated human-readable result tables                                                        |
| manifest.json                   | Environment, commits, resources, checkpoint provenance, exact configurations, protocol, and artifact hashes |

Every CSV row identifies its comparison group, execution source, timing
comparability, and fair-speedup eligibility. Native and historical rows leave
all speedup fields empty. Generated results are machine-specific and must not
be committed to the repository.

## Validation

The automated tests verify exact matched/native configurations, recipe
construction, CUDA synchronization boundaries, alternating order, discarded
warm-ups, fresh-worker failure propagation, trial aggregation, matched-only
speedups, exact historical-row selection, report separation, manifest hashes,
and CSV row counts. The existing smoke-adapter tests remain the regression
gate for the original PR behavior.
