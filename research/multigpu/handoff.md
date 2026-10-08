# Local integration and access-boundary handoff

Initial audit: 2026-10-07 UTC / 2026-10-08 Asia/Seoul. **The overall goal is not complete.** Permitted networking has subsequently returned. The original instances and their task-tagged EBS volumes are no longer present; results retained only on their deleted disks are unrecoverable. Replacement execution is being arranged separately. Missing scientific controls, final resource closure and publication status remain explicit below.

## Delivered locally

- Integration worktree: `/Users/ardrianw/repositories/sdm-kumo-multigpu-20261008`.
- Branch: `research/kumo-multigpu-20261008`, based on `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`.
- Main report: [README.md](README.md); extraction recommendations: [integration.md](integration.md); independent audits: [quality.md](quality.md).
- Original user worktree and its unrelated changes were preserved. The first confirmed push published commit [`af55e434bbe9074aef51ade1de261e1661471e2c`](https://github.com/NVIDIA/structured-data-models/commit/af55e434bbe9074aef51ade1de261e1661471e2c) to [`origin/research/kumo-multigpu-20261008`](https://github.com/NVIDIA/structured-data-models/tree/research/kumo-multigpu-20261008). No PR has been opened. Later local commits are not assumed published without a subsequent push confirmation.
- Ten delegated workstreams covered implementation, workloads, execution, independent quality review, documentation, and cloud operations. GPU benchmarking skills were not used.

The core-source candidates are `sdm/models/ensemble_parallel.py` and its export, `sdm/nn/context_parallel.py`, the attention hook, and ICL integration in KumoTabular and TabICLv2 (also used by KumoRelational). Process/thread query DP, batched/process/graph EP, stage/layer placement, cache compaction, persistent autocast, profiling, and cluster runners remain research modules. These are experimental integration candidates, not a claim that the full research branch is production-ready.

Each benchmark result retains its own measured source revision and runner hash. The final local integration includes subsequent harness/tests/documentation fixes and must not be substituted for those measured revisions.

## Evidence verification

The coordinator independently reran `collect_evidence.verify(..., external=True)` over every `research/multigpu/evidence/*/index.json` after integration:

| Check | Result |
|---|---:|
| Evidence snapshots | 26 |
| Archived-file references checked | 277 |
| Original-artifact references checked | 629 |
| Missing/mismatched references | 0 |
| External references left unchecked | 0 |

Counts are references, not unique artifacts or independent trials. Large outputs remain under `/Users/ardrianw/repositories/.kumo-multigpu-20261008`; do not remove that local store without relocating and verifying them. Models and raw datasets are not committed. Exact verification commands are in the report.

After access resumed, the report owner independently reran external verification at integrated source `bd7e1ba40`: all 26 snapshots, 277 archived references and 629 original-artifact references passed again, with zero unchecked external references. This verifies the surviving local evidence, not any undownloaded remote result.

The original cuDF results saved repeat-zero arrays but scored the last repeat. Original evidence remains unchanged; additive independent audits score the archived arrays. The runner now archives every repeat and uses the same first repeat for NPY, PT, and reported reference quality. A deliberately nondeterministic CPU fixture tests this contract.

## Validation scope

Historical actual-GPU checks are recorded with their source in the method documents: core EP tests, 36 placement CPU/CUDA tests, and nine original CP Gloo/NCCL tests passed. Process-EP CUDA tests used a synthetic cache model; graph-EP CUDA tests used reduced KumoTabular. Neither establishes pretrained benchmark performance.

The ensemble owner's integrated local suite reported 110 passes and 17 hardware skips; two process reruns failed before model execution because the new sandbox denied `torch_shm_manager`. These are environment-blocked reruns, not passing tests and not evidence of a model regression. GPU tests cannot be rerun locally.

The former CPU spawn restriction is now cleared. At `bd7e1ba40`, the data-parallel owner reran the following command from the integration worktree: **42 passed, zero skipped, 12.51 seconds**, including actual CPU process spawning. This is CPU runtime evidence only, not a GPU rerun; do not add it to earlier overlapping suite counts.

```sh
PYTHONPATH=. /Users/ardrianw/repositories/sdm-kumotabular-stream-main-20261001/.venv/bin/python -m pytest \
  research/multigpu/test_query_parallel.py \
  research/multigpu/test_query_shards.py \
  research/multigpu/test_multihost_query_bench.py -q
```

After the repeat-provenance fix, the coordinator reran evidence-collector, relational-scoring/archive, and query-sharding tests: **32 passed**. Ruff passed on these changed files and the graph diagnostic. A separate multihost protocol run exposed a stale integer-column test fixture incompatible with SDM's string prediction-column contract; its correction is tracked separately rather than hidden as an environment failure.

The fixture correction is integrated as `a886c2223`. The coordinator reran protocol and relational scoring tests: **11 passed**, including binary/seven-class cases across 1/2/4 fake workers and reordered semantic classes. Ruff passed. These overlap four tests in the preceding count; counts must not be summed as unique tests. Fake-worker timings are not GPU benchmark results.

A subsequent research-only scorer fix (`f13e29828`) preserves the full prediction class support even when a validation subset omits classes. It rejects unknown targets and malformed probabilities rather than dropping mass, and reports undefined binary AUROC as null with a reason. The coordinator's combined class-support/protocol/scoring suite passed **22 tests**, with Ruff clean; an independent reviewer also checked randomized semantic-order/subset cases. Existing archived Covertype metrics reproduced without changing raw records. Semantically duplicate prediction names such as `1` and `01` remain unsupported malformed input, not a measured-model case.

## Lost remote evidence and remaining controls

| Item | Last known state | Required interpretation |
|---|---|---|
| H&M C16k/E8 native, EP1/2/4, hybrid queue on original L40S | Launched before access restriction; outputs were not downloaded | Original remote-only outputs lost with disk deletion; no completion, speed or quality claim |
| H&M C64k/E8 capacity on original L4 | EP1 stdout showed OOM; wrapper finished | Undownloaded stage2 outcome/raw receipts lost; visible EP1 failure is not a recovered complete comparison |
| Original placement Nsight captures | Captured remotely but not downloaded | Remote-only captures lost; no trace-derived conclusion |
| Native CPU-offload E8 capacity control | Never launched | No multi-GPU-only feasibility claim |
| Pretrained process-EP and graph-EP ladders | Never measured | Fixture correctness only |
| Mixed eight-GPU query DP | Staged, never launched | No eight-GPU scaling result |
| Full-model FP32 F1 CP, GPU all-gather, C32k MHA | Staged, never launched | CPU-only all-gather validation; diagnostics unmeasured |
| cuDF graph repeatability diagnostic | Prepared, never executed | Nondeterminism cause unresolved |

These limitations supplement the objective-by-objective completion checklist in README. No additional parallelism method needs to be invented to resolve the operational interruption.

Any replacement execution needs a new attempt directory, exact runtime/source receipt, and same-host native/resident controls. New measurements cannot be attributed to an original host, silently replace a lost attempt, or use the old L4 denominator for a new L40S result.

## Paper-informed follow-up

The user's subsequent literature request is documented in [literature.md](literature.md) and four linked primary-source reviews. Published measurements are kept separate from SDM results. The review prioritizes exact execution/caching, bounded upstream fit allocations, and only then approximate context selection or distillation.

The research-only [destination-block GNN](blocked-gnn.md) is integrated, with optional `--gnn-block-size` benchmark selection and independent reduction/cache/RNG/empty-segment review. Its target is the observed 6.10 GiB statistics allocation, not a new distributed-serving framework. The coordinator's combined synthetic/scoring/protocol suite passed 119 tests, and the evidence collector's four tests passed. The actual-checkpoint CPU screen was independently rerun: 29 passed, two FP32 singleton-block cases failed the unchanged gate. All 30 checkpoint-case receipts, including failures, are archived and verified in the new CPU-only evidence snapshot. Ruff and `git diff --check` passed. No CUDA, full-model quality, measured memory or performance claim is made for this prototype.

## Historical access interruption

At 16:30:55 UTC the session's network restrictions denied SSH to both hosts and AWS EC2 endpoint access. Approval policy is `never`; no alternative route or bypass was attempted. User authorization to use `al` does not remove this execution restriction. This is not an observed expired-login error.

Fresh read-only probes at 16:44:41–16:44:51 UTC produced the same SSH `Operation not permitted` and regional EC2 endpoint connection failures. No new instance/process state was obtained. Academic-paper browsing remains available through the separate web tool; that does not establish SSH or EC2 access.

The third consecutive goal-turn check at 16:57:56 UTC again denied both SSH hosts and both EC2 endpoints. The previous two turns made local integration, scoring, evidence and literature progress despite that restriction; this turn completed the bounded-GNN CPU follow-up and preserved its failures. Local follow-up work is now handed off. Remaining execution, remote evidence recovery, Git publication, cost reconciliation and teardown require restored permitted networking. The full goal remains unachieved; unsupported network bypasses and assumed job termination are not alternatives.

Both original task hosts were last verified reachable at 16:27:25 UTC, without a Spot interruption notice. At the blocked handoff their lifecycle was unknown and charges might have continued. Previously verified shutdown-to-terminate deadlines were 19:28:00 UTC for L4 and 23:15:34 UTC for L40S, on 2026-10-07. The subsequent read-only checks below supersede that uncertainty for the original instances; a timer alone was never accepted as termination evidence.

## Resumed lifecycle and execution

With permitted networking restored, the sole cloud operator's checks beginning **2026-10-08 01:34:46 UTC**, before replacement launch, found both original instance IDs absent and no remaining task-tagged EBS volumes. The retained L40S Spot request reports `instance-terminated-by-user` at 2026-10-07 23:23:31 UTC. The L4 request has already been purged, so its exact termination time and cause are unavailable. The local receipt is [old-host-lifecycle-resume-20261008.json](../../../.kumo-multigpu-20261008/ops/old-host-lifecycle-resume-20261008.json). These facts establish original-host closure and loss of undownloaded disk artifacts; they do not identify final cost or certify removal of every access resource.

The coordinator has authorized two replacement homogeneous L40S Spot hosts with three-hour shutdown guards. The operator reports first replacement `i-0fa5c18a88d2eb124` launched in us-east-2a at 2026-10-08 01:38:13 UTC; second-host capacity was still being sought at this update. Their subsequent runtime state and identifiers belong in the operator's live [operations record](operations.md); this handoff does not assume readiness or termination. Original security groups/imported keys are retained for reuse; stale cross-host ingress rule `sgr-028d1b8e6657ba968` was revoked. These access resources and replacements must be reconciled before final cleanup is declared. No original remote-only result is marked a numerical failure simply because it was lost.

The coordinator confirmed the branch's first push at `af55e434b`; the remote is `git@github.com:NVIDIA/structured-data-models.git`. The subsequent lifecycle follow-up integrated as `46f05c2cb` was still local at that confirmation. Publication does not close final resource teardown, actual cost accounting or remaining scientific controls.
