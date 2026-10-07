# Local integration and access-boundary handoff

Audit date: 2026-10-07 UTC / 2026-10-08 Asia/Seoul. **The overall goal is not complete:** cloud state, remaining remote evidence, and resource teardown cannot currently be verified.

## Delivered locally

- Integration worktree: `/Users/ardrianw/repositories/sdm-kumo-multigpu-20261008`.
- Branch: `research/kumo-multigpu-20261008`, based on `842c408fe2a8711bdf2e7cff4bfbe54266d6b940`.
- Main report: [README.md](README.md); extraction recommendations: [integration.md](integration.md); independent audits: [quality.md](quality.md).
- Original user worktree and its unrelated changes were preserved. This study branch has not been pushed and no PR has been opened.
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

The original cuDF results saved repeat-zero arrays but scored the last repeat. Original evidence remains unchanged; additive independent audits score the archived arrays. The runner now archives every repeat and uses the same first repeat for NPY, PT, and reported reference quality. A deliberately nondeterministic CPU fixture tests this contract.

## Validation scope

Historical actual-GPU checks are recorded with their source in the method documents: core EP tests, 36 placement CPU/CUDA tests, and nine original CP Gloo/NCCL tests passed. Process-EP CUDA tests used a synthetic cache model; graph-EP CUDA tests used reduced KumoTabular. Neither establishes pretrained benchmark performance.

The ensemble owner's integrated local suite reported 110 passes and 17 hardware skips; two process reruns failed before model execution because the new sandbox denied `torch_shm_manager`. These are environment-blocked reruns, not passing tests and not evidence of a model regression. GPU tests cannot be rerun locally.

After the repeat-provenance fix, the coordinator reran evidence-collector, relational-scoring/archive, and query-sharding tests: **32 passed**. Ruff passed on these changed files and the graph diagnostic. A separate multihost protocol run exposed a stale integer-column test fixture incompatible with SDM's string prediction-column contract; its correction is tracked separately rather than hidden as an environment failure.

The fixture correction is integrated as `a886c2223`. The coordinator reran protocol and relational scoring tests: **11 passed**, including binary/seven-class cases across 1/2/4 fake workers and reordered semantic classes. Ruff passed. These overlap four tests in the preceding count; counts must not be summed as unique tests. Fake-worker timings are not GPU benchmark results.

A subsequent research-only scorer fix (`f13e29828`) preserves the full prediction class support even when a validation subset omits classes. It rejects unknown targets and malformed probabilities rather than dropping mass, and reports undefined binary AUROC as null with a reason. The coordinator's combined class-support/protocol/scoring suite passed **22 tests**, with Ruff clean; an independent reviewer also checked randomized semantic-order/subset cases. Existing archived Covertype metrics reproduced without changing raw records. Semantically duplicate prediction names such as `1` and `01` remain unsupported malformed input, not a measured-model case.

## Remaining remote evidence and unrun work

| Item | Last known state | Required interpretation |
|---|---|---|
| H&M C16k/E8 native, EP1/2/4, hybrid queue on L40S | Launched before access restriction | Completion and outputs unobserved; do not restart blindly |
| H&M C64k/E8 capacity on L4 | EP1 stdout showed OOM; wrapper finished | Stage2 child outcome and complete raw receipts unverified |
| Placement Nsight captures | Captured remotely | Not downloaded/analyzed; no trace-derived conclusion |
| Native CPU-offload E8 capacity control | Never launched | No multi-GPU-only feasibility claim |
| Pretrained process-EP and graph-EP ladders | Never measured | Fixture correctness only |
| Mixed eight-GPU query DP | Staged, never launched | No eight-GPU scaling result |
| Full-model FP32 F1 CP, GPU all-gather, C32k MHA | Staged, never launched | CPU-only all-gather validation; diagnostics unmeasured |
| cuDF graph repeatability diagnostic | Prepared, never executed | Nondeterminism cause unresolved |

These limitations supplement the objective-by-objective completion checklist in README. No additional parallelism method needs to be invented to resolve the operational interruption.

## Paper-informed follow-up

The user's subsequent literature request is documented in [literature.md](literature.md) and four linked primary-source reviews. Published measurements are kept separate from SDM results. The review prioritizes exact execution/caching, bounded upstream fit allocations, and only then approximate context selection or distillation.

The research-only [destination-block GNN](blocked-gnn.md) is integrated, with optional `--gnn-block-size` benchmark selection and independent reduction/cache/RNG/empty-segment review. Its target is the observed 6.10 GiB statistics allocation, not a new distributed-serving framework. The coordinator's combined synthetic/scoring/protocol suite passed 119 tests, and the evidence collector's four tests passed. The actual-checkpoint CPU screen was independently rerun: 29 passed, two FP32 singleton-block cases failed the unchanged gate. All 30 checkpoint-case receipts, including failures, are archived and verified in the new CPU-only evidence snapshot. Ruff and `git diff --check` passed. No CUDA, full-model quality, measured memory or performance claim is made for this prototype.

## Access and resource risk

At 16:30:55 UTC the session's network restrictions denied SSH to both hosts and AWS EC2 endpoint access. Approval policy is `never`; no alternative route or bypass was attempted. User authorization to use `al` does not remove this execution restriction. This is not an observed expired-login error.

Fresh read-only probes at 16:44:41–16:44:51 UTC produced the same SSH `Operation not permitted` and regional EC2 endpoint connection failures. No new instance/process state was obtained. Academic-paper browsing remains available through the separate web tool; that does not establish SSH or EC2 access.

The third consecutive goal-turn check at 16:57:56 UTC again denied both SSH hosts and both EC2 endpoints. The previous two turns made local integration, scoring, evidence and literature progress despite that restriction; this turn completed the bounded-GNN CPU follow-up and preserved its failures. Local follow-up work is now handed off. Remaining execution, remote evidence recovery, Git publication, cost reconciliation and teardown require restored permitted networking. The full goal remains unachieved; unsupported network bypasses and assumed job termination are not alternatives.

Both task hosts were last verified reachable at 16:27:25 UTC, without a Spot interruption notice. They **may still incur charges**. Previously verified shutdown-to-terminate deadlines were 19:28:00 UTC for L4 and 23:15:34 UTC for L40S, on 2026-10-07. Their execution is not yet confirmed.

Instance IDs, regions, access resources, and exact cleanup scope are in [operations.md](operations.md). When permitted SSH/AWS connectivity returns, first inspect live state, retrieve/checksum remaining artifacts without overwriting prior attempts, then terminate task-owned resources and verify teardown. Preserve unrelated infrastructure. Remote result recovery, final cost accounting, and teardown cannot be replaced with assumptions from expired tool handles or shutdown schedules.
