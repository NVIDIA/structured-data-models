# Independent completion audit

Audit snapshot: 2026-10-08, integration branch at `a39365a48`. This is a scope/claim audit of the study documentation, local branch inventory, and retained result/audit pointers, not a fresh cloud-state query or another GPU experiment. Later result commits and teardown receipts supersede the pending states below.

## What the user actually requested

The user requested a worktree/branch; investigation and prototype benchmarking of ensemble, context and data parallelism; real RelBench experiments on Spot GPUs; substantial agent delegation; documented setup, measurements and negative results; careful baseline comparisons; and subsequently a deep study covering both KumoTabular and KumoRelational, further viable approaches, scale, performance, prediction quality, memory, literature inspiration and generic SDM integration. The explicit prohibition on GPU-benchmarking skills remains part of the scope. "Be relentless" requires pursuing a well-supported investigation, not implementing every architectural idea or producing a positive speedup for every method.

## Requirement-by-requirement assessment

| Requirement | Assessment at snapshot | Evidence / important boundary |
|---|---|---|
| Both model families and real workloads | Delivered | Covertype/California Housing and native RelBench H&M/F1, TRAIN-context/VAL-only quality; relational graphs retain complete two-hop neighborhoods rather than flattening them |
| Ensemble parallel prototype and fair controls | Delivered | Resident/thread/process/graph variants, native estimator-batch controls, logical member seeds and output ordering; published minimal branch contains only the smaller threaded resident candidate |
| Context parallel prototype and investigation | Delivered with explicit limitations | Both ICL families, native/LSE1/CP2/4 controls, empty/uneven shards and native GQA; full fit remains replicated; BF16 F1 gate fails, full-model FP32 passes at a throughput/memory cost |
| Query/data parallelism | Delivered at 1/2/4 GPUs | Persistent replicas, complete fixed batches, strong/weak scaling and later final-gather controls; large-batch graph composition is a supported numerical-tabular research backend, not a relational capture implementation |
| Go beyond the three initial options | Delivered | Stage/layer placement, hybrids, cache compaction, process isolation, CUDA graph replay, destination-block GNN prototype, and reasoned TP/table/full-fit/graph-partition designs |
| Proper scale rather than tiny smoke only | Delivered with topology scope | Fixed-work 1/2/4-GPU ladders, increased queries/batches/estimators, contexts through64k and retained OOM/retry controls; homogeneous8/NVLink unavailable; mixed8 is a separate pending operational test, not a prerequisite to declaring the measured4-GPU scope valid |
| Quality and correctness | Delivered for accepted claims | Ordered IDs/graphs/checkpoints, all repeats and regression999 quantiles, fixed tolerances, independent audits; failed BF16CP, original batching comparisons and actual-checkpoint GNN module screens remain failures even when aggregate metrics are close |
| Memory and capacity | Delivered, no exclusive-capacity proof | Resident versus CPU-cache policies, unique backing storage, allocated versus reserved peaks, fit/prediction separation and allocator controls; native H&M64k/E8 offload fits one L4, so do not claim this workload requires multiple GPUs |
| Profiling and explanation | Delivered | Retained EP/CP and graph-replay Nsight evidence, launch counts, active intervals, transfers and collective waiting; no unsupported occupancy, pure-bandwidth or GIL-only causal claim; lost placement traces explicitly unavailable |
| Papers and prior user/Aki ideas | Delivered | Existing branches/commits compared; four primary-source literature reviews and synthesis distinguish measured SDM behavior, published claims and proposed checkpoint-changing/approximate methods |
| Generic SDM integration and branch publication | Candidate delivered; final study sync pending | `feature/ensemble-parallel` local and remote-tracking refs both `75bd74af2`, four files/599 additions over842c408fe; no cloud machinery, CP or process/graph adapters in that candidate. Study locala39365a48 is ahead of remote-trackingc8b028b40 at this snapshot; this was not a fresh remote query |
| Reproducible evidence | Substantially delivered, final reconciliation pending | README reports42 verified indexes with451 archived and1,070 external references; source/runtime/checkpoint identities and immutable failures retained. These are reference counts, not necessarily unique files, and local hash verification does not establish cloud state |
| Spot lifecycle and final handoff | Not complete at snapshot | Original instances/disks confirmed absent; replacement Ohio4L40S and Frankfurt4L4 still assigned work. Timers are safeguards, not teardown verification |

## Genuinely remaining closure work

1. Resolve the explicitly queued mixed-eight comparison and verified-allocator resident retries with downloaded result/failure/stop receipts and independent review. These are already selected controls, not an invitation to broaden the experiment matrix. A stopped or failed attempt is a legitimate final outcome if its reason and scope are retained.
2. Finish additive evidence collection and verify the final snapshot set against both archived records and existing local large artifacts. Reconcile the actual measured source per arm; do not label an unrun final branch revision as the benchmark source. Preserve earlier missing remote results as lost evidence, not recovered or numerical failures.
3. Refresh stale handoff/summary statements, publish the final study commits, and confirm the final branch identifiers. The small feature candidate is already separately scoped; process/graph speedups must not be presented as features delivered by that candidate.
4. Obtain the sole operator's explicit final instance/disk/access-resource cleanup receipts after scientific owners release both hosts. Remove only task resources, including task ingress/key material, and preserve user/unrelated resources. Record observed launch/termination intervals and a clearly labeled price-based cost estimate; delayed billing data must not be invented as a final invoice.
5. Give the user a concise self-contained comparison table: strongest useful throughput decomposition, memory tradeoffs, failed numerical arms, hardware/context/E/query settings, quality scope, branch links, reproducible evidence location and remaining design-only options.

No newly invented TP, full-context fit-sharding, graph-partition, table-fan-out, EP×CP or overlapped pipeline implementation is required to close the user-requested investigation. Neither productionizing every research adapter, proving a multi-GPU-only capacity frontier, testing every schema/model size, nor obtaining denied NVSwitch capacity is required. Long-MHA32kCP and lost placement profiling are optional diagnostics with explicit unmeasured status, not hidden success claims.

## Claim hygiene for the final report

- Keep native versus executor1 versus executorN ratios separate. GraphEP4 versus tuned native includes a single-GPU graph improvement; that entire ratio is not multi-GPU scaling.
- Preserve timer boundaries. Fully gathered CPU-complete query results, worker-return timers, preloaded-pipe inputs and timed IPC are not interchangeable denominators.
- Mixed eight consists of four Frankfurt L4s and four Ohio L40S GPUs across the Atlantic, not a homogeneous eight-GPU machine or NVLink result. Any speedup needs matched constituent controls and actual unique-query counting.
- Context parallelism's full-FP32 pass does not repair the BF16/partial-FP32 quantile failures. All-gather's roughly2.5% nominal changes are ordered three-pass observations, not robust randomized run replications or a native throughput crossover.
- A requested allocator environment variable is not evidence of effective policy. Keep the additive ignored-variable correction and later verified retry distinct from historical legacy-variable runs.
- Destination-block GNN is not an exact approved replacement: actual-checkpoint GPU module failures remain despite full-model task probabilities passing a looser screen. Lower live allocation can coexist with higher reservation.
- Preserve validation-prefix/single-fit limitations. Repeat inference samples do not become independent fits, general model-quality rankings, or service-tail latency measurements.
- Hash-index references to local large artifacts are retention pointers, not embedded public datasets/prediction arrays. Do not delete the local artifact store merely because small JSON records are committed.

## Documentation freshness issues identified

At this snapshot, `handoff.md` retains a historical replacement paragraph saying the second host is still being sought and a mixed-eight row saying never launched. These need a historical qualifier or final queue-state update. `literature.md` still frames pretrained process/graph controls as a future first experiment although they now have results, and its unresolved F1-gate wording should explicitly say BF16. `integration.md` priority4 asks for higher-precision recombination even though partial-FP32 failure and full-FP32 completion are now measured. These are summary freshness issues; they do not invalidate the retained raw results or require new experiments.

Overall judgment: the scientific investigation is broad enough to satisfy the requested approach/family/depth scope. Completion should now be driven by selected final controls, honest synthesis, publication and verified resource closure—not further architecture expansion.
