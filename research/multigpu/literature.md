# Foundation-model papers: next SDM experiments

Reviewed 2026-10-08. **Recommendation: improve current-checkpoint execution and fit-memory behavior first; evaluate retrieval and distillation as separate predictors.** This review adds no GPU measurements or core changes. The completed experiments and unresolved controls remain in the [study report](README.md).

Detailed primary-source notes: [TabPFN through 3.5](literature-tabpfn.md), [TabICL/TabICLv2](literature-tabicl.md), [relational models](literature-relational.md), and [TabDPT, Turbo, and distillation](literature-retrieval-distillation.md). They pin paper versions, distinguish inspected implementation from paper claims, and record access limits.

## What the papers change about our priorities

TabPFN-3's two-phase encoder and TabICLv2's staged inference suggest attacking memory **before** full row/cell activations exist; simply distributing a completed ICL cache is insufficient. These are architectural boundaries to investigate, not evidence that independent shard fits are equivalent. [TabPFN-3 §2.4](https://arxiv.org/html/2605.13986v2), [TabICLv2 Appendix H](https://arxiv.org/html/2602.11139v2).

Relational papers reinforce the distinction between database retrieval, task-conditioned encoding, and final context attention. KumoRFM-2's early task conditioning is a warning against caching learned entity embeddings solely by entity ID. Its database-scale serving discussion does not establish one-request multi-GPU scaling. [KumoRFM-2 §§2–3](https://arxiv.org/html/2604.12596v1).

Retrieval has competing evidence: original TabDPT trains around local contexts, whereas Turbo changes training/architecture to exploit a shared long context. The practical question is quality per **total request cost**, including lost cache reuse, not just attention length. [TabDPT §§3.3–3.4](https://arxiv.org/html/2410.18164v3), [Turbo §§3.4–4.2](https://arxiv.org/html/2608.01400v1).

## Five ranked experiments

These were proposed SDM tests; subsequent status updates below refer to this study's measured controls, not paper speedup claims.

| Rank | Experiment | Semantic category | First decisive control |
|---|---|---|---|
| 1 | Pretrained process-EP/graph replay controls are now completed; next consolidate member plans, local batching and cache ownership | Current checkpoint; preserve explicit batching/numerical policy | Retain fixed member seeds/context/query batches and measured native/resident/graph controls; graph query-DP scales3.716× at four GPUs on the larger-batch cohort |
| 2 | Bound upstream fit allocations: destination-block GNN statistics/projection, globally merged inducing summaries, then local row encoding | Current checkpoint, mathematically equivalent target; floating-point order may change | Compare intermediate FP32 tensors and BF16 full predictions before scaling; preserve all GNN statistics, globally selected keys and nonlinear ordering |
| 3 | Explicit prepared-topology and cache-residency policies: compact/full-KV versus offloaded or representation-cache/recompute | Topology/compaction intended exact; recomputation has numerical controls; quantization is a separate approximate arm | Immutable input identity and invalidation; fit peak, retained unique storage, transfers, host RAM and repeated-query latency; never reuse task-conditioned embeddings by entity ID alone |
| 4 | Shared local contexts or smaller-support ensembles, followed by fixed-policy query DP | Changes context/predictor, even without changing weights | Global-context, per-query retrieval and shared-local-cache controls at matched context/E and separately matched GPU-seconds; include search, every refit and relational sampling |
| 5 | Dataset-specific teacher-to-student distillation | Replaces model; offline training required | Same-architecture supervised student, full-TRAIN-context versus out-of-fold targets, target-generation/training cost and request-volume break-even; preserve or explicitly change the quantile/graph-input contract |

**Why this order:** exact tuned process DP reached3.777× at four L40S GPUs; subsequent graph-backed DP reaches3.716× at batch1,024, with within-backend exact outputs. Fixed-local-batch EP2 reaches1.446×. These are measured baselines, not paper predictions. Destination-block GNN now reduces full-model fit allocation but fails23 actual-checkpoint GPU hidden-output gates; it remains exploratory despite passing downstream probability screens. Native CPU cache offload fits E8/C64k on one L4, while resident stage2 succeeds where resident EP1 fails: capacity conclusions must specify policy. Cached CP still does not distribute upstream fit activations. Its F1 **BF16** full-quantile failure remains; full-model FP32 passes separately without a throughput win. See the [measured evidence](README.md).

The unimplemented global-summary part of rank2 should begin as a single-GPU memory experiment; the bounded-GNN component already has measured failures above. Merge local attention numerators/normalizers **before** residuals, normalization and MLP; use globally consistent key selection and context-length scaling. For relational graphs, preserve disjoint example ownership and edge semantics. Do not average nonlinear shard outputs or independently fit each shard's recipe. Only then test distributed fit.

Rank 5 follows the recent distillation paper's distinction between one-time conversion cost and resident inference, but must use our tuned cached teacher baseline. A faster student is not automatically cheaper before the dataset's next refresh. [Distillation of Tabular Foundation Models, §4.3](https://arxiv.org/html/2610.01435v1).

## Non-negotiable comparison controls

- **Precision:** record parameter, Q/K/V, cache, local-output, reduction and actual kernel dtypes independently. FP32 output does not establish FP32 attention. Keep native, single-rank same-kernel and multi-rank controls; no relaxed threshold to hide the existing CP regression failure.
- **Quality:** exact-execution arms compare ordered probabilities and all 999 regression quantiles, not only accuracy/median. Approximate arms need log loss/calibration or pinball loss/coverage, paired uncertainty, and an explicit quality–cost frontier. Keep unchanged baseline failures in the record.
- **Temporal scope:** select retrieval/sampling settings on a TRAIN-only inner split; score fixed VAL queries. Preserve timestamp cutoffs, complete sampled neighborhoods, class vocabulary, member IDs and stable query order. Paper protocols that add VAL labels before TEST evaluation cannot be copied into our VAL experiment.
- **Cost and scale:** report cold setup, repeated requests, sampler/preprocessing/IPC/gather, per-device peak memory, CPU RSS, and dollar/GPU-second cost separately. Scale a fixed predictor before changing context or ensemble breadth. An H100 paper curve, training GPU count, database-size claim or multiplied single-estimator time is not an SDM strong-scaling result.

Keep these experiments in research/examples until their contract is established. Minimal core candidates remain explicit fitted-cache lifecycle and composable attention/encoding primitives, not fleet orchestration. Learned compression, fewer KV heads, altered target routing, and new relational bridges belong to a new-checkpoint track. Original remote-only results were lost; replacement runs, final verification and teardown are separately tracked in the handoff. This literature review does not establish operational closure.
