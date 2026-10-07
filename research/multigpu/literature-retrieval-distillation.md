# Retrieval, shared contexts, and distillation

Reviewed 2026-10-08. These three primary papers complement the [TabPFN](literature-tabpfn.md), [TabICL](literature-tabicl.md), and [relational](literature-relational.md) notes. Paper observations below are not SDM measurements. Proposed experiments have not been run.

## Three sources, two different inference strategies

**Original TabDPT — local context is part of the learned method.** [TabDPT: Scaling Tabular Foundation Models on Real Data](https://arxiv.org/html/2410.18164v3), NeurIPS 2025, inspected arXiv revision 2026-01-17, §3.3–3.4 and §4.4. It uses row tokens, real-data self-supervised pretraining, and nearest-neighbor contexts during both pretraining and inference. A query retrieves labeled training rows; retrieval and context construction are part of its cost. Its ablations favor retrieval over random selection in that model. This does not establish that replacing the context of an unchanged Kumo checkpoint preserves its quality. The main reported evaluation also distinguishes a training-plus-validation context from a training-only variant (§4.3); our existing validation labels must remain unavailable to context selection.

**TabDPT-Turbo — shared long context can beat query-specific retrieval.** [TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction](https://arxiv.org/html/2608.01400v1), preprint 2026-08-02, §3.4 and §4.1–4.2. New architecture and long-context pretraining allow shared-context inference without retrieval. Its same-host comparison uses one H100, 96 vCPUs, and eight inference passes, separately reporting fit and prediction time. The retrieval ablation explicitly includes repeated query-specific contexts. This is a useful counterpoint to treating retrieval as an automatic optimization: shared context enables amortization and batching. It is not an executor-only improvement applicable to arbitrary old checkpoints. The [official repository](https://github.com/layer6ai-labs/TabDPT-inference) identifies Turbo with v1.2; its September 2026 v1.3 release changes architecture/training again. Pin paper, package, and checkpoint versions separately.

**Distillation — buy a different predictor with offline work.** [Distillation of Tabular Foundation Models into Efficient Predictors](https://arxiv.org/html/2610.01435v1), preprint 2026-10-01, §3 and §4.3. It studies TabICLv2/TabPFN-v3 teachers and TabM/XGBoost students. The selected recipe uses full labeled training context, teacher supervision, and synthetic CutMix queries; out-of-fold supervision is a distinct ablation. Its latency comparison uses resident models and 32-row requests on an H200, while offline cost includes target generation and student fitting. It reports a quality–latency trade-off, not prediction equivalence. This very recent preprint is motivation for controlled replication, not established Kumo evidence. The linked implementation was unavailable through the browser; no code behavior was verified.

## Proposed SDM experiments

The following designs are our inferences from those papers and this study's bottlenecks, not claims about either authors' implementation.

### 1. Shared-local-context routing, with a global-context control

Keep the current checkpoint and compare three explicit context policies: fixed global random context; query-specific nearest neighbors; and a bounded collection of shared local contexts with one fitted cache per group. The third policy is our proposed compromise, not the original TabDPT algorithm. Start with Covertype and California Housing, before adding native H&M task observations and timestamp-valid two-hop neighborhoods.

- Fit scaling, distance representation, clusters, and search index using TRAIN only. Freeze selected context IDs and routing before the GPU-count sweep; an approximate search index or different tie-breaking may itself change the predictor.
- At fixed context size and ensemble count, report context-coverage/rare-class diagnostics, quality, and cost. Then separately compare configurations at equal latency or GPU-seconds; do not present fewer context rows as exact strong scaling.
- Measure index creation, search, preprocessing, context assembly, cache fitting, routing, output restoration, and prediction. Query-specific contexts can eliminate the reuse that makes Kumo's fitted cache economical. Count physical cache storage across all groups, cache misses, and eviction/refit cost.
- Exact process query DP can distribute a frozen routing policy, provided each worker receives the same complete query batches and contexts. GPU count must not determine the statistical policy. Compare against our tuned resident and process-DP references, not repeated uncached fitting.
- For relational input, retrieve task observations, then construct their native temporal neighborhoods. Do not independently partition related-table row spaces. A reusable row embedding needs the same preprocessing/member identity, cutoff, and sampled neighborhood; entity identity alone is insufficient.

### 2. Teacher-to-student distillation for repeated inference

Use existing exact process DP to generate teacher targets offline. First compare a directly supervised student and a distilled student of the same architecture under equal tuning budgets. Separate full-TRAIN-context targets from out-of-fold targets; include all fold-specific fit/target-generation costs. Full-context self-inclusion is an intentional training recipe, not validation leakage, but its generalization needs its own held-out evaluation. Add synthetic queries only as a separate arm; relational synthetic rows must not invent invalid keys or future observations.

Measure total offline cost, student load, CPU and GPU serving, memory, and refresh frequency. For matched requests, the break-even count is incremental offline cost divided by the teacher-minus-student latency saving; if the student is not faster, there is no finite break-even. Include any costs incurred by both deployment alternatives consistently. Our teacher reference already caches fitted state and has a tuned four-GPU process path, so a naive uncached teacher would exaggerate savings.

Classification needs aligned class probabilities, log loss, AUROC where defined, and calibration. Regression needs the full output contract: a scalar-only student does not preserve Kumo's 999 quantiles. Either train and evaluate a distributional/quantile student, including crossing rates, pinball loss and interval coverage, or label the scalar interface as a deliberate model change. A flat student for KumoRelational also changes its inputs unless temporally valid graph features are supplied; a compact graph-aware student is a separate experiment.

## Decision boundary

Prefer improving shared-cache execution first: we already measured exact throughput gains without replacing Kumo. Retrieval is a quality–cost hypothesis, not a substitute for CP. Distillation is attractive only when a stable dataset/task has enough repeated queries to amortize training and the quality loss is acceptable. Long-context architecture, learned compression, fewer KV heads, or changed target routing require a new checkpoint and a separate evaluation track. No paper result here establishes multi-GPU speedup for current SDM models.
