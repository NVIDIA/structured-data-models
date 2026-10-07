# TabPFN literature: inference ideas for SDM

Reviewed 2026-10-08. Primary papers and the official PriorLabs repository only. **Reported** denotes authors' results or inspected implementation behavior; **SDM proposal** denotes our inference, not a reproduced TabPFN result. This review adds no implementation or GPU measurements.

The strongest immediate lesson is to treat preprocessing, member identity, batching, and cache placement as one explicit inference plan. More GPUs alone do not solve short-kernel launch overhead. The most promising capacity idea beyond our completed experiments is staged row-encoder execution that preserves global context statistics, rather than merely distributing an already-built cache.

## Primary papers and scope

### 1. Ensemble semantics, not 32 independently trained networks

[TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/pdf/2207.01848v6), Hollmann et al.; ICLR 2023; first submitted 2022-07-05, inspected revision 2023-09-16. Relevant: Figure 1, §3, §5.2, Appendix E.2.

**Reported:** Query tokens attend to training tokens rather than other query tokens. The ensemble evaluates one checkpoint under 32 transformed/permuted datasets, including feature/class-index rotations and optional power transformation. Its main numerical-data evaluation restricts training size to 1,000 rows, 100 features, and ten classes. Figure 5 reports 0.62 seconds on GPU for the 32-permutation method; this is not a multi-GPU scaling result. §5.2 also studies extrapolation to larger contexts, separately from the main benchmark.

**SDM proposal:** Parallelize complete logical members without independently re-fitting their preprocessing or changing class alignment. Query DP requires a fixed fitted context and independent query computations; it does not license arbitrary splitting of relational neighborhoods. Preserve our per-member seeds and regression inverse transforms before aggregation.

### 2. The Nature result is a quality/runtime baseline, not a distributed inference protocol

[Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6), Hollmann et al.; Nature 637, 319–326; published 2025-01-08. Inspected: publisher abstract/main-text search extract; full methods unavailable through this session's browser.

**Reported:** The paper evaluates small-to-medium tabular learning up to 10,000 samples and 500 features, including classification and regression. Its headline 2.8-second classification result compares against heavily tuned baselines, not an EP1/EP2/EP4 ladder. Do not repurpose that number as GPU scaling evidence.

### 3. TabPFN-2.5 explicitly measures multiple GPUs—but cold fit plus predict

[TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/html/2511.08667v2), Grinsztajn et al.; first submitted 2025-11-11, inspected revision 2026-02-05. Relevant: §4.3, §5, Figures 8 and 18–20.

**Reported:** Figures 8/18 compare one and four GPUs for processing context plus 500 query rows, across H100/A100/T4 configurations. This is not repeated cached inference. §5 distinguishes uncached batching from `fit_with_cache` and reports substantial cache memory: approximately 6.1 KB GPU and 48.8 KB CPU per training cell for classification in that release. Feature subsampling caps each estimator at 500 features in the runtime curves. The plots show capacity limits as well as latency; no numeric multi-GPU speedup is extracted here without digitizing the figures.

**SDM proposal:** Retain three separate baselines: public offloaded inference, resident one-GPU execution, and distributed resident execution. Include cold fit/cache creation and amortized repeated prediction; do not compare this paper's cold curve to SDM's warm rows/s.

### 4. TabPFN-3: bound encoder activation memory before distributing attention

[TabPFN-3: Technical Report](https://arxiv.org/html/2605.13986v2), Grinsztajn et al.; first submitted 2026-05-13, inspected revision 2026-05-28. Relevant: §2.1, §2.3, §2.4.1–2.4.2, Figures 4 and 6–8.

**Reported:** The architecture compresses columns into row representations. Its two-phase chunking first computes global inducing states, then reuses them while processing row chunks; naively fitting separate chunks would change attention. It reports roughly fivefold peak-memory reduction at its largest tested shapes. Single test-side KV-head attention reduces the cache to about 7 GiB per estimator at one million rows. Figure 7 is a **single-estimator model-forward benchmark excluding preprocessing**. Figure 4 describes a row/feature validation frontier, not simultaneous support for every pair of advertised maxima.

**SDM proposal:** Examine a public row-encoder fit-summary/replay boundary. Existing Kumo weights already use inducing states and grouped KV attention, so reducing head count is a separate numerical/model change, not a safe executor switch. Measure fit peak as well as retained cache: our compaction saved retained memory but did not reduce fit peak.

### 5. TabPFN-3.5: stronger representations can simplify preprocessing

[TabPFN-3.5: Technical Report](https://arxiv.org/html/2609.17895v2), Jäger et al.; first submitted 2026-09-15, inspected revision 2026-09-22. Relevant: §2.5–2.6, §3.1–3.4, Figure 7.

**Reported:** New cell encodings replace separate quantile/robust-scaling/SVD transformations; ensembles still use permutations. Width increases while single test-side KV-head width stays fixed. Decoder keys are cached directly. Figure 7 uses one RTX PRO 6000 Blackwell, measures one estimator's model forward, and multiplies by default ensemble size: **not a measured parallel ensemble**. Relational results include a Plus checkpoint/internal harness; Plus/Thinking implementation details are proprietary.

**SDM proposal:** Cache only tensors actually consumed by replay. Longer-term, learn invariances that reduce preprocessing diversity, but retrain and re-evaluate; removing existing Kumo recipes is not semantics-preserving. These latest claims do not establish open-checkpoint relational or L40S multi-GPU speedups.

## What the official implementation actually does

Sources inspected on 2026-10-08: [inference.py](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/inference.py) and [parallel_execute.py](https://raw.githubusercontent.com/PriorLabs/TabPFN/main/src/tabpfn/parallel_execute.py). These are moving `main` links, **not commit-pinned evidence**; the browser could not retrieve the GitHub commit API. No source code was copied into SDM.

**Reported implementation, inference.py:** `_member_groups` distributes compatible members before batching; `_member_key` considers model and prepared shape. `_yield_in_member_order` restores logical ordering. `InferenceEngineExplicitKVCache` keeps external member/group caches and stateless model replicas, avoiding a model copy per member. It supports resident or CPU-offloaded caches, can stage caches on CPU during memory-constrained fit, and bounds query chunks using row/cell budgets. Optional cache precisions include int8/fp8/adaptive on supported architectures. Prediction calls `cache.to(device)` on the device assigned to the task: stable cache affinity is not guaranteed merely by using this scheduler.

**Reported implementation, parallel_execute.py:** Tasks use a Python thread pool, an available-device queue, ordered result delivery, and CUDA completion events. One device uses the caller thread. The comments explicitly condition speedup on work releasing the GIL. GPU SVD's first-use initialization is warmed separately to avoid a documented threading race. These are implementation choices, not demonstrated SDM performance improvements.

## Ranked SDM follow-ups

These are proposed experiments, not work performed by this literature review. Existing measurements are in the [central report](README.md), [ensemble design](ensemble.md), and [trace analysis](ensemble-profile.md).

| Priority | Proposal | Why it fits our evidence | Required boundary / acceptance test |
|---|---|---|---|
| 1 | One explicit member plan, shared native batching, stable cache ownership | Fixed-batch-two EP2 reached 1.446× resident EP1; changing local width explained the initial four-GPU numerical failure | Freeze recipe/member RNG/class order and local batch width before placement; compare native tuned, resident1, distributed2/4 |
| 1 | Prefer persistent process query DP for throughput when context replication fits | Our fully gathered tuned tabular process DP reached 3.777× at four GPUs; threads showed little kernel overlap | Fixed complete query batches and identical fitted state; include IPC/gather, child GPU memory, and CPU RSS; do not infer process-EP performance from DP |
| 2 | Two-phase encoder summary plus chunked row replay | Cached CP does not remove our full-context fit work; compaction left fit peak unchanged | Prove global attention/preprocessing equivalence first; test chunk boundaries, uneven final chunks, peak fit memory and full prediction arrays |
| 2 | Separate cache storage precision from compute precision | We already found physical cache storage differed from logical tensor size | First retain exact compaction baseline; quantization gets a new numerical gate, calibration/log loss/full quantiles, and dequantization cost; measure unique storage rather than tensor bytes alone |
| 2 | Reduce host launch work using already-prototyped graph replay | EP4 trace had only 1.085% of time with two or more GPUs computing; persistent autocast barely helped | Finish pretrained graph/process-EP runs before more implementations; report capture/spawn costs and memory separately; small CUDA tests are insufficient |
| 3 | Evaluate quality versus context, member count, and latency together | Larger context/ensemble changes the predictor; a faster executor should not silently alter either | First strong-scale a fixed configuration, then independently sweep context/E under a fixed resource budget; preserve temporal validation and report task quality, not only numerical parity |

For KumoRelational, retain complete two-hop neighborhoods and temporal cutoffs. A cached tabular summary does not prove that GNN message passing or shared-entity neighborhoods can be independently chunked. Table encoders may expose useful independent work, but attention summaries and graph aggregation require their own dependency analysis.

## Access and interpretation limits

- Nature full HTML/PDF returned browser errors; only the publisher's indexed abstract/main excerpt was available. OpenReview imposed a browser challenge; the original TabPFN arXiv PDF was readable instead.
- Current arXiv revisions and official raw repository files were readable. GitHub commit pinning failed, so implementation observations are access-date-qualified. No restricted host/network workaround, checkpoint download, GPU execution, or benchmark skill was used.
- Paper speedups use different models, hardware, workloads, timing boundaries, and sometimes proprietary paths. None is substituted for an SDM measurement. No reviewed source establishes that context parallelism or tensor parallelism will accelerate current Kumo checkpoints.
- Cache/head/representation changes may require retraining or materially change predictions. Source inspection is not license permission to redistribute checkpoints or proprietary implementations; any later adoption needs the relevant release's terms checked separately.
