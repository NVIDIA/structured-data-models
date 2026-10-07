# Kumo multi-GPU investigation

Started 2026-10-08 from SDM `842c408fe2a8711bdf2e7cff4bfbe54266d6b940` on branch `research/kumo-multigpu-20261008`.

## Questions

- Which decomposition improves warm inference throughput and latency for KumoTabular and KumoRelational while preserving the same logical model computation?
- Which decomposition increases feasible context or model capacity, and what are its fit, communication, memory, and quality costs?
- How do simple public-API compositions compare with model-internal changes and combinations of parallelism axes?
- Which small, generic pieces merit integration into SDM, and which belong in examples or experimental tooling?

## Ownership

| Owner | Responsibility |
| --- | --- |
| Coordinator | Integration, experiment priorities, resource envelope, final evidence-backed conclusions |
| Spot operator | Sole owner of EC2 resource changes, authentication, runtime, transfers and teardown |
| Ensemble implementation | Stable estimator placement, shared preprocessing, device-resident member caches |
| Data-parallel implementation | Independent complete query batches, persistent workers, EP/DP composition |
| Context-parallel implementation | Cached attention KV shards, normalized partial-attention reduction |
| Model-parallel implementation | Stage and layer placement, capacity and communication tradeoffs |
| Workloads and models | Exact checkpoints, real TRAIN/validation workloads and prepared graph batches |
| Tabular runner | KumoTabular baseline, scale matrix, raw measurements and predictions |
| Relational runner | KumoRelational baseline, graph-preserving scale matrix, raw evidence |
| Quality audit | Independent semantics, numerical differences, metric comparison and timing review |
| Research integration | Approach comparison, limitations, prior work and integration recommendations |

Each implementation uses an isolated worktree based on the same revision. GPU execution is scheduled with the operator and runners to prevent simultaneous experiments contaminating timings. Source revisions and commands are recorded for every run. The GPU benchmarking skills are not used, per the user's instruction.

## Approaches

| Approach | Principal question | Initial scope |
| --- | --- | --- |
| Ensemble parallel | Does dividing the same E members scale efficiently? | 1, 2, 4 GPUs; stable RNG/member identity; resident cache |
| Query data parallel | Does distributing unchanged query batches improve aggregate throughput? | Persistent replicas; fixed batch/graph boundaries; threads and processes |
| Hybrid query/ensemble | Is 2 query groups x 2 ensemble workers better than pure 4-GPU EP or DP? | Four-GPU composition on the same inputs |
| Context parallel | Can KV shards reduce cache pressure and accelerate long-context replay? | Both Kumo ICL blocks; full replicated fit initially |
| Stage/layer placement | Can distributed weights/cache enable workloads exceeding one GPU's capacity? | Encoder/ICL separation and contiguous ICL block placement |
| Further approaches | Are tensor/head, feature, graph, or row partitions worth implementing? | Architecture analysis, measured probes where viable; document rejections |

## Workloads

Initial real-data candidates are Covertype multiclass classification, California Housing regression, RelBench H&M user churn, and RelBench F1 driver position. Context examples and learned preprocessing come from TRAIN. Quality evaluation uses validation labels; TEST remains unused. Dataset preparation records exact row identities, split recipe, context selections, schema, class ordering, checkpoint revision and hashes.

Relational execution uses native RelatedTables with exactly sampled graph neighborhoods. Initial H&M sampling uses two hops with fanout `[16,16]` and temporal `last` sampling. A query batch and its complete related graph are indivisible when comparing GPU counts. Sampled batches are reused to isolate execution scaling; sampling and complete pipeline timings are reported separately.

## Comparison rules

- Record native public SDM 1-GPU performance and a 1-GPU executor with the same cache residency and RNG policy as each multi-GPU variant. Current native multi-estimator fit offloads caches, so native-versus-resident gains must not be attributed solely to parallel execution.
- Use the same model weights, preprocessing state, context examples, ensemble members, query examples, batch boundaries, precision, and output reduction within each scaling comparison.
- Include estimator batching as a single-GPU control. Increase context, batch and E deliberately rather than exhaustively multiplying every parameter.
- Start with smoke runs, then at least three measured repetitions after warmup. Reverse or interleave arm order where practical. Report individual samples and variation, not just a best run.
- Test strong scaling with a fixed total workload and weak scaling with increasing query load per GPU as separate experiments.
- Compare CP against both native attention and the same partial-attention kernel on one rank. Do not attribute a kernel switch to distributed speedup.
- Report load, preprocessing, sampling, fit, warmup, transfer, prediction, gather, and complete invocation costs. Synchronize participating GPUs at timing boundaries. Profile separately from the primary timed pass.
- Record per-GPU allocated/reserved peaks, model/cache bytes, host RSS, device utilization, topology and communication. Aggregated device time is not elapsed wall time.
- Save raw ordered predictions and evaluate numerical error, classification agreement, AUROC/log loss/accuracy or regression median RMSE/MAE. Preserve and compare all regression quantiles. Tolerance and nonfinite failures are explicit results.
- Classify changed-context/changed-neighborhood methods as approximate approaches with their own quality comparison. Do not mix them into equal-work scaling claims.
- Keep failures, OOMs, invalid-semantic approaches, capacity rejections, and abandoned prototypes in the report with the cause and attempted remedy.

## Initial resource plan

Use one four-GPU EC2 Spot host initially, favoring available L40S/L4/A10G capacity. The coordinator's initial operating envelope is $100, with an eight-hour per-host termination timer. This is an initial allocation, not a claim that the user supplied a spending limit. Additional topology experiments may use eight GPUs or NVLink if they answer a concrete question. No On-Demand fallback is inferred. Task-owned resources are tagged `kumo-multigpu-20261008`; unrelated infrastructure is preserved.

## Completion evidence

The study should deliver tested SDM integration candidates, runnable experiments, real-data quality and scaling tables for both model families, documented failures and coverage limitations, exact commands/runtime/source provenance, and a final resource teardown record. A prepared harness or architecture proposal alone does not count as a measured result.
