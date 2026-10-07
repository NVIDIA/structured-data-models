# Kumo model parallelism investigation

Source baseline: `842c408fe`, October 8, 2026. This document separates implemented prototypes, numerical evidence, and unmeasured proposals. It does not claim GPU performance until the shared runners supply results.

## Implemented experiments

Two minimal capacity experiments preserve the original attention and graph computations. Both use one model, not one complete model per device.

| Experiment | Placement | Communication during cached prediction | Expected benefit | Main limitation |
|---|---|---|---|---|
| `stage` | Cell/row encoder and relational GNN on GPU0; entire ICL on final GPU | Row embeddings cross once; final logits return | Separates front-end and ICL allocations, minimal integration | ICL cache remains concentrated on one GPU; middle GPUs unused when N > 2 |
| `layers` | Front-end on GPU0; contiguous equal-count groups of ICL layers across N GPUs | Hidden rows cross each stage boundary; final logits return | Shards ICL weights and layer KV caches without replicating full model | Sequential execution; front-end memory remains concentrated on GPU0 |

These are **sequential placement**, not an overlapped pipeline. Extra GPUs do not divide the compute time of one request. A reduction in single-GPU memory is the primary hypothesis; a throughput increase would need a separate pipeline scheduler overlapping different batches, which this implementation does not contain.

Implementation: `research/multigpu/placement.py`. Runner adapter: `research.multigpu.icl_placement:factory`, with `--mode adapter --replicas 1 --gpus N --placement stage|layers`. The factory returns the same `EnsembleParallel` executor used for the one-replica resident baseline. This preserves shared recipe preprocessing, member order/seeds, target inversion, and final ensemble reduction. It invokes placed inner modules while leaving each member cache on its producing devices.

Construct the model on GPU0, install placement, then fit through the returned executor. Do not call `.to()` on the installed model or fitted cache. The hook-based prototype is inference-only and is not a supported `torch.compile` or checkpoint serialization interface. Equal layer counts are an initial placement policy; measured memory can motivate a weighted partition later.

At the constructor boundary, all participating devices synchronize once after parameter migration. During forward, remote CUDA events join the caller stream. This includes fit calls returning zero query rows: a zero-byte output transfer alone cannot be assumed to wait for remote cache writes. The resident executor can then drain its caller stream safely. Measurements must still explicitly synchronize every participating GPU and collect memory by actual tensor device.

## Architecture facts affecting the choice

Current public KumoTabular defaults to the large configuration: 256 cell channels, six embedding stages, four readout tokens, 24 ICL layers of width 1024, and 16 query heads. Cached query attention uses only two KV heads, each of width 64. Medium also uses reduced cached KV heads. Small has 12 layers of width 512 and full KV heads.

KumoRelational uses a TabICLv2-style row embedding, one shared invariant GNN repeated for `num_hops`, and a TabICLv2 ICL block: 12 layers of width 512 with eight heads. Relational ICL supports hierarchical classification, which requires class IDs, singleton leaves, and head outputs to return to the same entry device. Placement explicitly preserves this.

Parameter counts from meta-device construction at the baseline revision, classification task only:

| Model | Row encoder | Projection | GNN | ICL | Total FP32 parameter size |
|---|---:|---:|---:|---:|---:|
| KumoTabular large | 9,964,560 | 0 | — | 203,703,690 | 815.08 MiB |
| KumoRelational | 1,275,768 | — | 2,363,392 | 26,275,850 | 114.12 MiB |

These weight sizes are modest relative to common GPU memory. Context cache and feature/graph activations can dominate. Splitting weights alone therefore does not guarantee a useful capacity improvement. For cached KumoTabular-large ICL, approximate KV storage per member is `2 × 24 × R × 2 × 64 × element_size`; BF16 at 32,768 context rows is 384 MiB. Relational full-head ICL is `2 × 12 × R × 512 × element_size`; BF16 at the model's 20,000-row cap is about 468.75 MiB. Both formulas exclude encoder caches, ensembles, graph state, and workspaces.

The cell/row encoder alternates attention across context rows and across columns. Rows are not independent during fit. A naive row split with independent local column attention changes the model. During prediction, fitted induced column state makes query rows independent; query partition then belongs to the data-parallel approach.

## Other viable designs and rejected shortcuts

| Approach | Exact semantics possible? | Concrete implementation path | Why not first prototype |
|---|---|---|---|
| Head/tensor parallel ICL | Yes, with rounding differences | Shard Q/K/V heads and FFN intermediate features, sum output projections; keep residual/norm inputs replicated | Roughly two all-reduces per ICL layer; 24 layers means many latency-sensitive collectives. Large Kumo has only two cached KV heads, so 4-way head partition must replicate KV or redesign grouping. Native GEMMs already small compared with language models. |
| Pipeline microbatches | Yes for cached independent queries | Build on contiguous layer placement; stage-local queues and CUDA events; overlap different query batches | Needs multiple in-flight batches, balanced stage work, and careful cache lifetime. Cold fit has no corresponding independent-query pipeline and GPU0 front-end can bottleneck. |
| Relational table encoder fan-out | Yes | Complete target propagation, relative-time standardization, and table preprocessing centrally; send each table's encoder call to a stable worker; gather row embeddings before GNN | Number and sizes of tables can be badly imbalanced. Must preserve generator consumption and per-table fitted column caches. Promising when several similarly expensive tables dominate profiling. |
| Feature/column partition | Yes with communication | Shard cell embedding and independent feature-wise induced attention; exchange feature states before every row-attention block, maintaining global RoPE feature positions | Hidden feature state is `R × C × D`, far larger than the `R × K × D` readout boundary. Six exchanges can dominate; independently processing feature shards drops cross-feature interactions. |
| Graph destination partition | Yes | Partition destination nodes; replicate or exchange source embeddings/halos each hop; keep original node/edge identities and edge-type randomness | More promising for large graphs than small sampled ego graphs. Every hop needs all needed incoming source states. Naively partitioning edges and averaging worker outputs is incorrect. |
| Edge-statistic parallel GNN | Yes, with carefully merged sufficient statistics | Reduce count, sum, sum of squares, min, max for each destination, then reconstruct mean/std and apply original nonlinearities | Current GNN uses total/mean/std/min/max. Local standard deviations cannot be averaged; empty nodes, NaNs, variance clamping, and nonlinear activation order must match. Communication is several dense node-feature arrays per hop. |
| Disjoint task-graph batching | Conditional | Partition independent sampled task neighborhoods and run complete pipelines, retaining temporal and ownership rules | Equivalent to data parallel only when graph expansion, target injection, and fitted table preprocessing are unchanged. Splitting an already shared graph can change messages. |
| Shared immutable weights | Yes | Reuse one network per device across multiple fitted contexts; caches remain per fit/member | Saves weight duplication within a device but is not cross-GPU compute parallelism. Composes well with all other methods. |
| Context-row subset ensembles | No relative to full-context E-member baseline | Give each member a TRAIN subset and measure predictive quality against full-context baseline | Changes attention context and preprocessing, so latency improvement is an accuracy/capacity tradeoff, not exact model parallelism. |

No measured failure is claimed for designs that were only inspected. The table identifies semantic blockers or expected communication costs that justify prioritization; hardware measurements are still required to decide practical viability.

## Historical branches inspected

- `prototype/kumotabular-row-partition`, commit `a261577628032feafe298ecb46017cc79831f987`, partitions aligned context rows between ensemble members in `RecipeExecution`. This is an approximate estimator-context experiment, not exact intra-member row-embedding parallelism. Preprocessing occurs after partitioning features, while target transformations are fitted first. Any new benchmark needs to report the resulting quality tradeoff explicitly.
- `aki/kumo-shared-weights`, head `273fdecff409ee1e67a4fbf764f53cab88afd5d2`, shares Kumo networks through the AutoGluon arena registry for sequential bagged children and stores CUDA weights in FP16. The base sharing change is `6b4f0e074`; mixed precision is an additional numerical variable. Sharing is within process/device, not a distributed executor.
- Aki's context-attention approach is being evaluated by the context-parallel agent. It is complementary to stage placement, but this prototype deliberately keeps native attention unchanged so it can be a clean capacity reference.

## Integration finding: cache ownership matters

At baseline, `ICLModel.fit` copies multi-member caches to pinned CPU memory and `ICLModel.predict` unconditionally migrates whole caches to the input device. Merely moving ICL modules to another GPU does not provide a working distributed cache: it either creates a device mismatch or copies all KV state through GPU0, undermining the memory claim. Cache prefetch and stream ownership must follow the producing layer/member placement.

The experimental resident `EnsembleParallel([model])` solves this for the current research path because it calls the native `_forward`/`_forward_batch` with the original device-resident caches. Longer-term SDM integration should expose a generic cache residency/placement policy used by fit, prefetch, stream lifetime tracking, memory accounting, and serialization. The policy belongs outside Kumo model-specific wrappers. A production placement API should use explicit stage modules or execution plans rather than persistent Python hooks.

## Tests and experiment matrix

CPU tests cover classification and regression for both ICL families, one-shot and cached parity, relational hierarchical classification, reduced query KV heads, and full KumoTabular E2 recipe/executor parity. CUDA-only cases cover true two-device cache placement, nondefault caller and setup streams, empty fit output, repeated fit/cache lifetimes, and final output-device restoration. Randomized nonzero ICL parameters avoid vacuous tests from residual branches initialized to zero. Local CPU success does not validate CUDA stream behavior; the GPU cases must pass on the experiment host.

Run `pytest test/research/test_icl_placement.py`. The runner adapter depends on the resident ensemble executor commit `d1775c0f3` (or its integrated equivalent).

| Dimension | Initial required matrix |
|---|---|
| Models | KumoTabular large; KumoRelational |
| Modes | One-replica resident baseline; stage N=2; layers N=2 and N=4 |
| Estimators | E=1 to isolate model placement; E=8 for cache pressure |
| Context | 1,024; 8,192; largest common successful context; relational respects 20,000-row architecture cap |
| Queries | Batch 256 and 1,024, with multiple batches for steady state |
| Precision | Same baseline/candidate parameter and autocast dtype, initially BF16 autocast |
| Evidence | Three or more timed repeats after warmup; synchronized fit/predict latency; per-GPU peak allocation/reservation; actual per-device cache storage; prediction max/mean error and task quality |

Report both equal-workload comparison and capacity frontier. If one GPU OOMs and the placed model succeeds, report increased supported workload; do not fabricate a speedup against an unsuccessful baseline. Include all GPUs in dollar- and GPU-second efficiency even when one replica is used. Stage with N > 2 intentionally leaves middle devices idle and should not be marketed as N-way scaling.

GPU performance and quality results are pending the shared tabular/relational runners. No cloud instances were launched by this workstream.
