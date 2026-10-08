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

These weight sizes are modest relative to common GPU memory. Context cache and feature/graph activations can dominate. Splitting weights alone therefore does not guarantee a useful capacity improvement. For cached KumoTabular-large ICL, approximate KV storage per member is `2 × 24 × R × 2 × 64 × element_size`; BF16 at 32,768 context rows is 384 MiB. Relational full-head ICL is `2 × 12 × R × 512 × element_size`; BF16 at 20,000 task context rows is about 468.75 MiB. Both formulas exclude encoder caches, ensembles, graph state, and workspaces. Relational `max_train_size=20_000` is the per-table row encoder's `max_keys` sampling budget, not an absolute task-context/ICL limit; preserve that budget across arms and report sampled key counts separately.

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
| Context | 1,024; 8,192; largest common successful context; preserve relational per-table 20,000-key sampling policy |
| Queries | Batch 256 and 1,024, with multiple batches for steady state |
| Precision | Same baseline/candidate parameter and autocast dtype, initially BF16 autocast |
| Evidence | Three or more timed repeats after warmup; synchronized fit/predict latency; per-GPU peak allocation/reservation; actual per-device cache storage; prediction max/mean error and task quality |

Report both equal-workload comparison and capacity frontier. If one GPU OOMs and the placed model succeeds, report increased supported workload; do not fabricate a speedup against an unsuccessful baseline. Include all GPUs in dollar- and GPU-second efficiency even when one replica is used. Stage with N > 2 intentionally leaves middle devices idle and should not be marketed as N-way scaling.

## First measured placement comparison

All 36 placement tests passed on the GPU host in 5.88 seconds, including real cross-device execution, nondefault streams, empty-output fit, repeated fits, hierarchical classification, reduced KV heads, and resident E2 execution with both CPU and CUDA inputs.

The first pretrained comparison used one AWS Spot `g6.12xlarge` with four 24 GB L4 GPUs, PCIe topology without CUDA peer access, PyTorch 2.9.1/CUDA 13.0, source `55dba6bb5`, and the common tabular runner. All four arms used KumoTabular-large, Covertype, 1,024 TRAIN context rows, 2,048 identical ordered validation queries, batch 256, E=4, seed/member seed 1729, FP32 parameters and BF16 autocast. Predictions were warmed up once; three complete prediction passes were timed with every participating device synchronized. The reference was the resident one-replica executor on this same L4 host.

| Metric | Resident 1 GPU | Stage 2 GPUs | Layers 2 GPUs | Layers 4 GPUs |
|---|---:|---:|---:|---:|
| Median prediction pass (s) | 1.1039 | 1.1611 | 1.1191 | 1.1456 |
| Median throughput (rows/s) | 1,855.27 | 1,763.91 | 1,830.08 | 1,787.74 |
| Speedup relative to resident baseline | 1.000x | 0.951x | 0.986x | 0.964x |
| Maximum per-GPU fit allocation (GiB) | 1.769 | 1.074 | 1.248 | 0.994 |
| Reduction in that fit peak | — | 39.3% | 29.5% | 43.8% |
| Maximum per-GPU prediction allocation (GiB) | 1.598 | 1.060 | 1.071 | 0.813 |
| Maximum prediction difference | — | 0 | 0 | 0 |
| Byte-identical predictions | reference | yes | yes | yes |

Accuracy was 0.76416015625, log loss 0.5758563280105591, and macro OVR AUROC 0.9474312941515698 in every arm. No change in quality was observed because the complete prediction arrays were identical.

The result supports a memory benefit with a small latency penalty, not throughput scaling: warm prediction throughput decreased by 1.4–4.9%. Single cold fit measurements were 1.674/1.222/1.219/1.321 seconds; these are not repeated cold-start estimates and should not be interpreted as established fit speedups. Order-dependent initialization can affect them.

The fitted cache occupied 512,754,868 bytes in every arm. Stage placement moved only 50,331,648 bytes of ICL cache to GPU1; 462,423,220 bytes remained on GPU0. Thus approximately 90% of the cache at this small context belongs outside ICL. Layer placement divided that ICL component equally while retaining the encoder state on GPU0. At longer contexts, ICL cache grows with context length while induced encoder cache follows different scaling, motivating the 16k/32k follow-up.

Each result directory retains configuration, runtime and source identity, three raw prediction times, per-batch latency, per-GPU allocated/reserved memory, cache storage by physical device, query identities, targets, predictions, and sampled utilization/power. Local evidence root: `.kumo-multigpu-20261008/placement/host2/placement-large-e4-c1024-q2048-v1-{ep1,stage2,layers2,layers4}`. The GPU test log is `placement-tests-55dba6bb5.log` in the same root. These results are not compared by ratio with the separate L40S throughput host.

## Longer contexts and native relational measurements

The Covertype comparison was repeated at 16,384 TRAIN rows with the same 2,048 queries, E=4, batch 256, precision, and L4 hardware. Source was `1686803e4`. All predictions remained byte-identical; validation accuracy was 0.91455078125, log loss 0.22567126154899597, and OVR AUROC 0.9925490711478547.

| Covertype C=16,384 | Resident 1 GPU | Stage 2 GPUs | Layers 2 GPUs | Layers 4 GPUs |
|---|---:|---:|---:|---:|
| Median throughput (rows/s) | 1,865.11 | 1,830.66 | 1,754.70 | 1,809.19 |
| Relative throughput | 1.000x | 0.982x | 0.941x | 0.970x |
| Maximum per-GPU fit allocation (GiB) | 3.049 | 2.085 | 2.264 | 1.876 |
| Maximum per-GPU prediction allocation (GiB) | 2.187 | 1.764 | 1.309 | 0.876 |

Four-layer-device placement reduced the maximum prediction allocation by 59.9% and fit allocation by 38.5%, with a 3.0% throughput penalty. Total logical cache and unique backing storage both equaled 1,132,463,344 bytes. A separate compact-cache arm therefore correctly made no storage reduction on this workload and produced identical predictions. This differs from the small-context case, where oversized backing views existed; compaction benefit depends on actual cache layout.

| Cache configuration | Logical bytes, summed over members | Unique backing bytes | Physical bytes by GPU in stage mode | Physical bytes by GPU in four-device layer mode |
|---|---:|---:|---|---|
| Covertype C=1,024, E=4 | 358,614,196 | 512,754,868 | `[462423220, 50331648]` | `[475006132, 12582912, 12582912, 12582912]` |
| Covertype C=16,384, E=4 | 1,132,463,344 | 1,132,463,344 | `[327156976, 805306368]` | `[528483568, 201326592, 201326592, 201326592]` |

The non-ICL cache changed between contexts because fitted preprocessing and batching affect its dimensions and backing allocation; do not extrapolate the measured C=1,024 encoder-cache byte count as a fixed constant for every context. All arms within each workload had the same total cache bytes.

Native KumoRelational was then measured on `rel-hm/user-churn`: 16,384 TRAIN context rows, 4,096 ordered validation queries, eight batches of 512, two-hop `[16,16]` temporal-last sampling, E=4, BF16 autocast, and the same L4 host. The context graph contained 319,383 sampled nodes: 142,681 article, 16,384 customer, and 160,318 transaction rows. Source was `1686803e4`, with phase-accounting-fixed runner `36b4f6fe5`; the runner SHA is recorded in each result. Graph SHA256 was independently matched across local and remote copies before the accepted four-arm comparison. An earlier baseline overlapping graph transfer is retained as `v1` but excluded; accepted comparisons use `v2` only.

| Native H&M C=16,384 | Resident 1 GPU | Stage 2 GPUs | Layers 2 GPUs | Layers 4 GPUs |
|---|---:|---:|---:|---:|
| Median throughput (rows/s) | 1,050.73 | 1,066.06 | 1,060.83 | 1,057.60 |
| Relative throughput | 1.000x | 1.015x | 1.010x | 1.007x |
| Maximum per-GPU fit allocation (GiB) | 5.322 | 4.061 | 4.687 | 4.374 |
| Maximum per-GPU prediction allocation (GiB) | 2.111 | 1.650 | 1.293 | 0.886 |

All four 4,096-by-2 prediction arrays were byte-identical. Accuracy was 0.80810546875, positive-class-1 AUROC 0.670048930298348, and runner log loss 0.46131277084350586. The small throughput differences are not convincing evidence of acceleration. Stage placement achieved the lowest fit peak, a 23.7% reduction, because the encoder/GNN remained on GPU0 without any ICL cache there. Layer placement achieved the lowest prediction peak, a 58.0% reduction, by spreading the ICL cache. This illustrates why fit and prediction need separate memory measurements.

The relational executor reported 1,636,581,408 logical cache bytes in every arm. This runner revision did not separately record unique physical cache storage, so no measured physical-cache total is inferred from the logical number. Larger-context follow-ups add that measurement. Relational joins used SDM's CPU fallback because cuDF was unavailable; the runtime used eight CPU threads. The measured end-to-end latency includes that fallback and host/device synchronization.

Independent audits passed input/graph identities, member seeds, prediction columns, prediction hashes, numerical equality, and recomputed performance/quality for both tables. Evidence directories are `placement-large-e4-c16384-q2048-v1-*` and `placement-hm-e4-c16384-q4096-v2-*` beneath the same local evidence root. Each contains an independent quality-audit sidecar.

## Capacity stress and allocator sensitivity

The next native H&M workload used 65,536 TRAIN context rows and the same 4,096 validation queries, eight batches of 512, E=4, precision, and two-hop sampling. Its context graph contained 1,279,607 nodes (571,241 article, 65,536 customer, 642,830 transaction). The per-table row-encoder 20,000-key cap remained unchanged; the final ICL still saw all 65,536 context rows. Library source was `4e1c5c33d`, with versioned runner `b9026ea3f` recording both logical cache bytes and deduplicated backing storage.

Each attempt used a fresh process with a hard timeout and retained stdout, child exit status, and an attempt receipt, including failed attempts. No same-process `empty_cache` retry was used. The L4 exposed 22.04 GiB usable memory, rather than the nominal product capacity.

| Native H&M C=65,536, E=4 | Resident 1 GPU, default allocator | Layers 4 GPUs, default | Stage 2 GPUs, default | Resident 1 GPU, expandable segments | Layers 4 GPUs, expandable segments |
|---|---:|---:|---:|---:|---:|
| Outcome | OOM | OOM | success | success | success |
| Median throughput (rows/s) | — | — | 955.32 | 921.47 | 944.61 |
| Maximum per-GPU fit allocation (GiB) | — | — | 16.105 | 20.732 | 17.259 |
| Maximum per-GPU prediction allocation (GiB) | — | — | 6.150 | 6.607 | 2.010 |
| Fit time (s), single cold observation | — | — | 53.286 | 54.025 | 54.171 |

Both default-allocator failures occurred when `segment_multi_reduce` allocated a 6.10 GiB GNN statistics tensor, before any remedy from sharding later ICL layers. The baseline reported 10.40 GiB allocated and 5.86 GiB reserved but unused; four-device layer placement reported 9.18 GiB allocated and 9.31 GiB reserved but unused. These are **allocator-sensitive OOMs, not demonstrated fundamental capacity limits**: a fresh baseline with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` succeeded. Stage placement also succeeded without that allocator change because it moved all retained ICL state off the GNN device. This is a useful operational result, but it does not establish that multi-GPU was necessary at E=4.

All successful arms retained 6,468,419,616 cache bytes, with logical bytes equal to unique backing bytes. Stage placement split these as `[25968672, 6442450944]`; layer placement split them as `[1636581408, 1610612736, 1610612736, 1610612736]`. The dominant 6 GiB ICL cache was therefore actually distributed, not merely attributed to nominal executor owners.

The expandable single-GPU and default stage predictions were byte-identical, with positive-class AUROC 0.666514263969927 and runner log loss 0.4616449475288391. The expandable four-device layer arm was **not** byte-identical: maximum absolute prediction difference was 0.010411202907562256, AUROC was 0.6668017727143439, and log loss was 0.4615814983844757. Accuracy was 0.80810546875 in all three. Independent audit measured mean absolute difference 0.001082822, p99 0.006126165, relative L2 error 0.0031663, and no predicted-class flips. The log-loss delta was -0.000063467 with an entity-bootstrap interval [-0.000206287, +0.0000774024], which does not establish a quality improvement. Inputs, graph, member seeds, columns, and artifact hashes matched.

The placement algorithm preserves attention and model dependencies, but this long-context BF16 observation prevents a blanket claim of bitwise equality. Changed layout or kernel execution is a hypothesis, not an isolated cause; tolerances and task-quality checks must gate deployment. SDM's chunk memory limit uses total device capacity and a fixed fraction, not currently free memory, so the observed drift cannot simply be attributed to memory-pressure-driven chunk sizing.

The 6.10 GiB statistics allocation suggests a different optimization boundary: destination-node blocks can compute the existing total/mean/std/min/max statistics and immediately apply their projection, avoiding the full dense intermediate. Such a fused or chunked GNN path was not implemented here and would require dedicated parity tests for variance, empty segments, and graph identity. ICL placement alone cannot remove this allocation.

Evidence is retained under `capacity-hm-c65536-e4-v1-{ep1,layers4,stage2}` and `capacity-hm-c65536-e4-expandable-v1-{ep1,layers4}`. All five attempts are downloaded locally; all three successful arms have independent audit sidecars.

### Pending remote evidence at the access interruption

An E=8, C=65,536 follow-up launched two fresh expandable-allocator attempts, resident EP1 then stage2, with a 150-second deadline per child. An observed remote EP1 stderr tail reported an OOM requesting 1.22 GiB at a linear projection: 21.02 GiB was allocated, only 198.42 MiB reserved but unused, and 585.12 MiB free. This is stronger evidence of live capacity pressure than the fragmented E=4 failures, but its receipt has not yet been downloaded.

The enclosing SSH session (local process handle `90036`) subsequently completed with exit code zero and printed both arm-completion markers. **The stage2 outcome remains unverified:** the wrapper deliberately returns normally even if a child fails or times out, so the shell exit code is not a benchmark success signal. Both attempts are remote at `/home/ubuntu/kumo-multigpu/results/capacity-hm-c65536-e8-expandable-v1-{ep1,stage2}` on the four-L4 host `98.93.170.140`. Source was `4e1c5c33d`; the runner was `relational_bench-b9026ea3f.py`, wrapper `capacity_attempt-2361cb51c.py`, and input was `workload-hm-c65536-b512`.

A public native single-GPU E=8 control with CPU-offloaded caches was authorized but **not launched** before remote access was restricted. Therefore the experiment does not establish that two GPUs are required: CPU-offload may fit where the resident single-GPU executor fails, at a different transfer/performance cost. Native and resident paths also need their preprocessing/member-randomness equivalence checked before any quality comparison.

Separate Nsight captures of the Covertype C=1,024 E=4 resident EP1 and four-device layer arms completed remotely before the interruption. Reports are `/home/ubuntu/kumo-multigpu/results/nsys-placement-c1024-v1-{ep1,layers4}.nsys-rep`; corresponding child results are `profile-placement-c1024-v1-{ep1,layers4}`. No report or SQLite export was downloaded. The attempted export SSH call was aborted, so no export completion is claimed. Transfer/kernel/synchronization attribution is therefore pending; the unprofiled throughput tables above do not depend on these captures. When recovered, analyze CUDA activity within the main-thread `prediction_pass` NVTX time span, including worker-thread launches, and distinguish summed kernel durations from wall time.

After network access was restored, the cloud operator found the old instance absent and no surviving tagged EBS volumes; its historical Spot record was also unavailable. The remote-only E=8 receipts and placement Nsight captures therefore could not be recovered. The E=8 stage2 outcome remains unknown, and no transfer attribution can be reported from those lost captures. Any replacement-host attempt must be a new experiment with fresh controls, not a recovered completion of the old arm. Downloaded E=4 evidence remains intact. No cloud instances were launched by this workstream.
