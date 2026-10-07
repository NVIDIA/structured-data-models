# Kumo multi-GPU investigation

Status: investigation in progress, 2026-10-08. Implementation baseline: `842c408fe`. This report separates measured results from architectural expectations. Blank result cells mean that a measurement has not arrived; they do not mean zero, parity, or success.

The goal is practical multi-GPU inference for both `KumoTabular` and `KumoRelational`, including throughput, latency, capacity, prediction quality, and a lightweight integration into SDM. The implementation recommendations are in [integration.md](integration.md).

The strongest current throughput result is persistent process query DP with local estimator batching: KumoTabular reaches 13,991 rows/s on four L40S GPUs, 3.817× its matched one-process control and 3.492× the best native one-GPU result. Native relational H&M reaches 3,859 rows/s, 3.254× its process control and 2.981× public native. Predictions remain byte-identical to the matching native batching policy. Batched KumoTabular ensemble parallelism also scales 1.836× on two GPUs. Initial threaded small workloads scale poorly; sequential layer placement improves capacity; cached context parallelism saves KV memory but its current BF16 relational regression fails the declared full-quantile numerical gate. Final CPU concatenation is outside these process timers, so cross-runner application-boundary comparisons need that caveat.

## Approach matrix

| Approach | Work placement | Expected benefit | Main limitation / correctness requirement | Current owner and evidence |
|---|---|---|---|---|
| Query data parallelism (DP) | Complete, fixed query batches across persistent replicas | High aggregate throughput; no attention collectives | Full model and fitted context replicated; preserve batch boundaries and complete relational neighborhoods | Tuned process tabular DP achieves 3.817× and native H&M 3.254× at four GPUs; initial threaded results weak; hybrid follow-ups pending |
| Ensemble parallelism (EP) | Fixed logical estimator IDs across replicas | Reduce member work and member cache per GPU; small output gather | Same recipes, member RNG, class alignment, output reduction, and cache residency as reference | Tested: initial unbatched scaling poor; batched E8 gives 1.836× on two GPUs with exact predictions; four-GPU numerical gate fails |
| Cached context parallelism (CP) | ICL KV rows across ranks; queries replicated | Larger retained context; possible attention acceleration | Full context compute still replicated at fit; stable softmax reduction, global length scaling, uneven shards; collectives every layer | Real NCCL tests passed; 1k and resident 16k tabular runs slower; BF16 F1 all-quantile gate fails |
| Layer/stage placement | Row encoder, GNN, and/or ICL layers on different devices | Parameter/cache capacity; potential pipeline overlap across query batches | Single-query latency can worsen; cache ownership and transferred activations must follow stages | Source `55dba6bb5`: 36 CPU/CUDA tests passed on L4; real model memory decreases with small throughput cost; sequential placement, not overlapped pipelining |
| Table embedding parallelism | Independent related-table encoders across devices, then gather row embeddings | Parallel relational table encoding before GNN | Tables may be imbalanced; shared preprocessing/target propagation and per-table RNG must remain fixed | Design inspected; no measurement |
| Tensor parallelism (TP) | Projection/MLP widths or attention heads across ranks | Parameter and compute sharding for a single member | Many collectives, packed projections/custom kernels, small widths and GQA limit useful partitions | Design inspected; implementation not yet established |
| Full context fit sharding | Row encoder inducing attention and context self-attention distributed during fit | Reduce peak fit memory, not only retained cache | Requires global induced-state/attention reductions in both row encoder and ICL | Design inspected; separate from cached CP |
| Relational graph partition | Nodes/edges and GNN state split with boundary exchange each hop | Larger graph capacity | Exact halo/state exchange, relation types, target ownership, temporal sampling; irregular load balance | Design inspected; no exact implementation or result |
| Context-subset ensemble | Different context rows assigned to independent members | Less fit/cache work per member | Changes model input and potentially prediction quality; not exact CP | Existing `prototype/kumotabular-row-partition` inspected; separate quality arm |
| EP × DP / EP × CP | Process groups split along two axes | Balance query volume, ensemble width, and context capacity | Product of group sizes must equal GPUs; group-local output order and RNG | Integration design; measurements pending |

No approach is declared a speedup merely because its implementation runs. Capacity gains, single-request latency, and aggregate throughput are separate outcomes.

## Existing work inspected

| Source | What it contributes | What it does not establish |
|---|---|---|
| `feature/tabfm-ensemble-parallelism` at `81b8f7fc4` | Explicit replica executor; stable member/device/cache affinity; placement-independent member seeds; ordered aggregation; prior TabFM diagnostic | Current Kumo integration or performance on today's estimator batching/cache implementation |
| Aki `aki/cp` at `499485af5` | Partial attention with log-sum-exp; maximum plus two sum all-reduces; cached TabICLv2 ICL sharding | Distributed fit, native KumoTabular integration, multi-rank GPU validation |
| User example commit `6aa9f4f14` | `examples/kumo/relational/multi_gpu.py`: persistent process per GPU, seeded local fit, fixed complete query graphs, indexed merge | Real RelBench quality/scaling; example uses tiny synthetic inputs |
| `prototype/kumotabular-row-partition` | Randomly partitions aligned context rows among ensemble members and fits member-specific feature transforms | Full-context prediction equivalence; it changes the statistical method |
| Shared weights commit `6b4f0e074` | Reuses KumoTabular weights across sequential fits in one process through the arena adapter | Cross-GPU shared physical weights or distributed inference |

The previous TabFM diagnostic reported 224.98 to 421.99 rows/s on one versus two L4 GPUs (1.876×), with identical prediction bytes. It used E8, 1,024 context rows, 2,000 queries, batch 250, and a target-free materialization of `rel-hm/user-churn` two-hop `[16,16]` neighborhoods. It was one exploratory run, not a Kumo result, repeated trial, or quality evaluation. These values are historical conversation/report evidence and must not populate this investigation's Kumo result rows.

## Comparison contract

1. Preserve weights, model size, dtype, recipe, context row identities/order, estimator count and logical member seeds, query identities/order, batch boundaries, and relational sampled neighborhoods across GPU counts.
2. Run a public one-GPU baseline and a one-GPU version of each executor. Current `ICLModel.fit` offloads CUDA multi-estimator caches to pinned CPU; a resident EP executor can improve transfers even on one GPU. Report that gain separately from concurrency. Preprocessing must run on the same device for EP1/EP2/EP4; if native preprocessing uses another device, identify that additional difference explicitly.
3. Compare against feasible `estimator_batch_size > 1` for KumoTabular. Current SDM can batch compatible members; relational members remain separate. An old sequential baseline overstates gains against today's best one-GPU execution.
4. Separate startup, checkpoint load, sampling, preprocessing, fit, warmup, synchronized prediction, result gather, and score calculation. Warm throughput excludes profiler overhead. End-to-end latency includes the actual application boundary.
5. Use at least three measured repeats after warmup, interleave configurations where practical, and retain individual samples. Report median/range or uncertainty with sample count; eight batches within one fit are not eight independent fit trials.
6. Strong scaling keeps workload fixed on 1/2/4 GPUs. Weak scaling increases query work with GPU count and labels that change. Context capacity tests increase context until OOM or a documented limit; OOM is a result, not a reason to silently shrink context.
7. Report per-GPU and aggregate parameters, fitted cache bytes (GPU and CPU), peak allocated/reserved memory, host RSS, and observed device memory. Reset phase peaks appropriately; allocator reservation is not live tensor memory.
8. Evaluate quality on the identical ordered validation target vector after inference. Classification needs ROC-AUC/AP where defined, log loss, and prediction/label agreement; regression needs RMSE/MAE and quantile metrics when applicable. Report maximum/mean prediction error and tolerances separately from aggregate quality. TEST data is unnecessary.
9. Preserve exact temporal cutoffs, two-hop neighbor policy, selected row IDs, and graph hashes. For relational DP, slicing every related table by task-row index is incorrect. Freeze the full sampled neighborhood for each task batch.
10. Record hardware topology, peer access, GPU count used versus allocated, software versions, checkpoint/data/source IDs, configuration, wall time, and raw outputs. Dollar efficiency should reflect the complete EC2 instance while GPU-hour efficiency uses GPUs actually exercised.

## Results

The first native and 1/2/4-GPU EP ladder has completed. For the small initial workload, thread-based EP scales poorly: two GPUs improve only 8.9% over the same resident executor on one GPU, and four GPUs are 4.9% slower. Native estimator batching on one GPU reaches 4,007 rows/s, more than twice the initial two-GPU EP throughput. This is a measured result for this workload and executor, not a rejection of ensemble parallelism. Three repeats reuse one fit, so they characterize warm prediction variability rather than independent fits.

| Hardware ladder | Instance / topology | Runtime | Scope |
|---|---|---|---|
| Host 1 | Spot `g6e.12xlarge`, 4× L40S, 46,068 MiB each; all pairwise links reported `NODE` (PCIe host bridges, no NVLink); CUDA peer-access checks false between devices | Python 3.12.3, PyTorch 2.9.1+cu130, CUDA 13.0, driver 595.91.07 | Main tabular and relational EP/DP comparisons |
| Host 2 | Spot `g6.12xlarge`, 4× L4, 23,034 MiB each; all pairwise links `NODE` (no NVLink) | PyTorch 2.9.1+cu130, CUDA 13.0, driver 595.91.07 | Separate CP/placement ladder; do not combine raw speedup denominators with Host 1 |

Eight-GPU/NVSwitch capacity was not acquired: the operator reports A100-family launch denied by an organization policy and eight-GPU G-family shapes blocked by a regional 64-vCPU Spot quota (192 vCPUs required). This limits the hardware/topology scope, not the validity of any algorithm.

Prepared source workloads use fixed nested TRAIN/validation subsets, recorded in their manifests. Available rows are distinct from rows actually measured in any run:

| Model family | Workload | Available TRAIN / validation | Planned context range | Purpose |
|---|---|---:|---|---|
| Tabular classification | Covertype, 54 input columns | 464,809 / 116,203 | 1,024–65,536 | Large tabular throughput/context scaling; seven-class quality |
| Tabular regression | California housing, 8 columns | 16,512 / 4,128 | 1,024–16,384 | Regression and complete 999-quantile output behavior |
| Relational classification | `rel-hm/user-churn`, native 3-table database | 3,832,692 / 76,556 | 1,024–65,536 | Native two-hop `[16,16]` temporal relational scaling |
| Relational regression | `rel-f1/driver-position` | 7,453 / 499 | 1,024–4,096 | Regression and structurally different relational graph |

Tabular splits are custom deterministic 80/20 splits (seed `20261008`), not official test evaluations. RelBench retains native TRAIN/validation splits. The October KumoRelational experiment uses native related tables and should not be described as the earlier TabFM flattened two-hop representation.

| Model / dataset | Method | GPUs | Context / E / query / batch | Warm rows/s | Speedup vs same executor 1 GPU | Cold seconds | Peak GPU / aggregate memory | Quality / max prediction error | Evidence |
|---|---|---:|---|---:|---:|---:|---|---|---|
| KumoTabular large / Covertype | Public baseline, L40S | 1 | 1,024 / 4 / 2,048 / 256 | 1,818.93 median (1,807.68–1,824.77, n=3) | — | Fit 122.751, load 4.615 | Fit peak allocated 1.408 GiB; prediction 1.284 GiB; CPU cache 342.001 MiB | Accuracy 0.764160; log loss 0.576045; OVR AUC 0.947415; repeat max difference 0 | `tabular-native-large-e4-c1024-q2048/result.json`, source `9f0d6c765` |
| KumoTabular large / Covertype | Resident EP, L40S | 1 | 1,024 / 4 / 2,048 / 256 | 1,729.77 | 1.000× | Fit 1.075 (warmed environment) | Prediction peak 1.598 GiB | Exact native predictions | `tabular-ep1-large-e4-c1024-q2048`, `6c3f1e22e` |
| KumoTabular large / Covertype | Resident EP, L40S | 2 | 1,024 / 4 / 2,048 / 256 | 1,883.05 | 1.089× | Fit 1.114 | Prediction peak max 1.358 GiB/GPU; summed rank peaks 2.714 GiB | Max prediction difference 0 | `tabular-ep2-large-e4-c1024-q2048`, `6c3f1e22e` |
| KumoTabular large / Covertype | Resident EP, L40S | 4 | 1,024 / 4 / 2,048 / 256 | 1,644.81 | 0.951× | Fit 1.298 | Prediction peak max 1.176 GiB/GPU; summed rank peaks 4.699 GiB | Max prediction difference 0 | `tabular-ep4-large-e4-c1024-q2048`, `6c3f1e22e` |

The first native fit took 122.751 seconds; a repeated native E4 fit with estimator batch size one took 1.327 seconds. Thus the cold first fit is not an appropriate denominator for an EP fit speedup. First-use compilation/library initialization remains a hypothesis for the cold delay. The result's CPU-offloaded cache is explicitly recorded; EP comparisons require their resident one-GPU control. Independent quality review confirmed the baseline metrics and stable repeat outputs.

EP batch p50 was 147.96 / 135.90 / 155.55 ms at 1/2/4 GPUs. Scaling efficiency was 54.4% at two GPUs and 23.8% at four. All four saved prediction arrays, including native, are byte-identical (`92cb62ec…`). EP resident cache storage distributes as 489.00 / 244.50 / 122.25 MiB per GPU, while replicated parameters make aggregate device memory increase. Summed per-rank peaks are an upper bound on simultaneous aggregate use, not a synchronized aggregate trace.

The EP cache's 489 MiB unique backing storage exceeds the public CPU cache's 342 MiB, despite identical predictions. Source inspection identified row-encoder value views retaining fused KV backing buffers; a compact-cache clone prototype passed reduced real-model CPU parity, with GPU measurements pending. CPU dispatch/launch overhead and short member work remain candidate explanations for poor initial scaling, pending profiles. The historical TabFM 1.876× result does not predict these Kumo results.

The tuned one-GPU comparison uses the same large model, Covertype rows, E4, context 1,024, query 2,048, query batch 256, and BF16 autocast. Only estimator batching changes:

| Native estimator batch | Median rows/s | Batch p50 | Fit seconds, warmed environment | Accuracy | Log loss | Max probability difference vs estimator batch 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,758.63 | 145.36 ms | 1.327 | 0.764160 | 0.576045 | 0 |
| 2 | 2,984.01 | 85.76 ms | 1.253 | 0.764160 | 0.575996 | 0.008804 |
| 4 | 4,007.14 | 63.82 ms | 1.255 | 0.764160 | 0.575978 | 0.005607 |

Predicted class labels are unchanged for all 2,048 rows. Estimator-batched BF16 outputs are not byte-identical; mean absolute probability differences are about 0.000263 and 0.000259. These quality changes must remain visible when comparing speed against the sequential member path. Best measured native batching is 2.28× the repeated unbatched native run and 2.13× the initial two-GPU EP. Multi-GPU follow-ups must include local estimator batching before claiming a useful advantage.

### Batched ensemble follow-up on L40S

The larger workload uses KumoTabular large, Covertype, **E8, context 4,096, query 8,192, batch 1,024**, BF16 autocast, three measured passes, and source `1686803e4`. The native reference already uses estimator batch size eight. The resident batched executor distributes the same members and batches compatible local members.

| Arm | GPUs | Median rows/s | Speedup vs resident EP1 | Fit seconds | Max prediction peak per GPU | Prediction status |
|---|---:|---:|---:|---:|---:|---|
| Native, estimator batch 8 | 1 | 3,219.44 | — | 2.749 | 3.743 GiB | Reference |
| Resident batched EP | 1 | 4,013.81 | 1.000× | 2.077 | 3.503 GiB | Byte-identical to native |
| Resident batched EP | 2 | 7,368.45 | **1.836×** | 1.648 | 2.405 GiB | Byte-identical to native and EP1 |
| Resident batched EP | 4 | 5,981.13 | 1.490× | 1.477 | 1.612 GiB | Fails declared numerical gate; see below |

The two-GPU result is 91.8% scaling efficiency relative to resident EP1. Its 2.289× speedup over public native combines 1.247× from residency/scheduling with the separate 1.836× multi-GPU gain. Native/EP1/EP2 accuracy is 0.841797, log loss 0.413884, and multiclass AUC 0.974934. These are validation-prefix results, not a full dataset ranking.

The four-GPU result is slower than two GPUs and differs in one of 57,344 probability entries beyond the predeclared BF16 tolerance; maximum difference is 0.032253, with six changed labels. Accuracy is 0.841919 and log loss 0.413924, with a paired log-loss change interval spanning zero. Similar aggregate quality does not turn a failed numerical gate into a pass. Local estimator groups shrink from eight to four to two across placements; a one-GPU batch-two control is queued to distinguish batching arithmetic from placement effects. Keep the original failure visible.

### Process query DP on L40S

One persistent process per GPU changes the result for the original small Covertype workload (large model, E4, context 1,024, queries 2,048, batch 256). Each worker uses one CPU thread and the native unbatched estimator path. The timer is parent wall time including input IPC, transfers, inference, and CPU prediction return; final CPU concatenation/scoring are excluded consistently.

| Processes / GPUs | Median rows/s | Speedup vs DP1 | Scaling efficiency | Spawn + load + fit | Peak prediction allocation per GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,780.50 | 1.000× | 100% | 5.092 s | 1.284 GiB |
| 2 | 3,418.27 | 1.920× | 96.0% | 5.231 s | 1.284 GiB |
| 4 | 6,650.11 | **3.735×** | **93.4%** | 5.521 s | 1.284 GiB |

All predictions are byte-identical to unbatched native, and the independent audit verified full input/row identities. Model/cache storage and roughly 2.07 GiB of host RSS per worker are replicated. This unbatched four-GPU throughput is 1.66× the best measured one-GPU estimator-batched baseline (4,007 rows/s), not 3.735× that stronger baseline. The contrast with weak threaded scheduling supports process isolation as practical, but CPU thread-count differences and profiler coverage prevent attributing the entire effect to the Python GIL alone.

The follow-up combines the same process topology with **local estimator batch size four**; context/query batches and one CPU thread per worker remain unchanged:

| Arm | GPUs | Median rows/s | Speedup vs matched DP1 | Spawn + load + fit | Peak prediction allocation per GPU |
|---|---:|---:|---:|---:|---:|
| Tuned native, estimator batch 4 | 1 | 4,007.14 | — | Fit 1.255 s, excluding spawn/load | 1.404 GiB |
| Process DP, estimator batch 4 | 1 | 3,665.05 | 1.000× | 5.013 s | 1.404 GiB |
| Process DP, estimator batch 4 | 4 | **13,990.88** | **3.817×** | 5.469 s | 1.404 GiB |

Four-GPU scaling efficiency is 95.4%, and throughput is 3.492× tuned public native. Both process arms are byte-identical to the **estimator-batch-four** native reference (not the unbatched native reference); accuracy is 0.764160 and log loss 0.575978. The measured process boundary includes input/output IPC and GPU transfers but ends before final CPU concatenation. It must not be silently equated with later runners that time the final gather; that missing duration has not yet been measured for these attempts.

Weak scaling keeps 4,096 query rows per GPU while context 1,024, E4, estimator batch four, and query batch 256 stay fixed:

| GPUs | Total queries | Median rows/s | Rows/s per GPU | Weak efficiency vs DP1 |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 3,782.40 | 3,782.40 | 100% |
| 2 | 8,192 | 7,344.95 | 3,672.47 | 97.1% |
| 4 | 16,384 | 14,682.72 | 3,670.68 | 97.0% |

Prediction peak stays 1.404 GiB per GPU. Independent checks found exact outputs on the shared 4,096-row prefix and exact agreement with tuned native on its 2,048-row prefix. These arms score different full validation cohorts; their whole-cohort accuracy differences are not evidence of GPU-induced quality change. A one-GPU 16,384-row reference is still needed for full-cohort placement equivalence. This is weak scaling, not a fixed-workload 3.882× speedup.

### What the initial Nsight trace explains

The explicit `prediction_pass` ranges show many short kernels and little overlap in unbatched EP4. Unlike the main-thread PyTorch trace, Nsight captured all worker GPUs. Instrumented durations are diagnostic and are not clean benchmark throughput.

| Trace finding | Native unbatched 1 GPU | Unbatched EP4 |
|---|---:|---:|
| Kernel launches | 40,664 | 48,552 |
| Per-device compute-kernel active fraction | 12.26% | 3.82–4.09% |
| At least two GPUs computing concurrently | — | 1.085% of inference range |
| All four GPUs computing concurrently | — | 0.00148% |
| H2D bytes in range | 2.869 GB | 1.377 MB |

Residency removes almost all cache H2D traffic, but does not fix short-kernel launch scheduling. Of 7,888 additional EP kernels, 7,840 are float-to-BF16 conversions and 48 integer additions; kernel-name accounting does not support calling them cache-view packing. Worker autocast-context lifetime is a specific hypothesis being tested. No-compute intervals can still contain copy-engine work, and these traces do not measure GIL waiting. See [ensemble-profile.md](ensemble-profile.md) for interval-union methodology, API timings, trace limitations, and reproduction.

### Native relational EP and DP on L40S

Both tasks use 1,024 TRAIN observations, E4, BF16 autocast, and fixed native `[16,16]` temporal two-hop graphs. HM queries contain 2,000 rows in batches of 250; F1 uses all 499 validation rows in batches of 125/125/125/124. Throughput includes prepared CPU graph inputs through CPU-complete predictions; sampling is separate. Detailed fit, memory, and profiling findings are in [relational_results.md](relational_results.md).

| Task | Executor | 1 GPU rows/s | 2 GPUs rows/s | 4 GPUs rows/s | Four-GPU speedup vs same executor |
|---|---|---:|---:|---:|---:|
| HM user churn | Ensemble | 1,361.95 | 1,465.28 | 1,577.98 | 1.159× |
| HM user churn | Query DP, threads | 1,350.21 | 1,465.94 | 1,211.90 | 0.898× |
| F1 driver position | Ensemble | 384.04 | 379.23 | 356.47 | 0.928× |
| F1 driver position | Query DP, threads | 357.41 | 377.07 | 296.78 | 0.830× |

Public native throughput was 1,294.66 rows/s (HM) and 371.72 (F1). Every GPU placement within a reference policy preserved prediction bytes; F1 parity includes all 499 × 999 quantiles, with no nonfinite outputs or quantile crossings. HM churn-positive AUC was 0.661948 for native/DP and 0.663524 for EP. F1 MAE/RMSE was 3.410428/4.211033 for native/DP and 3.325543/4.094762 for EP. Native versus EP uses a different preprocessing device/member RNG policy, so these quality differences are not caused by additional GPUs.

EP reduces per-GPU prediction peak allocation from 0.496 to 0.371 GiB on HM and 0.388 to 0.259 GiB on F1. DP duplicates fitted state and its serial replica setup grows: HM fit 1.595 to 4.025 seconds, F1 1.785 to 4.861 seconds at one versus four GPUs. These small workloads do not approach L40S memory capacity.

The runtime lacked cuDF and emitted explicit CPU fallback warnings for string sorting/joins. The main-thread PyTorch EP trace also missed worker CUDA activity. Neither trace absence nor whole-process telemetry should be called proof of GPU inactivity; Nsight, process scheduling, larger workloads, and a separately controlled cuDF runtime are follow-ups. The original HM scorer assumed the second category was positive even when columns were `['1','0']`. Corrected class-1 scores are preserved in `quality-independent-audit.json` alongside unchanged original results.

Persistent-process DP improves the **same native H&M workload** without changing its graphs or fitted policy. Each worker uses eight CPU threads; all repeats are measured after warmup:

| Processes / GPUs | Median rows/s | Speedup vs process DP1 | Spawn + load + fit | Max prediction allocation per GPU |
|---:|---:|---:|---:|---:|
| 1 | 1,186.19 | 1.000× | 5.155 s | 0.459 GiB |
| 2 | 2,216.50 | 1.869× | 5.353 s | 0.459 GiB |
| 4 | **3,859.33** | **3.254×** | 5.658 s | 0.459 GiB |

The four-GPU result is 2.981× public native (1,294.66 rows/s), with 81.3% process scaling efficiency. Every process output is byte-identical to native, including class order, and complete workload hashes match. Churn-positive AUC remains 0.661948, accuracy 0.808, and log loss 0.466849. Whole graph batches travel through IPC; no related-table slicing or graph partition is used. As in the tabular process run, final CPU concatenation is outside this timer and preprocessing uses the existing Arrow/CPU fallback runtime.

### Cached context split on L4

This initial pretrained comparison uses **KumoTabular small**, Covertype, E4, context 1,024, queries 2,048, batch 256, BF16 autocast, and the public CPU cache-offload policy. All controls use the same input hashes and ordered query IDs. Independent recomputation matched each recorded quality metric, and every repeat/rank output hash agrees within its arm.

| Arm | GPUs | Median unique rows/s | ICL KV per rank, CPU | Peak prediction allocation per GPU | Accuracy | Log loss | Max probability difference vs native |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native SDPA | 1 | 3,082.25 | 96 MiB | 265.63 MiB | 0.756836 | 0.586389 | 0 |
| Efficient LSE attention | 1 | 3,085.89 | 96 MiB | 265.63 MiB | 0.755859 | 0.586496 | 0.006886 |
| Context parallel | 2 | 2,648.21 | 48 MiB | 241.13 MiB | 0.755859 | 0.586468 | 0.007744 |
| Context parallel | 4 | 2,576.37 | 24 MiB | 228.88 MiB | 0.754395 | 0.586577 | 0.008601 |

Fit peak remains 327.63 MiB per GPU in every arm: this implementation shards retained KV, not fit computation. Two/four GPUs are approximately 14.1%/16.4% slower than native on this short-context case. There are four/five changed predicted labels out of 2,048 for CP2/CP4; mathematical attention equivalence does not imply identical BF16 predictions. Accuracy changes are reported explicitly, not hidden behind a tolerance.

A FlashAttention LSE variant now preserves native GQA head counts instead of repeating KV heads. Its four-layer random-weight probe at context 1,024 measured native 3.117 ms, Flash LSE1 3.271 ms, and CP4 4.568 ms; the efficient backend's corresponding times were 3.021, 3.296, and 4.681 ms. This is a microbenchmark, not pretrained model quality; the following larger-context experiment tests its full-model behavior.

The resident-cache follow-up increases context to **16,384** and uses **KumoTabular large with two KV heads**, retaining E4, 2,048 queries, batch 256, and L4 hardware:

| Arm | GPUs | Rows/s | Resident cache per rank | ICL portion per rank | Peak prediction allocation per GPU |
|---|---:|---:|---:|---:|---:|
| Native | 1 | 1,887.25 | 1,080 MiB | 768 MiB | 2.189 GiB |
| Flash LSE1 | 1 | 1,824.62 | 1,080 MiB | 768 MiB | 2.189 GiB |
| CP2 | 2 | 1,525.92 | 696 MiB | 384 MiB | 1.816 GiB |
| CP4 | 4 | 1,492.31 | 504 MiB | 192 MiB | 1.628 GiB |

Thus CP also fails to accelerate this longer, resident GQA workload on PCIe. Native/LSE1 predictions are identical; CP2/CP4 maximum probability differences are 0.009093/0.008519, with accuracy 0.914551 in all arms. Independent checks matched actual input hashes, row IDs, and recomputed metrics. Fit peaks are about 2.258 GiB. In general, sharding saved KV after each layer can lower accumulated fit-cache memory; it still replicates full context computation and does not shard the large row/GNN activations. Do not generalize the unchanged fit peak in these cases into a claim that CP can never reduce any fit-memory component.

Native KumoRelational regression on F1 also tested the Flash LSE path, with E4, context 1,024, all 499 validation rows, and fixed two-hop graphs:

| Arm | GPUs | Rows/s | MAE | RMSE | Full 999-quantile numerical gate |
|---|---:|---:|---:|---:|---|
| Native | 1 | 364.23 | 3.414711 | 4.214993 | Reference |
| Flash LSE1 | 1 | 362.35 | 3.414711 | 4.214993 | Exact native predictions |
| Context parallel | 2 | 361.15 | 3.418062 | 4.219112 | **Fail** |
| Context parallel | 4 | 376.00 | 3.416702 | 4.217447 | **Fail** |

Both CP outputs are finite and monotone, and their median predictions pass tolerance, but the complete 499 × 999 arrays fail `atol=0.01, rtol=0.05`: 938 entries fail at two ranks and 915 at four. Maximum absolute error is about 0.20224; mean absolute errors are 0.017043 and 0.014253. The largest relative failures occur in near-zero tail quantiles, not necessarily the largest absolute-error entries. Paired MAE changes are small but consistently positive in the auditor's driver-cluster bootstrap. Consequently the approximately 3.2% CP4 throughput gain is not an accepted equivalent-result win. A higher-precision local-attention/merge arm is being tested with unchanged tolerances.

### Sequential model placement on L4

This separate comparison uses **KumoTabular large**, E4, context 1,024, queries 2,048, batch 256, and the resident executor. Source `55dba6bb5` passed 36 placement tests on the L4 host, including actual CUDA cases. Each candidate saved exactly the same prediction bytes as its resident one-GPU reference.

| Placement | GPUs | Median rows/s | Max fit peak per GPU | Max prediction peak per GPU | Prediction sum of device peaks |
|---|---:|---:|---:|---:|---:|
| Resident reference | 1 | 1,855.27 | 1.769 GiB | 1.598 GiB | 1.598 GiB |
| Encoder / ICL stages | 2 | 1,763.91 | 1.074 GiB | 1.060 GiB | 1.613 GiB |
| Contiguous ICL layers | 2 | 1,830.08 | 1.248 GiB | 1.071 GiB | 1.614 GiB |
| Contiguous ICL layers | 4 | 1,787.74 | 0.994 GiB | 0.813 GiB | 1.645 GiB |

The four-way layer split lowers the busiest device's fit peak by about 44% and prediction peak by 49%, at a 3.6% warm throughput penalty. This is a capacity tradeoff; the implementation does not overlap pipeline microbatches. Sum of device peaks is not necessarily simultaneous aggregate usage. At this short context, 462.4 MB of the 512.8 MB retained cache belongs to the row encoder on GPU0, so distributing ICL layers alone cannot balance all cache memory.

Native H&M relational placement extends context to **16,384**, E4, 4,096 queries, batch 512, fixed `[16,16]` two-hop neighborhoods, BF16, and eight CPU threads on the same L4 host:

| Placement | GPUs | Median rows/s | Max fit peak per GPU | Max prediction peak per GPU |
|---|---:|---:|---:|---:|
| Resident reference | 1 | 1,050.73 | 5.322 GiB | 2.111 GiB |
| Encoder/GNN → ICL stages | 2 | 1,066.06 | **4.061 GiB** | 1.650 GiB |
| Contiguous ICL layers | 2 | 1,060.83 | 4.687 GiB | 1.293 GiB |
| Contiguous ICL layers | 4 | 1,057.60 | 4.374 GiB | **0.886 GiB** |

All four outputs are byte-identical; the audit verified complete workload hashes, columns, and member seeds. Churn-positive AUC is 0.670049, accuracy 0.808105, and log loss 0.461313. Stage placement reduces maximum fit allocation by 23.7%, while four-layer placement reduces maximum prediction allocation by 58.0%; the best split depends on the constrained phase. Throughput differs by only 0.7–1.5%, insufficient here to establish a robust speed advantage. The runtime uses CPU joins because cuDF is absent, which can mask GPU stage-latency changes. The GNN remains on one device; these results do not establish distributed GNN computation or a maximum feasible context frontier.

## Retained evidence

The [initial tabular evidence index](evidence/initial-tabular-20261008/index.json) archives all seven small raw result JSON files in the repository. It binds predictions, row/target identity arrays, and available telemetry to SHA-256 hashes and their original local paths. The index and all 34 original artifacts passed verification at collection; weights and raw datasets are excluded. Large artifacts must remain in the local `.kumo-multigpu-20261008` result store or be copied to a durable user-selected location before that local store is removed. EC2 teardown does not remove these downloaded local files.

Additional checked snapshots retain the new measurements:

- [Initial relational L40S evidence](evidence/initial-relational-l40s-20261008/index.json): 29 archived records, 62 external artifacts verified; includes corrected class-aware quality audits and the separately labeled E1 smoke.
- [Initial context L4 evidence](evidence/initial-context-l4-20261008/index.json): 14 archived records, 36 external artifacts verified; includes native/LSE/CP full-model runs and both kernel probes.
- [Initial placement L4 evidence](evidence/initial-placement-l4-20261008/index.json): four archived records, 20 external artifacts verified.
- [Batched ensemble L40S evidence](evidence/batched-ensemble-l40s-20261008/index.json): eight archived result/audit records, 24 external artifacts verified; includes the passing EP2 and failing EP4 comparisons.
- [Context F1 L4 evidence](evidence/context-f1-l4-20261008/index.json): 16 archived result/rank/audit records, 32 external artifacts verified; preserves full-quantile failures and paired quality analysis.
- [Process DP L40S evidence](evidence/process-data-l40s-20261008/index.json): six archived result/audit records, 18 external artifacts verified.
- [Tuned and relational process DP evidence](evidence/tuned-process-data-l40s-20261008/index.json): ten archived result/audit records, 27 external artifacts verified; includes matched estimator-batch-four quality references.
- [Weak process DP evidence](evidence/weak-process-data-l40s-20261008/index.json): six archived result/audit records, 18 external artifacts verified; current equivalence audit covers common prefixes.
- [H&M 16k placement evidence](evidence/placement-hm16k-l4-20261008/index.json): 12 archived result/audit/command records, 24 external artifacts verified.
- [Resident 16k context L4 evidence](evidence/context-large16k-l4-20261008/index.json): 12 archived records, 28 external artifacts verified.
- [Initial Nsight evidence](evidence/initial-nsys-l40s-20261008/index.json): two archived analysis records, four external analysis/SQLite artifacts verified.

Collect each later completed group into a fresh directory; never overwrite an earlier collection. The collector preserves failed-run records too, and does not reconstruct a command that was never recorded. Source revisions, runner hashes, and exact parameters are preserved from raw result JSON. If the runner emits `command.txt`, that file is archived verbatim.

```sh
python research/multigpu/collect_evidence.py collect \
  --run cp2=/absolute/local/cp2-result \
  --output research/multigpu/evidence/context-comparison-20261008
python research/multigpu/collect_evidence.py verify \
  --index research/multigpu/evidence/initial-tabular-20261008/index.json
python research/multigpu/collect_evidence.py verify \
  --index research/multigpu/evidence/initial-tabular-20261008/index.json \
  --external
```

Default verification checks repository-retained records and reports how many external files were not checked. `--external` also requires the large local files and verifies their content. Checksums detect changes; they do not themselves certify benchmark methodology.

## Reproduction and failure log

The CP prototype (`8e22f29c8`) has three passing CPU tests exercising real 2/4-rank Gloo groups, both model ICL blocks, MHA/GQA, broadcast batches, empty/uneven shards, cache backing-storage release, and invalid topology. This establishes distributed CPU correctness within those tests. It does not establish CUDA kernel compatibility, model quality, or GPU speedup. See the implementation's `research/multigpu/context-parallel.md` for its exact scope.

The EP implementation's CPU tests include reduced real KumoTabular modules with 12-class ECOC, class shuffling, exact member codebooks/predictions across 1/2/4 replicas, and refit. The first CUDA run failed both tests because output stream bookkeeping called an unavailable `TableTensor._tensors()` method. `c4ddd7f15` switched to the public `TableTensor.record_stream` method; the runner then reported both CUDA tests passed in 0.80 seconds on integrated `6c3f1e22e`. Independent review also fixed replica-initialization stream dependencies and output tensor stream lifetime. Its public documentation is `research/multigpu/ensemble.md` and the independent comparison checklist is `research/multigpu/quality.md` (`bf3a602e8`).

The tabular runner (`af6d89791` + `10159e909`) provides the following initial comparison. Repeat for native estimator batch sizes that fit memory, then `--mode ensemble --gpus 1`, `2`, and `4`, each with a fresh output directory. CPU-complete outputs and profiler-only passes are separate from throughput measurements.

```sh
python research/multigpu/tabular_bench.py \
  --data /path/to/covertype --output /path/to/tabular-native \
  --task classification --size large --mode native --gpus 1 \
  --estimators 4 --context 1024 --queries 2048 --batch-size 256 \
  --repeats 3 --precision bfloat16 --profile \
  --source-commit "$(git rev-parse HEAD)"
```

The native relational harness exposes the following commands (paths must point to the installed runner revision and the actual RelBench cache):

```sh
python research/multigpu/relational_bench.py prepare \
  --raw /path/to/rel-hm --output /path/to/fixed-workload \
  --context 1024 --queries 2000 --batch-size 250
python research/multigpu/relational_bench.py run \
  --workload /path/to/fixed-workload --mode native --gpus 1 \
  --estimators 4 --repeats 3 --output /path/to/native-result --profile \
  --source-commit "$(git rev-parse HEAD)"
python research/multigpu/relational_bench.py run \
  --workload /path/to/fixed-workload --mode ensemble --gpus 2 \
  --estimators 4 --repeats 3 --output /path/to/ep2-result --profile \
  --source-commit "$(git rev-parse HEAD)"
```

Expected harness artifacts are `result.json`, predictions, utilization samples, and a profiler trace. Native complete graph batches are prepared once, with sampling time recorded separately. The independent CP probe uses random weights and is a kernel/scaling diagnostic, not a quality benchmark:

```sh
torchrun --standalone --nproc-per-node=4 research/multigpu/context_probe.py \
  --family tabular --context 4096 --queries 128 --channels 512 \
  --heads 8 --kv-heads 2 --layers 4 --dtype bfloat16 \
  --output /path/to/context-probe-result
```

For pretrained full-model CP, run a fresh process group for each arm; use one process for `--mode native` and `--mode lse`, then 2/4 processes for `--mode context`. The relational variant consumes all graphs from its fixed workload and records actual sizes from that workload, irrespective of tabular-only size arguments.

```sh
torchrun --standalone --nproc-per-node=2 \
  research/multigpu/context_model_bench.py \
  --family tabular --data /path/to/covertype \
  --output /path/to/cp2-result --mode context --size small \
  --context 4096 --queries 2048 --batch-size 256 --estimators 4 \
  --repeats 3 --precision bfloat16 \
  --source-commit "$(git rev-parse HEAD)"
```

These commands assume execution from a Git checkout. For an exported source archive, pass its recorded commit explicitly instead of invoking `git rev-parse`. An early integrated local validation reported 43 CPU tests passed and 22 CUDA tests skipped; subsequent real CUDA test results are reported in the approach matrix and corresponding implementation documents. Local skips alone never establish GPU correctness.

Historical inspection is reproducible with:

```sh
git show 81b8f7fc4:sdm/models/ensemble.py
git show 499485af5:sdm/nn/context_parallel.py
git show 6aa9f4f14:examples/kumo/relational/multi_gpu.py
git show 6b4f0e074 -- benchmark/tabular/model.py
git diff origin/prototype/kumotabular-row-partition^ origin/prototype/kumotabular-row-partition
```

| Attempt / finding | Outcome | Evidence | Follow-up |
|---|---|---|---|
| Treat context-subset ensemble as exact CP | Rejected by code inspection | Row-partition branch gives each estimator a different subset and fitted transform | Measure its quality/performance separately if pursued |
| Claim EP speedup against public offloaded baseline alone | Insufficient comparison | `sdm/models/base.py` CUDA multi-estimator fit offload and predict prefetch | Add EP1 with matching residency and estimator batching |
| Assume cached CP distributes fit computation or guarantees fit capacity | Rejected by code inspection | Each rank computes the full context, then shards each layer's saved KV | Accumulated cache memory can shrink, but full-row activation/compute remains; measure fit peak separately |
| Slice relational neighbors by query row indices | Invalid graph transformation | Related-table row spaces differ and shared neighbors cross task rows | Freeze complete sampled graph per batch |
| Suspected singular relationship metadata keys | Valid supported alias; no defect | Quality reviewer inspected `sdm/relational/data.py` and `task.py` | No change needed |
| Initial EP CUDA stream tests | Failed: unavailable `TableTensor._tensors()` | Initial host test log: two failed, eight passed | Fixed by `c4ddd7f15`; both CUDA tests passed on `6c3f1e22e` |
| CP full-model runner evidence audit | Six reporting/boundary gaps found | Cache residency, requested/actual graph sizes, silent truncation, eager input memory, failed-run artifacts, repeat parity | `250cae105` adds explicit evidence; inputs remain GPU-resident consistently across native/LSE/CP |
| Eight-GPU/NVSwitch capacity | Unavailable under observed regional Spot quota | Coordinator reports 64-vCPU regional limit | Use separate four-L40S and four-L4 ladders; no NVLink conclusion |
