# Kumo multi-GPU investigation

Status: measured study and operational cleanup complete, 2026-10-08. Implementation baseline: `842c408fe`. This report separates measured results from architectural expectations. Unmeasured proposals and explicit numerical failures remain documented limitations, not implied successes or queued experiments.

The goal is practical multi-GPU inference for both `KumoTabular` and `KumoRelational`, including throughput, latency, capacity, prediction quality, and a lightweight integration into SDM. The implementation recommendations are in [integration.md](integration.md).

## Executive comparison

**Use persistent query DP for many independent batches; add graph replay only for supported repeated tabular shapes. Use ensemble splitting for a single wider ensemble, and stage placement when retaining context exceeds one GPU's capacity. Cached CP is a memory option, not a speed recommendation on the tested PCIe hosts.** No approximate context partition/retrieval or retrained predictor is presented as an exact parallel speedup.

Setup: real Covertype classification, California Housing regression, `rel-hm/user-churn` classification and `rel-f1/driver-position` regression. Tabular uses fixed seed-20261008 TRAIN/validation splits; RelBench uses native TRAIN/VAL, never TEST labels. Native relational graphs preserve task timestamps and two-hop `[16,16]` neighborhoods. Measured contexts range from 1,024 to 65,536 rows; ensemble sizes are primarily E4/E8. Checkpoints are pinned to Kumo-Tabular `3c3e10bbdb590ace29e7026847f92db3c603096d` and Kumo-Relational `2bd603d3d8f25f67a7aaa8579908595e20567e22`. Each host verified all 15 manifest files. Source revisions and graph hashes are retained per run.

Hardware: Spot four-L40S hosts for throughput and separate four-L4 hosts for CP/capacity, PyTorch 2.9.1+cu130, PCIe `NODE` links and no NVLink. Replacement hosts are Ohio L40S and Frankfurt L4. Compare speed only within each matched host/runtime/workload ladder. Most rates are median three warm passes after one fit, not independent training/fit trials. The table's memory values are allocator peaks or specifically named cache storage, not whole-device or simultaneous aggregate peaks.

| Approach / workload | Matched 1→multi-GPU result | Memory / quality outcome | Practical conclusion |
|---|---|---|---|
| Graph query DP, Covertype E4/C1k/Q65k/B1k, L40S | 11,207→41,649 rows/s, **3.716× at 4 GPUs**; native-DP4 29,545, graph gain **1.410×** | Exact across graph GPU counts/all repeats; graph vs native batch4 BF16 gate passes but 50 labels change/65,536. Per-worker prediction allocation 1.291 vs native2.182 GiB; reservation 3.566 vs3.094 GiB | Best measured throughput for this supported numerical-tabular workload; includes IPC/final CPU gather; no quality gain claimed |
| Cross-host native query DP, Covertype E4/C1k/Q65k/B1k, mixed L4/L40S | Balanced2→8 GPUs: 6,239→24,770 rows/s, **3.970×**; weighted8 reaches31,877 (**28.7% over balanced8**) | All repeats exact to each row's physical-GPU-family oracle; weighted placement changes12 labels, BF16 gate passes | Heterogeneous Frankfurt/Ohio scaling, not homogeneous8; preloaded inputs, network/decode/gather included; separate from local graph-DP timer/backend |
| Native query DP, H&M E4/C1k/Q2k/B250, L40S | 1,186→3,859 rows/s, **3.254× at 4 GPUs** | Exact native/member policy; full fitted context replicated; original timer excludes final concatenation | Simple generic path for complete relational query graphs; not comparable unqualified to newer fully gathered timers |
| Process EP, fresh H&M E4/C1k/Q2k/B250, L40S | 1,232→2,396 rows/s, **1.944× at 4 GPUs**; resident1 1,412 | Every repeat exact; per-child prediction allocation 507→380 MiB, plus process overhead; member cache distributed | Helps a single ensemble, but startup/load rises to11.31s and fresh graph/RNG policy differs from historical native-DP ladder |
| Graph EP, Covertype E4/C1k/Q2k/B256, L40S | 6,398→11,146 rows/s, **1.742× at 4 GPUs**; graph1 already1.597× tuned native | Every repeat exact native-batch1; four retained captures, no timed recapture | Separate graph optimization from GPU scaling; neither process/graph adapter is in the minimal published EP branch |
| Cached CP, Covertype E4/C16k, L4 | Native1 1,853 vs all-gather CP4 1,526 rows/s; **no speedup** | ICL cache768→192 MiB/rank; prediction peak2.189→1.628 GiB; BF16 gate passes, two labels flip/2,048 | Retained-memory option; fit still computes full context per rank; PCIe collectives dominate benefits |
| CP precision isolation, F1 E4/C1k, L4 | Full-FP32 native304 vs CP2 300 rows/s; BF16 native377 | Full-FP32 passes every one of499×999 quantiles in every repeat; original BF16/partial-FP32 failures remain | Numerical closure costs precision/runtime; not a repaired or accelerated default BF16 path |
| Stage2, H&M E8/C64k, L4 | Resident stage2 succeeds at513 rows/s; resident1 and EP4 OOM | Fit per-device peaks16.130/12.891 GiB; no successful same-policy resident1 oracle. Native CPU offload fits one L4 | Capacity option for this resident-cache policy, **not proof the workload intrinsically requires multiple GPUs** |
| Blocked GNN, H&M E8/C64k native offload, L4 | Native436 vs blocked433 rows/s | Fit allocation16.737→11.865 GiB, reservation increases; final probability gate passes, but23 actual-checkpoint module GPU cases fail | Exploratory memory–numerics tradeoff; do not promote as an exact/default replacement |

For SDM, keep process ownership and fixed-batch examples outside the model core; extract small fitted-cache/ensemble contracts independently. Graph shape caches need explicit bounded growth, output lifetime and refit invalidation. Keep CP/placement opt-in and precision-aware; do not add cloud orchestration or automatically dispatch by GPU count. All required scientific evidence is downloaded and independently reviewed. Final checks at03:03:04 UTC verify both replacement instances terminated, their disks absent and no remaining task resources in the four used regions. Detailed baselines, negative results, regression quantiles, raw evidence and limitations follow.

Published branches: [full study](https://github.com/NVIDIA/structured-data-models/tree/research/kumo-multigpu-20261008) and [minimal resident ensemble executor](https://github.com/NVIDIA/structured-data-models/tree/feature/ensemble-parallel). The minimal branch excludes research process/graph adapters and their performance gains. No PR is open. Final local tests: **230 passed,101 CUDA-only skips**; actual GPU checks and the failed GNN cases have separate retained receipts. Cloud resources were removed after scientific execution and independent audits completed, not as a budget-based stopping rule. Operational and cost-accounting details are in the [handoff](handoff.md#final-verified-operational-closure).

The [foundation-model literature synthesis](literature.md) ranks five next experiments, distinguishing current-checkpoint execution changes from approximate context selection and new-model approaches. These paper-inspired proposals are not additional measurements.

Persistent process query DP is the most consistently useful throughput decomposition in the tested workloads. On the larger-batch Covertype ladder, graph-backed query DP reaches **41,649 rows/s on four L40S GPUs**, including final CPU gathering: **3.716× its matched graph-DP1 control**, or **1.410× native-DP4** on the same 65,536 queries/batch1,024. All repeats are byte-identical across GPU counts within each backend; graph versus native estimator-batch4 is not exact (50 label flips, unchanged BF16 gate passes). A separate batch256 ladder reaches 22,667 rows/s and 3.731× graph scaling. These are supported static-shape numerical-tabular results, not a universal throughput ceiling. Native process DP is the simpler generic option and scales well on relational H&M. Larger Covertype EP2 also scales 1.446× after fixing local estimator batching; California Housing EP and the threaded hybrid do not speed up. Placement and cached CP mainly offer memory tradeoffs, not throughput gains on these PCIe hosts. BF16 relational CP fails its full-quantile gate; a separately measured full-FP32 control passes without a speedup.

New same-host controls on the replacement L40S machine close the pretrained process/graph ensemble gap: graph EP reaches 6,398/9,134/11,146 rows/s on 1/2/4 GPUs, with byte-identical outputs on every saved repeat. Its four-GPU gain is **1.742× over graph EP1**, or **2.782× over tuned native**, which includes the single-GPU graph optimization. Process EP4 scales 2.724× over process EP1 but only 1.109× over tuned native. These are fresh-host comparisons, not new denominators for the historical process-query-DP ladder.

## Approach matrix

| Approach | Work placement | Expected benefit | Main limitation / correctness requirement | Current owner and evidence |
|---|---|---|---|---|
| Query data parallelism (DP) | Complete, fixed query batches across persistent replicas | High aggregate throughput; no attention collectives | Full model and fitted context replicated; preserve batch boundaries and complete relational neighborhoods | Fresh graph-backed tabular DP scales 3.731× including final gather and beats same-shape native-DP4 by 1.581×; native H&M process DP scales 3.254× excluding final concat; initial threaded results weak |
| Ensemble parallelism (EP) | Fixed logical estimator IDs across replicas | Reduce member work and member cache per GPU; small output gather | Same recipes, member RNG, class alignment, output reduction, and cache residency as reference | Initial unbatched threads weak; E8 fixed-batch EP2 gives 1.446×; fresh E4 process EP4 gives 2.724× and graph EP4 1.742× their own EP1 controls; graph EP1 itself beats tuned native |
| Cached context parallelism (CP) | ICL KV rows across ranks; queries replicated | Larger retained context; possible attention acceleration | Full context compute still replicated at fit; stable softmax reduction, global length scaling, uneven shards; collectives every layer | Real NCCL tests passed; 1k and resident 16k tabular runs slower; BF16 F1 all-quantile gate fails |
| Layer/stage placement | Row encoder, GNN, and/or ICL layers on different devices | Parameter/cache capacity; potential pipeline overlap across query batches | Single-query latency can worsen; cache ownership and transferred activations must follow stages | Source `55dba6bb5`: 36 CPU/CUDA tests passed on L4; real model memory decreases with small throughput cost; sequential placement, not overlapped pipelining |
| Table embedding parallelism | Independent related-table encoders across devices, then gather row embeddings | Parallel relational table encoding before GNN | Tables may be imbalanced; shared preprocessing/target propagation and per-table RNG must remain fixed | Design inspected; no measurement |
| Tensor parallelism (TP) | Projection/MLP widths or attention heads across ranks | Parameter and compute sharding for a single member | Many collectives, packed projections/custom kernels, small widths and GQA limit useful partitions | Design inspected; implementation not yet established |
| Full context fit sharding | Row encoder inducing attention and context self-attention distributed during fit | Reduce peak fit memory, not only retained cache | Requires global induced-state/attention reductions in both row encoder and ICL | Design inspected; separate from cached CP |
| Relational graph partition | Nodes/edges and GNN state split with boundary exchange each hop | Larger graph capacity | Exact halo/state exchange, relation types, target ownership, temporal sampling; irregular load balance | Design inspected; no exact implementation or result |
| Context-subset ensemble | Different context rows assigned to independent members | Less fit/cache work per member | Changes model input and potentially prediction quality; not exact CP | Existing `prototype/kumotabular-row-partition` inspected; separate quality arm |
| EP × DP / EP × CP | Process groups split along two axes | Balance query volume, ensemble width, and context capacity | Product of group sizes must equal GPUs; group-local output order and RNG | Threaded 2DP×2EP measured at 0.930× matched one-GPU EP, exact predictions; EP×CP remains design-only |

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

The EP cache's 489 MiB unique backing storage exceeds the public CPU cache's 342 MiB, despite identical predictions. Source inspection identified row-encoder value views retaining fused KV backing buffers; a later GPU-measured compact-cache clone reduces storage to 342 MiB with exact predictions, documented below. Nsight shows short-kernel scheduling and little compute overlap, but does not identify the complete causal chain. The historical TabFM 1.876× result does not predict these Kumo results.

The tuned one-GPU comparison uses the same large model, Covertype rows, E4, context 1,024, query 2,048, query batch 256, and BF16 autocast. Only estimator batching changes:

| Native estimator batch | Median rows/s | Batch p50 | Fit seconds, warmed environment | Accuracy | Log loss | Max probability difference vs estimator batch 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,758.63 | 145.36 ms | 1.327 | 0.764160 | 0.576045 | 0 |
| 2 | 2,984.01 | 85.76 ms | 1.253 | 0.764160 | 0.575996 | 0.008804 |
| 4 | 4,007.14 | 63.82 ms | 1.255 | 0.764160 | 0.575978 | 0.005607 |

Predicted class labels are unchanged for all 2,048 rows. Estimator-batched BF16 outputs are not byte-identical; mean absolute probability differences are about 0.000263 and 0.000259. These quality changes must remain visible when comparing speed against the sequential member path. Best measured native batching is 2.28× the repeated unbatched native run and 2.13× the initial two-GPU EP. Multi-GPU follow-ups must include local estimator batching before claiming a useful advantage.

### Resumed pretrained process and graph ensemble comparison

The replacement host runs frozen source `c0fb64fcc`, Torch 2.9.1+cu130 and the same verified KumoTabular-large checkpoint. Covertype remains E4, 1,024 TRAIN context rows, 2,048 VAL queries, batch 256, BF16 and seed 1729. All nine arms have fresh controls on that host, three timed passes, saved per-repeat arrays and independent quality audits. The public runner starts from GPU-resident queries and returns CPU-complete outputs; process-adapter serialization and gathering are included inside its prediction call. Detailed setup/capture costs are in [tabular-results.md](tabular-results.md).

| Arm | GPUs | Median rows/s | Matched executor scaling | Max prediction allocation per GPU, GiB |
|---|---:|---:|---:|---:|
| Native, member batch1 | 1 | 1,802.80 | — | 1.284 |
| Native, tuned member batch4 | 1 | 4,006.46 | — | 1.646 |
| Resident thread EP | 1 | 1,849.81 | — | 1.598 |
| Process EP | 1 | 1,631.39 | 1.000× | 1.597 child |
| Process EP | 2 | 2,900.90 | 1.778× | 1.357 child |
| Process EP | 4 | 4,444.76 | 2.724× | 1.173 child |
| Graph EP | 1 | 6,398.19 | 1.000× | 1.286 |
| Graph EP | 2 | 9,134.14 | 1.428× | 1.047 |
| Graph EP | 4 | 11,145.61 | 1.742× | 0.927 |

Every resident/process/graph prediction repeat is byte-identical to native batch1, with accuracy 0.764160, log loss 0.576045 and macro AUROC 0.947415. Tuned native batch4 changes arithmetic, passes the existing BF16 screen and has the same accuracy, but is not byte-identical; retain it as the stronger practical tuning control. Graph EP1 is already 1.597× tuned native; graph EP4's 2.782× native ratio combines that optimization with multi-GPU scaling. Process EP4's corresponding native ratio is only 1.109×. Three repetitions share one fitted state; these are not independent-fit trials. Resident1 has a slower third pass and 4.81% timing CV.

Process parent allocation adds a separately measured 2.80 MiB peak on GPU0; child peaks in the table are not whole-device usage and must not be summed as simultaneous peaks. Prediction-window physical telemetry maxima are 2,946/2,726/2,396 MiB for process1/2/4 and 2,383/1,989/1,645 MiB for graph1/2/4. Sampling can miss brief peaks, especially the three/four telemetry samples during graph runs. Child RSS is lifetime high-water and may include shared pages, not additive unique host RAM.

Graph counts stay four before and after timing: no measured pass captures another graph. Synchronized warmup wall time including capture is 0.525/0.466/0.540 seconds; summing overlapping worker capture durations is not wall latency. Fit-plus-warmup memory high-water was not reset between those phases, so it is not an isolated capture peak. Prediction peaks were reset. Process load/spawn takes 4.72/8.16/14.99 seconds separately from fit. Initial native fit takes 103.21 seconds and includes cold-start effects; no parallel fit-speedup claim uses it. These graph results cover one fixed numerical full-batch shape, not arbitrary shapes or relational graph capture. The paired Nsight follow-up below diagnoses launch behavior separately from these clean timings.

The [additive resumed evidence snapshot](evidence/resumed-tabular-executors-l40s-20261008/index.json) retains 19 small records, all nine independent per-run audits and the group timing/memory audit, with 82 externally verified artifact references. No historical result was overwritten.

### Batched ensemble follow-up on L40S

The larger workload uses KumoTabular large, Covertype, **E8, context 4,096, query 8,192, batch 1,024**, BF16 autocast, three measured passes, and source `1686803e4`. The native reference already uses estimator batch size eight. The resident batched executor distributes the same members and batches compatible local members.

| Arm | GPUs | Median rows/s | Speedup vs resident EP1 | Fit seconds | Max prediction peak per GPU | Prediction status |
|---|---:|---:|---:|---:|---:|---|
| Native, estimator batch 8 | 1 | 3,219.44 | — | 2.749 | 3.743 GiB | Reference |
| Resident batched EP | 1 | 4,013.81 | 1.000× | 2.077 | 3.503 GiB | Byte-identical to native |
| Resident batched EP | 2 | 7,368.45 | **1.836×** | 1.648 | 2.405 GiB | Byte-identical to native and EP1 |
| Resident batched EP | 4 | 5,981.13 | 1.490× | 1.477 | 1.612 GiB | Fails declared numerical gate; see below |

The two-GPU result is 91.8% scaling efficiency relative to resident EP1. Its 2.289× speedup over public native combines 1.247× from residency/scheduling with the separate 1.836× multi-GPU gain. Native/EP1/EP2 accuracy is 0.841797, log loss 0.413884, and multiclass AUC 0.974934. These are validation-prefix results, not a full dataset ranking.

The four-GPU result is slower than two GPUs and differs in one of 57,344 probability entries beyond the predeclared BF16 tolerance; maximum difference is 0.032253, with six changed labels. Accuracy is 0.841919 and log loss 0.413924, with a paired log-loss change interval spanning zero. Similar aggregate quality does not turn a failed numerical gate into a pass. Local estimator groups shrink from eight to four to two across placements; the following controls isolate that confound without removing the original failure.

The same E8 workload was rerun with native estimator batches two/four and fixed local batch-two resident controls. The stronger native baseline changes the performance interpretation:

| Arm | GPUs | Actual local member batch | Median rows/s | Speedup vs resident EP1 batch2 | Numerical comparison |
|---|---:|---:|---:|---:|---|
| Native batch2 | 1 | 2 | **4,433.85** | — | Batch2 reference |
| Native batch4 | 1 | 4 | 3,777.01 | — | Exact native batch8, differs from batch2 |
| Resident EP1 batch2 | 1 | 2 | 4,583.76 | 1.000× | Exact native batch2 |
| Resident EP2 batch2 | 2 | 2 | **6,626.77** | **1.446×** | Exact native batch2 and EP1 batch2 |
| EP2 reverse-order repeat | 2 | 4 | 7,282.49 | Not batch-matched | Exact native batch4/batch8 |
| EP4 reverse-order repeat | 4 | 2 | 5,875.07 | 1.282× | Exact native batch2 and original EP4 |

Fixed batch-two EP2 is 1.495× tuned public native and 1.446× its resident control. The earlier 1.836× result remains valid for its recorded execution policies, but its local member width changes eight→four and its native batch-eight baseline is not best tuned. Four-GPU outputs matching native batch2 exactly show that the original cross-batch numerical failure is a batching-shape effect rather than evidence of incorrect distributed member aggregation. The original native-batch8 comparison still fails; accepting a batch-two deployment means selecting and validating that numerical policy explicitly. Four GPUs remain slower than two here.

### Regression and hybrid counterexamples

California Housing uses **KumoTabular large, E8, context 4,096, queries 4,096, batch 512**, BF16, and fixed local estimator batch two across every arm. It has eight input features and 999 quantile outputs, versus Covertype's 54 features and seven probability outputs.

| Arm | GPUs | Median rows/s | Speedup vs resident EP1 | Max prediction allocation per GPU |
|---|---:|---:|---:|---:|
| Native batch2 | 1 | 2,815.71 | — | 1.368 GiB |
| Resident EP1 batch2 | 1 | 2,662.00 | 1.000× | 1.653 GiB |
| Resident EP2 batch2 | 2 | 2,575.15 | 0.967× | 1.392 GiB |
| Resident EP4 batch2 | 4 | 2,316.35 | 0.870× | 1.213 GiB |

All 4,096 × 999 quantiles are byte-identical, with independently verified original validation IDs/targets, no crossings, RMSE 0.416896, and MAE 0.244908. Memory per device falls, but throughput worsens; success on the larger Covertype configuration is not a family-wide EP recommendation. Feature count, batch size, and output shape all change between these workloads, so no single factor is identified causally.

A threaded **2 query-DP groups × 2 ensemble GPUs** hybrid was also measured on small Covertype E4/context-1,024/query-2,048/batch-256, using prepared **CPU query inputs in both arms**. Matched resident EP1 reaches 1,763.70 rows/s versus hybrid4 1,640.59 (0.930×), with exact predictions. Maximum per-GPU prediction allocation falls from 1.598 to 1.357 GiB. This particular threaded composition provides no throughput benefit; it does not rule out a process-based hybrid at other workloads.

### Process query DP on L40S

One persistent process per GPU changes the result for the original small Covertype workload (large model, E4, context 1,024, queries 2,048, batch 256). Each worker uses one CPU thread and the native unbatched estimator path. The timer is parent wall time including input IPC, transfers, inference, and CPU prediction return; final CPU concatenation/scoring are excluded consistently. Process DP starts with prepared CPU query batches, whereas the original tabular native/EP runner has GPU-resident query inputs. Consequently the native-versus-process ratios have a different input boundary as well as the final-concatenation caveat; each process-DP GPU-count ladder remains boundary-matched.

| Processes / GPUs | Median rows/s | Speedup vs DP1 | Scaling efficiency | Spawn + load + fit | Peak prediction allocation per GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,780.50 | 1.000× | 100% | 5.092 s | 1.284 GiB |
| 2 | 3,418.27 | 1.920× | 96.0% | 5.231 s | 1.284 GiB |
| 4 | 6,650.11 | **3.735×** | **93.4%** | 5.521 s | 1.284 GiB |

All predictions are byte-identical to unbatched native, and the independent audit verified full input/row identities. Model/cache storage and roughly 2.07 GiB of host RSS per worker are replicated. This unbatched four-GPU throughput is 1.66× the best measured one-GPU estimator-batched baseline (4,007 rows/s), not 3.735× that stronger baseline. The contrast with weak threaded scheduling supports process isolation as practical, but CPU thread-count differences and profiler coverage prevent attributing the entire effect to the Python GIL alone.

The follow-up combines the same process topology with **local estimator batch size four**; context/query batches and one CPU thread per worker remain unchanged:

| Arm | GPUs | Median rows/s | Speedup vs matched DP1 | Spawn + load + fit | Peak prediction allocation per GPU |
|---|---:|---:|---:|---:|---:|
| Tuned native, estimator batch 4 | 1 | 4,007.14 | — | Fit 1.255 s, excluding spawn/load | 1.646 GiB |
| Process DP, estimator batch 4 | 1 | 3,665.05 | 1.000× | 5.013 s | 1.404 GiB |
| Process DP, estimator batch 4 | 4 | **13,990.88** | **3.817×** | 5.469 s | 1.404 GiB |

Four-GPU scaling efficiency is 95.4%, and throughput is 3.492× tuned public native. Both process arms are byte-identical to the **estimator-batch-four** native reference (not the unbatched native reference); accuracy is 0.764160 and log loss 0.575978. The measured process boundary includes input/output IPC and GPU transfers but ends before final CPU concatenation. It must not be silently equated with later runners that time the final gather; that missing duration has not yet been measured for these attempts.

The explicit gather-timed rerun includes output order validation and final CPU concatenation before stopping the clock:

| Tuned processes / GPUs | Fully gathered median rows/s | Matched gathered speedup | Scaling efficiency |
|---:|---:|---:|---:|
| 1 | 3,659.40 | 1.000× | 100% |
| 4 | **13,822.14** | **3.777×** | **94.4%** |

Gather adds 0.324–0.666 ms across the six measured passes. Both outputs remain byte-identical to tuned native, and per-GPU prediction peak remains 1.404 GiB. These are new attempts, not retroactively adjusted original measurements; the 3.817× earlier result remains recorded with its narrower timing boundary. The fully gathered result is approximately 3.45× native's 4,007.14 rows/s, with the previously stated CPU-versus-GPU input-residency difference.

Weak scaling keeps 4,096 query rows per GPU while context 1,024, E4, estimator batch four, and query batch 256 stay fixed:

| GPUs | Total queries | Median rows/s | Rows/s per GPU | Weak efficiency vs DP1 |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 3,782.40 | 3,782.40 | 100% |
| 2 | 8,192 | 7,344.95 | 3,672.47 | 97.1% |
| 4 | 16,384 | 14,682.72 | 3,670.68 | 97.0% |

Prediction peak stays 1.404 GiB per GPU. Independent checks initially found exact outputs on the shared 4,096-row prefix and exact agreement with tuned native on its 2,048-row prefix. A subsequent one-GPU 16,384-row reference now verifies **byte-identical outputs for the entire four-GPU cohort**, retained as an additive full-cohort audit without overwriting the original prefix-only evidence. These weak-scaling arms still score different full validation cohorts; their whole-cohort accuracy differences are not evidence of GPU-induced quality change. This is weak scaling, not a fixed-workload 3.882× speedup.

### Graph-backed process query DP: fixed-batch scaling

The replacement Ohio L40S host runs source `ae4e5e549`, PyTorch 2.9.1+cu130, large Covertype E4/context1,024/query8,192/batch256/BF16. Each persistent worker owns a full fitted model/context and one GPU, with one CPU thread. Native local estimator batch is four; graph execution uses one. Parent timing includes CPU input IPC, transfers, inference, output IPC and the final ordered CPU gather; startup, fit, capture/warmup and scoring are excluded. Three repetitions reuse one fit.

| GPUs | Native DP batch4 rows/s | Graph DP rows/s | Graph scaling vs graph1 | Graph/native at same GPU count |
|---:|---:|---:|---:|---:|
| 1 | 3,744.04 | 6,074.73 | 1.000× | 1.623× |
| 2 | 7,505.81 | 12,072.97 | 1.987× | 1.609× |
| 4 | 14,341.62 | 22,667.07 | 3.731× | 1.581× |

Every saved repeat is exact within each backend across GPU counts. Graph versus native passes the unchanged BF16 screen but is not exact: maximum probability difference 0.006434, mean 0.000254 and six class flips among 8,192 rows (one fewer correct graph prediction). Paired log-loss change −0.0000138 has 95% interval [−0.0000792, +0.0000494], not evidence of a quality improvement. The first 2,048 rows also match their earlier graph-EP1/native-batch4 references respectively.

Graph workers retain four captures each, with no capture count/event change during prediction. Capture takes 0.439–0.478 seconds per worker; concurrent worker intervals are not additive wall time. Warm-host spawn/load/fit takes 4.78–5.28 seconds for graph and 4.98–5.47 for native. Per-worker **steady-state prediction** peak allocated/reserved memory, reset after warmup, is 1,316.65/1,802 MiB for graph versus 1,438.18/3,168 MiB for native. Startup/fit/warmup high-water allocation is higher: 1,811.49 MiB graph versus 2,268.21 MiB native. Model/cache memory is replicated, not ensemble-sharded. Worker lifetime RSS is 1,714–1,719 versus 2,114–2,119 MiB; these process high-water values are not additive unique host RAM or isolated prediction memory.

The [six-arm evidence snapshot](evidence/resumed-graph-query-dp-l40s-20261008/index.json) retains raw results, independent all-repeat audits and six passing GPU contract tests for fixed/irregular shapes on 1/2/4 GPUs at batch256/1024. Those tests extend shape/lifecycle coverage but do not establish pretrained throughput on every shape. Graph capture remains a research opt-in for supported numerical tabular shapes, not a relational backend or part of the published minimal ensemble executor.

### Larger-batch query-DP control

The same runtime/source/context/E4 policy subsequently measures batch1,024 and 65,536 queries with the same fully gathered input/output-IPC timer. This isolates GPU scaling within each backend while checking whether graph replay still helps after increasing native query batch size:

| GPUs | Native DP batch4 rows/s | Graph DP rows/s | Graph scaling vs graph1 | Graph/native at same GPU count |
|---:|---:|---:|---:|---:|
| 1 | 7,520.65 | 11,207.14 | 1.000× | 1.490× |
| 4 | 29,545.50 | 41,649.49 | 3.716× | 1.410× |

Independent checks cover all three full-cohort repetitions and original validation IDs/labels. Outputs are byte-identical across GPU counts within native and within graph. Graph versus native passes BF16 screening (max difference 0.010723, mean 0.000251), but changes 50 labels and has four fewer correct predictions among 65,536. Paired log-loss change −0.00000327 has 95% interval [−0.0000248, +0.0000181]; no quality improvement is demonstrated. Stable graph counts/events show no timed recapture.

Steady-state prediction peak allocation is 1.291 GiB per graph worker versus 2.182 GiB native; **reservation goes the other way**, 3.566 versus 3.094 GiB. Startup/fit/warmup high-water allocation is approximately 1.769 versus 2.215 GiB. Warm-host spawn/load/fit takes graph1/4 4.82/5.25 seconds versus native 5.00/5.44. Whole-cohort warmup is 6.40/2.06 seconds graph and 8.75/2.27 native; these are not isolated capture costs. The [four-arm snapshot](evidence/resumed-graph-query-dp-b1024-l40s-20261008/index.json) retains raw results and all-repeat audits.

A separate [Ohio-local pipe protocol control](evidence/resumed-dp-local-pipe-l40s-20261008/index.json) reaches native 7,684.21/30,554.81 rows/s on 1/4 GPUs with the same full predictions. Its CPU inputs are preloaded, unlike the timed-input-IPC ladder above; it is not a network or eight-GPU result. The [transport audit](evidence/resumed-dp-b1024-transport-audit-20261008/index.json) establishes identical cohort outputs, not interchangeable timing boundaries. Do not use the fastest result from one boundary as a denominator for another.

### Completed mixed-host eight-GPU scaling

Thirteen new runs at source `b721583a9` use one Frankfurt coordinator, four local L4 GPUs and four remote Ohio L40S GPUs. CPU query batches are preloaded per worker before timing; one coordinator monotonic clock measures dispatch, network/output decoding and final ordered gather. Worker clocks are never summed or compared. Model/context/E4/native-estimator-batch4/BF16/batch1,024 remain fixed. Effective device count comes from worker identities/READY and unique physical GPU UUIDs, **not** the retained `args.workers=4` default overridden by explicit worker indices.

| Plan | GPUs (L4 + L40S) | Queries | Median rows/s | Valid comparison |
|---|---:|---:|---:|---|
| Balanced strong | 1 + 1 | 65,536 | 6,238.82 | Mixed2 reference |
| Balanced strong | 2 + 2 | 65,536 | 12,433.52 | 1.993× mixed2 |
| Balanced strong | 4 + 4 | 65,536 | 24,770.03 | **3.970× mixed2**, 99.26% fourfold efficiency |
| Capacity-weighted | 4 + 4 | 65,536 | 31,877.39 | **1.287× balanced8**; different row-family assignment |
| Balanced weak | 1 + 1 | 16,384 | 6,215.24 | 8,192 queries/GPU |
| Balanced weak | 2 + 2 | 32,768 | 12,415.49 | Same work/GPU |
| Balanced weak | 4 + 4 | 65,536 | 24,770.03 | **Same run as strong8**, 99.63% weak efficiency versus mixed2 |

Balanced strong2/4/8 keeps each query row on the same hardware family, so it tests placement count without silently changing hardware numerics. Every saved repetition is byte-exact to its row's independently measured L4/L40S oracle; all TRAIN/VAL identities, 64 batch hashes and eight unique GPU UUIDs pass. L4 versus L40S outputs are not bitwise equal (max0.010063, mean0.000208; BF16 gate passes), so an all-L40S oracle alone would be incorrect for this mixed fleet.

Weighted assignment uses measured local four-GPU capacity divided by four: L4 group12,411.89 and L40S group30,554.81 rows/s. Each L4 worker receives five query batches and each L40S worker eleven. It moves28,672 rows to a different hardware family versus balanced assignment, changing12 predicted labels (max probability change0.007027; BF16 gate passes); paired log-loss interval [−0.0000238,+0.0000012] spans zero. This is scheduling benefit, not improved model quality or an additional same-plan strong-scaling point.

The coordinator-matched remote Ohio4 control reaches24,095.59 rows/s versus Ohio-local4 at30,554.81. Same predictions, different coordinator/network path. Weighted8 improves32.3% over coordinator-matched Ohio4, but is not an eightfold/homogeneous speedup or a direct comparison to41,649 local graph-DP rows/s. All process/model fit state remains replicated. The [13-run snapshot](evidence/resumed-mixed-query-dp-20261008/index.json), [family-oracle audit](evidence/resumed-mixed-family-audit-20261008/index.json) and [scaling/weighted-quality audit](evidence/resumed-mixed-scaling-audit-20261008/index.json) retain all39 prediction-repeat hashes and independently checked semantics. Three warm repetitions within each fit are not independent trial confidence intervals.

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

Two follow-ups test distinct hypotheses on the original E4/context-1k/query-2k workload:

| Follow-up | GPUs | Rows/s | Change vs original matching EP | Unique fitted cache per GPU | Max prediction allocation |
|---|---:|---:|---:|---:|---:|
| Persistent worker autocast context | 1 | 1,753.36 | +1.36% | 489.00 MiB | 1.598 GiB |
| Persistent worker autocast context | 4 | 1,660.36 | +0.95% | 122.25 MiB | 1.238 GiB |
| Compact cache backing storage | 1 | 1,816.92 | +5.04% | **342.00 MiB** | **1.453 GiB** |

All three outputs are byte-identical to their original reference, with full input/seed/column checks. Persistent autocast does not produce a meaningful scaling improvement in this ladder; its EP4 peak allocation is higher than original EP4. Compaction cuts actual retained backing storage from 512,754,868 to 358,614,196 bytes (30.1%) for 3.23 ms of fit-time copying; prediction peak decreases from 1,716,191,744 to 1,560,609,280 bytes, but fit peak and allocator reservation stay unchanged. Freed live storage is reusable by the allocator, not necessarily immediately returned to the driver. Three passes from one fresh fit are not enough to call the small throughput changes robust. This establishes a cache-memory improvement separately from any speed claim.

Separate process-EP CUDA tests pass using a synthetic stochastic-cache fixture (two tests, 5.30 s), validating cache/member-plan and IPC behavior rather than actual Kumo model execution. CUDA-graph EP tests use actual reduced KumoTabular modules (two tests, 2.52 s), including changed query values, remainder batches, retained outputs, and refit. These tests establish their exercised behaviors, not pretrained model throughput or broad graph-capture compatibility; full-model measurement is separate.

### Resumed graph-replay profiling on L40S

Three new traces isolate the `prediction_pass` NVTX range for fresh resident EP1, graph EP1 and graph EP4. Independent SQL checks confirm one kernel-owning process per trace, all four graph-EP4 GPUs present, and zero kernels crossing range boundaries. The same analyzer bytes as frozen source `c0fb64fcc` reproduce all three saved analyses exactly; its SHA256 is `44b183ffd2426cc4b19c09647b43d25d243c1c189a55892d703fdee989c28dc5`.

| Diagnostic within instrumented prediction range | Resident EP1 | Graph EP1 | Graph EP4 |
|---|---:|---:|---:|
| Profiled wall seconds, not a clean benchmark | 1.5515 | 0.3978 | 0.2380 |
| Direct `cudaLaunchKernel` calls | 39,832 | 944 | 992 |
| `cudaGraphLaunch` calls | 0 | 32 | 32 |
| Total executed GPU kernels | 42,624 | 48,536 | 48,584 |
| Kernel-active fraction per GPU | 13.58% | 55.96% | 22.90–24.31% |
| Range with at least two GPUs executing kernels | — | — | 23.69% |
| Range with all four GPUs executing kernels | — | — | 8.58% |

Graph replay amortizes host dispatch; it does not remove the model's GPU computation. Single-GPU kernel-union time rises from 210.68 to 222.60 ms even while profiled wall time falls. More BF16 casts remain inside captured execution, and CPU-side class metadata also changes synchronization, so no GIL-only or single-cause attribution is made. Kernel-active fraction measures the presence of traced kernels, **not SM occupancy or hardware utilization**. API durations summed across threads are not critical-path wall time. Nsight's legacy software-instrumentation and auxiliary-process no-event/NVTX warnings remain retained; complete CPU trace coverage is not claimed. All three profiled prediction outputs match the clean reference, but their timings never replace clean repetitions.

The [trace/analysis snapshot](evidence/resumed-graph-profiles-l40s-20261008/index.json) retains small result/analysis records and hashes the original SQLite/NSight files; a subsequent [independent SQL audit snapshot](evidence/resumed-graph-profile-sql-audit-l40s-20261008/index.json) preserves process/boundary checks. Reproduce with `python research/multigpu/analyze_nsys.py /absolute/trace.sqlite --phase prediction_pass`.

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

### Resumed relational process ensemble comparison

On the replacement L40S host, frozen source `c0fb64fcc` compares resident thread EP1 with process EP1/2/4 using E4, context 1,024, 2,000 H&M validation queries, batch 250 and native temporal two-hop `[16,16]` graphs. These newly prepared query graphs differ from the historical run; all four new arms share their exact graph and member identities. The CPU-input prediction timer includes final CPU concatenation, IPC and transfers. Arrow joins remain the reference; cuDF is absent.

| Arm | GPUs | Median rows/s | Scaling vs process EP1 | Max child prediction allocation, MiB |
|---|---:|---:|---:|---:|
| Resident thread EP1 | 1 | 1,411.87 | — | 507.76 in parent |
| Process EP1 | 1 | 1,232.49 | 1.000× | 507.23 |
| Process EP2 | 2 | 1,916.98 | 1.555× | 445.25 |
| Process EP4 | 4 | 2,396.15 | 1.944× | 380.22 |

All three repeats in every arm are byte-identical to the fresh resident control; original VAL targets and churn-positive class mapping were independently checked (AUROC 0.663516). Process EP4 is 1.697× resident EP1, not a comparison against historical native or query-DP timings. Total member-cache backing storage stays 126,631,968 bytes, distributed across workers. Process parent GPU allocation is zero here; worker allocation/RSS remains separate from whole-device memory and shared host pages. Load/spawn rises from 3.53 seconds at one worker to 11.31 seconds at four; warm prediction is not cold invocation latency. The [four-arm snapshot](evidence/resumed-relational-process-ep-l40s-20261008/index.json) retains exact commands, raw results, all-repeat audits and output hashes. This validates native relational process EP, not CUDA graph capture for relational inputs.

### Relational Arrow versus cuDF control

Six matched runs test the CPU-fallback hypothesis on H&M E4/context-1,024/query-2,000/batch-250. Both modes run inside the **same cuDF 26.6 overlay environment** with unchanged Torch 2.9.1+cu130. An explicit harness switch controls cuDF discovery; receipts record actual backend checks. The native switch also changes string preprocessing availability, so that comparison is not join-only. These are backend controls, not new parallelism methods.

| Executor | Arrow rows/s | cuDF rows/s | cuDF / Arrow | Saved-array class-1 AUC: Arrow / cuDF | Max saved-array probability difference |
|---|---:|---:|---:|---|---:|
| Native1 | 1,328.04 | 1,082.90 | 0.815× | 0.661948 / 0.662338 | 0.011796 |
| Resident EP1 | 1,347.55 | 1,154.27 | 0.857× | 0.663524 / 0.663781 | 0.011978 |
| Resident EP4 | 1,364.54 | 1,059.51 | 0.776× | 0.663524 / 0.664230 | 0.011602 |

All three saved cuDF arrays pass the unchanged BF16 tolerance against their Arrow control; native changes one predicted label and EP changes none. Paired log-loss intervals include zero. cuDF is slower in this workload, so installing it does not resolve the observed scaling bottleneck. Separate inclusive TaskGraph host timings also increase, but nested/concurrent host spans must not be summed as GPU kernel time or used alone to identify the cause.

This control exposes a reporting limitation: the harness saves the **first** repeat's prediction array/hash but scores the **last** repeat. cuDF repeats are not identical according to logged maximum differences (roughly 0.0020–0.0041), and later arrays were not saved. Consequently the table uses independent metrics recomputed from the saved first array; original logged last-repeat scores remain unchanged in the raw record. Cross-repeat differences cannot be independently reconstructed from the available artifact. Native cuDF saved-array AUC is 0.662338 rather than the logged 0.662308; the same distinction applies to EP1/EP4. This is a measurement limitation, not evidence of a quality benefit.

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

The dedicated CP4 Nsight trace records all four CUDA devices and all 32 expected query-batch ranges. It observes 1,536 NCCL kernels per rank, consistent with two reductions per member/layer/batch. NCCL kernel-duration sums are 0.892/0.960/0.941/0.250 seconds on ranks 0/1/2/3; total GPU-kernel interval unions are 1.346/1.401/1.386/0.695 seconds. The large rank imbalance includes arrival/synchronization waiting and is **not** a measurement of pure transfer bandwidth. Transport logs show shared-memory routing without direct GPU peer access. Only 64 bytes of H2D traffic per rank appear during prediction, consistent with resident caches. The profiled pass takes 1.894 seconds versus roughly 1.37 seconds unprofiled: use it for diagnosis, not clean throughput. Trace warnings and analyzer output are retained.

Native KumoRelational regression on F1 also tested the Flash LSE path, with E4, context 1,024, all 499 validation rows, and fixed two-hop graphs:

| Arm | GPUs | Rows/s | MAE | RMSE | Full 999-quantile numerical gate |
|---|---:|---:|---:|---:|---|
| Native | 1 | 364.23 | 3.414711 | 4.214993 | Reference |
| Flash LSE1 | 1 | 362.35 | 3.414711 | 4.214993 | Exact native predictions |
| Context parallel | 2 | 361.15 | 3.418062 | 4.219112 | **Fail** |
| Context parallel | 4 | 376.00 | 3.416702 | 4.217447 | **Fail** |

Both CP outputs are finite and monotone, and their median predictions pass tolerance, but the complete 499 × 999 arrays fail `atol=0.01, rtol=0.05`: 938 entries fail at two ranks and 915 at four. Maximum absolute error is about 0.20224; mean absolute errors are 0.017043 and 0.014253. The largest relative failures occur in near-zero tail quantiles, not necessarily the largest absolute-error entries. Paired MAE changes are small but consistently positive in the auditor's driver-cluster bootstrap. Consequently the approximately 3.2% CP4 throughput gain is not an accepted equivalent-result win.

The attempted correction keeps BF16 model inputs but computes local attention outputs and distributed recombination in FP32 before one cast back. It reduces error but **does not fix the gate**:

| FP32-partial arm | Rows/s | MAE | Failed quantiles vs original native | Failed quantiles vs matching FP32-partial LSE1 | Max difference vs matching LSE1 |
|---|---:|---:|---:|---:|---:|
| LSE1 | 336.76 | 3.414720 | 771 | 0 | 0 |
| CP2 | 365.78 | 3.414638 | 792 | 98 | 0.057785 |
| CP4 | 369.30 | 3.414847 | 786 | 93 | 0.086674 |

The higher-precision single-rank kernel itself changes results versus original native (maximum difference 0.144457), so it cannot be silently substituted as an identical control. Even against that new matching LSE1, both distributed arms fail the unchanged tolerance. All outputs remain finite and monotone, and paired MAE intervals span zero. This remains a failed numerical-fix attempt. The full-FP32 control below subsequently isolates the precision issue without changing or erasing these BF16 results.

The resident native H&M comparison uses **actual context 16,384, queries 4,096, batch 512**, E4, and two-hop `[16,16]` graphs. These sizes come from the input identity manifest; the runner's unused CLI defaults still display 1,024/2,048/256 and must not be mistaken for executed shapes.

| Arm | GPUs | Unique rows/s | ICL cache per rank | Max prediction peak per GPU | Max probability difference vs native |
|---|---:|---:|---:|---:|---:|
| Native | 1 | 1,061.85 | 1,536 MiB | 2.370 GiB | 0 |
| Flash LSE1 | 1 | 1,057.77 | 1,536 MiB | 2.370 GiB | 0.008284 |
| CP4 | 4 | 1,048.54 | 384 MiB | 1.245 GiB | 0.016804 |

CP4 passes the unchanged BF16 probability tolerance and changes no class labels, but is 1.25% slower than native. Its mean probability error is 0.002624, and paired log-loss change is +0.0000328 with 95% interval [−0.0002254, +0.0003025]. Peak prediction allocation falls 47.5%; fit peak remains about 4.471 GiB. All rank/repeat hashes, workloads, row IDs, and slowest-rank timings were independently checked. This supports retained-cache capacity, not a throughput win. These native RNG/preprocessing controls are separate from the resident-member policy in the placement table; do not compare their different quality values as a GPU-count effect.

### Resumed CP precision and collective controls on L4

The replacement Frankfurt L4 host runs `c0fb64fcc` with newly measured same-host controls. Full-model FP32 disables autocast, retains FP32 parameters and caches, uses highest matmul precision with CUDA matmul TF32 disabled, and uses the FP32 efficient attention backend. The runner separately records the cuDNN TF32 flag; this is not a blanket assertion that every library flag is disabled. Native F1 remains E4, context 1,024, all 499 VAL queries, batch 125 and the same complete sampled graphs; default CPU cache offload and GPU-resident prepared inputs are shared across these four arms.

| Arm | GPUs | Median rows/s | Max prediction allocation, MiB | Full 499 × 999 output comparison |
|---|---:|---:|---:|---|
| BF16 native | 1 | 377.43 | 338.86 | Precision reference only |
| FP32 native | 1 | 303.73 | 416.39 | FP32 reference |
| FP32 LSE1 | 1 | 299.96 | 416.39 | Byte-identical to FP32 native |
| FP32 CP2 | 2 | 300.31 | 367.39 | Strict FP32 gate passes; not byte-identical |

Every CP repeat passes the unchanged strict `atol=1e-5, rtol=1e-4` gate with zero failed quantiles, maximum error 0.0000133514 and mean error 0.000000899185. Outputs are finite, monotone and repeat-stable; other ranks report matching output hashes. This is **numerical isolation, not a throughput win**: CP2 is essentially tied with FP32 LSE1, slower than FP32 native, and about 20% slower than BF16 native. FP32 fit peak is approximately 1,100 MiB in all three arms versus 779 MiB for BF16. Quality changes between FP32 and BF16 are a precision effect, not a benefit of distributing attention. Original BF16/FP32-partial failures remain unchanged. See the [precision-isolation evidence](evidence/resumed-context-fp32-l4-20261008/index.json).

A second same-host ladder tests resident Covertype-large E4, context 16,384, 2,048 queries and batch 256, using Flash local attention and either two all-reduces or an all-gather of partial statistics:

| Arm | GPUs | Median rows/s | Max prediction allocation, GiB |
|---|---:|---:|---:|
| Native | 1 | 1,853.19 | 2.189 |
| Flash LSE1 | 1 | 1,807.12 | 2.189 |
| All-reduce CP2 | 2 | 1,500.81 | 1.816 |
| All-reduce CP4 | 4 | 1,489.72 | 1.628 |
| All-gather CP2 | 2 | 1,541.44 | 1.816 |
| All-gather CP4 | 4 | 1,525.77 | 1.628 |

All repeats/rank hashes and unchanged BF16 gates pass. Native/LSE1 are identical; two-rank gather/reduce are identical, while four-rank gather differs from reduce by up to 0.002826. Accuracy is 0.914551 in every arm, but each CP variant changes two predicted labels out of 2,048 versus native (one gain and one loss); unchanged accuracy does not mean unchanged predictions. The nominal gather gain is 2.71% at two ranks and 2.42% at four, from one non-randomized ladder—not established robust acceleration. Neither collective crosses the native or single-rank LSE baseline. Both retain the same 768→384→192 MiB ICL-cache reduction. The [collective evidence snapshot](evidence/resumed-context-collectives-l4-20261008/index.json) preserves all six arms and their independent audits.

### Sequential model placement on L4

This separate comparison uses **KumoTabular large**, E4, context 1,024, queries 2,048, batch 256, and the resident executor. Source `55dba6bb5` passed 36 placement tests on the L4 host, including actual CUDA cases. Each candidate saved exactly the same prediction bytes as its resident one-GPU reference.

| Placement | GPUs | Median rows/s | Max fit peak per GPU | Max prediction peak per GPU | Prediction sum of device peaks |
|---|---:|---:|---:|---:|---:|
| Resident reference | 1 | 1,855.27 | 1.769 GiB | 1.598 GiB | 1.598 GiB |
| Encoder / ICL stages | 2 | 1,763.91 | 1.074 GiB | 1.060 GiB | 1.613 GiB |
| Contiguous ICL layers | 2 | 1,830.08 | 1.248 GiB | 1.071 GiB | 1.614 GiB |
| Contiguous ICL layers | 4 | 1,787.74 | 0.994 GiB | 0.813 GiB | 1.645 GiB |

The four-way layer split lowers the busiest device's fit peak by about 44% and prediction peak by 49%, at a 3.6% warm throughput penalty. This is a capacity tradeoff; the implementation does not overlap pipeline microbatches. Sum of device peaks is not necessarily simultaneous aggregate usage. At this short context, 462.4 MB of the 512.8 MB retained cache belongs to the row encoder on GPU0, so distributing ICL layers alone cannot balance all cache memory.

The larger **Covertype context-16,384** ladder keeps E4, queries 2,048, and batch 256:

| Placement | GPUs | Median rows/s | Max fit allocation | Max prediction allocation |
|---|---:|---:|---:|---:|
| Resident reference | 1 | 1,865.11 | 3.049 GiB | 2.187 GiB |
| Encoder → ICL stages | 2 | 1,830.66 | 2.085 GiB | 1.764 GiB |
| ICL layers | 2 | 1,754.70 | 2.264 GiB | 1.309 GiB |
| ICL layers | 4 | 1,809.19 | 1.876 GiB | 0.876 GiB |

All predictions are byte-identical, with accuracy 0.914551, log loss 0.225671, and macro AUC 0.992549. Four-way placement reduces maximum prediction allocation 59.9% and fit allocation 38.5%, at a 3.0% throughput penalty. The compact-cache control makes no memory reduction here: logical and backing cache bytes both equal 1,132,463,344, so there are no oversized retained views to remove. Its measured 1,799.68 rows/s is not an accepted optimization. Short-context compaction benefits cannot be extrapolated to every cache layout.

Native H&M relational placement extends context to **16,384**, E4, 4,096 queries, batch 512, fixed `[16,16]` two-hop neighborhoods, BF16, and eight CPU threads on the same L4 host:

| Placement | GPUs | Median rows/s | Max fit peak per GPU | Max prediction peak per GPU |
|---|---:|---:|---:|---:|
| Resident reference | 1 | 1,050.73 | 5.322 GiB | 2.111 GiB |
| Encoder/GNN → ICL stages | 2 | 1,066.06 | **4.061 GiB** | 1.650 GiB |
| Contiguous ICL layers | 2 | 1,060.83 | 4.687 GiB | 1.293 GiB |
| Contiguous ICL layers | 4 | 1,057.60 | 4.374 GiB | **0.886 GiB** |

All four outputs are byte-identical; the audit verified complete workload hashes, columns, and member seeds. Churn-positive AUC is 0.670049, accuracy 0.808105, and log loss 0.461313. Stage placement reduces maximum fit allocation by 23.7%, while four-layer placement reduces maximum prediction allocation by 58.0%; the best split depends on the constrained phase. Throughput differs by only 0.7–1.5%, insufficient here to establish a robust speed advantage. The runtime uses CPU joins because cuDF is absent, which can mask GPU stage-latency changes. The GNN remains on one device; these results do not establish distributed GNN computation.

### Capacity stress: H&M context 65,536 on L4

E4, queries 4,096, batch 512, two-hop `[16,16]`, and the same graph/member policy expose an important allocator control. Both default-allocator resident EP1 and four-way ICL placement fail during fit while requesting a 6.10 GiB GNN statistics tensor on GPU0. Stage2 succeeds. However, repeating EP1 and layers4 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` makes **both succeed**, so the original OOM does not prove that this E4 workload fundamentally requires multiple GPUs.

| Arm | Allocator | Outcome / rows/s | Max fit allocation | Max prediction allocation | Quality vs successful EP1 |
|---|---|---:|---:|---:|---|
| Resident EP1 | Default | Fit OOM | Incomplete | — | No predictions |
| ICL layers4 | Default | Fit OOM | Incomplete | — | No predictions |
| Encoder/GNN → ICL stage2 | Default | 955.32 | 16.105 GiB | 6.150 GiB | Byte-identical |
| Resident EP1 | Expandable segments | 921.47 | 20.732 GiB | 6.607 GiB | Reference |
| ICL layers4 | Expandable segments | 944.61 | 17.259 GiB | 2.010 GiB | BF16 gate passes; not exact |

Layer placement reduces prediction allocation by 69.6% against its allocator-matched EP1, but full GNN fit remains on one GPU. Its maximum probability difference is 0.010411, mean 0.001083, with no changed labels; paired log-loss change is −0.0000635 with 95% interval [−0.0002063, +0.0000774]. Reference/stage accuracy is 0.808105 and churn-positive AUC 0.666514. No quality improvement is claimed. Stage comparisons also change allocator policy, so their small throughput difference is not isolated placement acceleration. The raw OOM records, retry settings, successful outputs, and independent audits are preserved. No maximum-context frontier or multi-GPU-only feasible workload is established by these E4 results.

### Paper-inspired bounded GNN follow-up: numerical failures and fit-memory tradeoff

The optional research adapter [destination-blocked GNN](blocked-gnn.md) bounds each five-statistic workspace before the existing projection; `relational_bench.py run --gnn-block-size 16384` selects it, while zero preserves native execution. For the exact capacity graph, the shape-derived statistics allocation changes from 6.102 GiB to at most 80 MiB. This is **not measured GPU peak savings**: full node/source/output states and other model buffers remain.

The integrated synthetic/scoring/protocol suite passed 119 CPU tests. The independent actual-checkpoint GNN-only **CPU** screen, rerun by the coordinator, passed 29 of 31 checks; FP32 singleton blocks failed for both classifier and regressor. Those CPU results do not establish CUDA parity.

Fresh L4 testing at `b721583a9` passes 57 low-level CUDA fixtures but fails **23 of 30 actual-checkpoint GPU cases**. Seven GPU cases pass; the separately counted threshold-policy test makes the full pytest result 8 passed/23 failed. All six single-block controls are bitwise exact. Every genuine 257-node/block64 or block128 checkpoint/precision combination fails the original hidden-output gates. A first-hop diagnostic reconstructs identical statistics but changes projection shape: FP32 maximum difference is 0.0000472 and BF16-autocast maximum difference is 0.125. This supports a shape-dependent projection/reduction-rounding mechanism; it does not excuse the failed gates or validate every internal value. The [GPU evidence](evidence/blocked-gnn-gpu-review-20261008/index.json) retains all 30 case receipts, test XML/logs, exact probe source and outputs, including failures. Independent review reconciles recorded errors and logs; full hidden tensors were not saved for independent numerical recomputation.

An explicitly exploratory full-model H&M pair on the same L4 host uses source `ae4e5e549`, E8, context 16,384, 4,096 queries, batch 512, `[16,16]` graphs and public CPU cache offload. These are new controls, not comparisons to earlier resident E4 placement results:

| GNN execution | Fit seconds | Median rows/s | Fit peak allocation, GiB | Prediction peak allocation, GiB |
|---|---:|---:|---:|---:|
| Native | 24.73 | 558.20 | 4.326 | 1.396 |
| Destination block 16,384 | 24.83 | 549.88 | 3.107 | 1.396 |

Fit peak drops 28.2%; prediction peak is unchanged and throughput is 0.985× native. Every retained repeat is stable. Final probabilities pass the existing BF16 gate with maximum difference 0.010768, mean 0.001582 and zero class flips, but are not byte-identical. Paired log-loss difference is −0.0001093 with interval [−0.0002632, +0.0000479], so no quality improvement is established. Passing this downstream task gate does **not** supersede the failed module gates: keep the adapter opt-in and experimental, not an accepted numerically equivalent replacement. The [full-model pair snapshot](evidence/blocked-gnn-hm16k-l4-20261008/index.json) preserves outputs/audits.

The same-host E8/context65,536 public CPU-offload pair subsequently completes on **one L4**, with matched requested allocator environments. A later separate fresh-process CUDA probe using that runtime/environment reports `expandable_segments=false`: the new `PYTORCH_ALLOC_CONF=expandable_segments:True` request was ignored. Effective-default labeling for the preceding C16k/C64k arms is therefore a **same-runtime/environment inference, not a retrospective per-run measurement**. Within-pair comparisons stand, but earlier prose describing these new runs as verified expandable was incorrect. The [allocator correction receipt and independent review](evidence/resumed-allocator-correction-l4-20261008/index.json) supersede that interpretation without changing original environment requests, records or the historical runs that used `PYTORCH_CUDA_ALLOC_CONF`.

Native versus block16,384 fit peak allocation is 16.737 versus 11.865 GiB (29.1% lower), but peak reservation is 19.342 versus 19.592 GiB (slightly higher). Throughput is 435.78 versus 433.44 rows/s, essentially unchanged. All repeats are stable and final probabilities pass BF16 screening (max 0.008892, mean 0.000949, zero label flips), not byte equivalence. The log-loss interval [−0.0001109, +0.0001025] spans zero. The [64k offload snapshot](evidence/blocked-gnn-hm64k-offload-l4-20261008/index.json) establishes native single-GPU feasibility under that cache/default-allocator policy, **not a multi-GPU-only capacity unlock**. Resident-cache attempts and verified allocator retries are distinct controls; module-level failures still apply.

Both corresponding resident EP1 attempts fail during fit: native requests a 6.10 GiB statistics allocation; the blocked variant progresses to a 2.44 GiB full-node LayerNorm allocation and also fails. The [failure snapshot](evidence/resumed-resident-hm64k-oom-l4-20261008/index.json) retains both failures and independent interpretation; no predictions, throughput or quality exist for these attempts. Memory after exception unwinding is not error-time live allocation. Verified legacy-key retries and stage placement are separate attempts, not permission to overwrite these OOMs.

The [final resident-capacity controls](evidence/resumed-resident-capacity-l4-20261008/index.json) retain stage2 success, EP4 failure and two legacy-key retries whose allocator settings were captured **inside the benchmark process** as `expandable_segments=true`. Both retries still OOM: native at a1.22 GiB skip-projection allocation, blocked at a2.44 GiB LayerNorm allocation. EP4 also OOMs at the6.10 GiB GNN statistics buffer because ensemble splitting does not split a member's full graph. Stage2 completes at512.93 rows/s with bitwise-stable repeats; fit peaks are16.130/12.891 GiB on GPUs0/1 and prediction peaks0.502/12.150 GiB. GPU1 retains12 GiB ICL cache. These device peaks are not simultaneous aggregate memory. No same-policy successful resident1 output exists, so stage2 is not assigned a speedup or numerical-parity claim against one GPU. Its absolute VAL quality is AUROC0.670341, log loss0.460326, accuracy0.808105. Public CPU-offload feasibility remains the simpler single-GPU control; its preprocessing/member policy is not an exact stage2 oracle.

## Completion against the original study plan

This checklist follows the [study questions and completion evidence](study-plan.md), rather than treating implemented code as completion by itself. It distinguishes delivered scope from optional diagnostics and required operational closure. No additional parallelism methods are required to close this study.

| Original objective | Evidence delivered | Remaining gap / explicit limit |
|---|---|---|
| Support and compare both model families | Real KumoTabular Covertype/California and native KumoRelational H&M/F1, including all regression quantiles | Validation prefixes and one fitted context per arm, not full model-quality rankings |
| Test viable multi-GPU decompositions | GPU-measured EP, thread/process query DP, cached CP, stages/layers, threaded 2DP×2EP, pretrained process/graph EP and graph-backed process query DP | Graph throughput covers supported numerical batches256/1,024, not arbitrary schemas or relational graph capture |
| Proper strong and weak scaling | Fixed-work1/2/4 ladders; mixed2/4/8 fixed-family and weak8k/GPU ladders; weighted8 and coordinator-matched controls; all repeats independently audited | Homogeneous eight-GPU/NVLink unavailable; weighted assignment changes hardware-family ownership, and weak8 reuses strong8 |
| Separate native tuning from parallel gains | Native estimator batching, resident-one-GPU controls, fixed local batch-two EP, reverse-order repetitions, explicit input/gather boundaries | Not every method has an identical complete invocation boundary; claim warm execution only where measured |
| Prediction quality and correctness | Fixed row/graph/member identities, positive-class correction, byte comparisons, unchanged BF16 gates, paired metrics and full999 quantiles; resumed full-FP32 F1 CP2 passes its strict gate | Original CP BF16 and FP32-partial F1 failures remain; passing FP32 does not repair or accelerate the default BF16 path |
| Memory and feasible workload capacity | Per-device allocator peaks/cache/RSS; E8/64k CPU-offload fits one L4; resident stage2 fits while EP1/EP4 and verified-expandable EP1 retries fail | Resident-policy feasibility benefit, not intrinsic multi-GPU requirement; no successful same-policy resident1 quality oracle; allocated savings do not imply lower reservation |
| Profile and explain successes/failures | Separate Nsight EP/CP and paired graph-replay traces; independently checked launch counts, device overlap, cache traffic, collective waiting, autocast/compaction interventions | Full frontend/ICL phase attribution and lost original placement traces remain unavailable; no GIL-only, occupancy or pure-bandwidth causal claim |
| Native relational setup and pipeline costs | Fixed complete temporal graphs, sampling/load/fit timing, matched Arrow/cuDF arms audited and slower with cuDF | Whole-pipeline one-shot latency and service-tail latency are not uniformly measured; cuDF first-saved/last-scored repeat mismatch is retained explicitly |
| Go beyond prior user/Aki work | Native Kumo adaptations, batching/process isolation, explicit CP global-length/GQA handling, placement/capacity work, analyzed TP/table/graph/full-fit options | Analyzed-only options remain unimplemented, with architecture/communication reasons documented; no implied benchmark |
| Generic SDM integration | Tested candidates and public-API examples; small extraction boundaries in integration.md; source inventory and scoped integrated checks in handoff.md | Study harness is research, not production API; blocked GNN fails 23 actual-checkpoint GPU cases despite its downstream probability gate passing |
| Reproducibility and failures | Raw small records, exact commands where recorded, hashes for local predictions/profiles, hardware/runtime/checkpoint receipts and failures; all47 indexes pass archived/external verification; published study and minimal branches | Original remote-only artifacts were lost, not measured numerical failures; experimental harness is not production-ready API |
| Resource closure | Both replacement instances terminated, roots absent, four-region task instance/EBS/SG/key inventories empty; Spot requests closed/cancelled; receipts retained | Cost is quote integration/bounds, not an invoice; original L4 exact end remains unknown; unrelated resources and user keys preserved |

Full-FP32 F1 and alternative-collective CP diagnostics were never launched on the original hosts, but now have independently audited replacement-host results above. Mixed-host performance and all10 resumed relational capacity outcomes are also audited. The optional32k MHA diagnostic remains unmeasured; no crossover is claimed. Original E8 remote-only capacity evidence was lost and is not substituted for fresh controlled runs. Operational cleanup and local evidence verification remain separate obligations.

**Resumed lifecycle:** checks beginning 2026-10-08 01:34:46 UTC verified the original instances and task-tagged disks absent. Undownloaded original result/trace artifacts are therefore lost, not recoverable pending runs and not numerical failures. Replacement L40S execution requires fresh attempts and same-host controls; no new performance or eight-GPU result is implied. The [handoff](handoff.md) records the receipt, surviving evidence, resumed CPU validation and still-open publication/cleanup obligations.

## Retained evidence

The final local verification sweep on2026-10-08 passed **all47 snapshot indexes: 495 archived-file references and1,229 original-artifact references, zero mismatches or missing files**. The final snapshot adds the immutable teardown and cost receipts; thirteen mixed-host runs, family/scaling audits, resident capacity controls and allocator failures, larger-batch query DP, preloaded local-pipe controls, GNN failures, traces, CP controls and [both replacement runtimes](evidence/resumed-runtime-20261008/index.json) remain unchanged. The earlier GNN screen remains labeled CPU-only. Runtime collection allowlists just five safe files per host: software inventory, GPU identity/topology and verified model-file hashes, not weights or data. Counts include references repeated across additive snapshots, not necessarily unique physical files. Local verification proves retained bytes; the separate final operational receipt establishes resource closure, not recovery of never-downloaded original artifacts.

The [initial tabular evidence index](evidence/initial-tabular-20261008/index.json) archives all seven small raw result JSON files in the repository. It binds predictions, row/target identity arrays, and available telemetry to SHA-256 hashes and their original local paths. The index and all 34 original artifacts passed verification at collection; weights and raw datasets are excluded. Large artifacts must remain in the local `.kumo-multigpu-20261008` result store or be copied to a durable user-selected location before that local store is removed. EC2 teardown does not remove these downloaded local files.

Additional checked snapshots retain the new measurements:

- [Initial relational L40S evidence](evidence/initial-relational-l40s-20261008/index.json): 29 archived records, 62 external artifacts verified; includes corrected class-aware quality audits and the separately labeled E1 smoke.
- [Relational backend controls](evidence/relational-backend-l40s-20261008/index.json): 18 archived result/audit/backend records, 30 external artifacts verified; saved-first-array quality and logged-last-repeat caveat preserved.
- [Initial context L4 evidence](evidence/initial-context-l4-20261008/index.json): 14 archived records, 36 external artifacts verified; includes native/LSE/CP full-model runs and both kernel probes.
- [Initial placement L4 evidence](evidence/initial-placement-l4-20261008/index.json): four archived records, 20 external artifacts verified.
- [Batched ensemble L40S evidence](evidence/batched-ensemble-l40s-20261008/index.json): eight archived result/audit records, 24 external artifacts verified; includes the passing EP2 and failing EP4 comparisons.
- [Ensemble batching controls](evidence/ensemble-batch-controls-l40s-20261008/index.json): 12 archived result/audit records, 36 external artifacts verified; isolates local batching arithmetic and preserves the stronger one-GPU baseline.
- [Ensemble cache controls](evidence/ensemble-cache-controls-l40s-20261008/index.json): six archived result/audit records, 18 external artifacts verified; exact-output autocast and compact-cache interventions.
- [Context F1 L4 evidence](evidence/context-f1-l4-20261008/index.json): 16 archived result/rank/audit records, 32 external artifacts verified; preserves full-quantile failures and paired quality analysis.
- [F1 FP32-partial correction evidence](evidence/context-f1-fp32partial-l4-20261008/index.json): 13 archived result/rank/audit records, 25 external artifacts verified; both original-native and matching-kernel comparisons remain failures.
- [Process DP L40S evidence](evidence/process-data-l40s-20261008/index.json): six archived result/audit records, 18 external artifacts verified.
- [Tuned and relational process DP evidence](evidence/tuned-process-data-l40s-20261008/index.json): ten archived result/audit records, 27 external artifacts verified; includes matched estimator-batch-four quality references.
- [Weak process DP evidence](evidence/weak-process-data-l40s-20261008/index.json): six archived result/audit records, 18 external artifacts verified; current equivalence audit covers common prefixes.
- [H&M 16k placement evidence](evidence/placement-hm16k-l4-20261008/index.json): 12 archived result/audit/command records, 24 external artifacts verified.
- [Tabular 16k placement evidence](evidence/placement-tabular16k-l4-20261008/index.json): ten archived records, 30 external artifacts verified, including the no-benefit compact-cache control.
- [H&M 64k capacity evidence](evidence/placement-capacity-hm64k-l4-20261008/index.json): 18 archived attempt/result/failure/audit/command records, 34 external artifacts verified; original OOMs and allocator retries remain separate.
- [Resident 16k context L4 evidence](evidence/context-large16k-l4-20261008/index.json): 12 archived records, 28 external artifacts verified.
- [Native H&M 16k context evidence](evidence/context-hm16k-l4-20261008/index.json): 12 archived records, 24 external artifacts verified; actual input dimensions retained alongside stale CLI defaults.
- [Initial Nsight evidence](evidence/initial-nsys-l40s-20261008/index.json): two archived analysis records, four external analysis/SQLite artifacts verified.
- [Long-context CP Nsight evidence](evidence/context-nsys-l4-20261008/index.json): one archived analysis record and four verified external artifacts, including the SQLite database and original trace.
- [Base runtime receipts](evidence/runtime-base-20261008/index.json): ten explicit small records containing both hosts' package freezes, GPU/driver topology, runtime versions, and checkpoint verification; cuDF overlays are separate environments.
- [Regression ensemble evidence](evidence/regression-ensemble-l40s-20261008/index.json): eight archived records, 24 external artifacts verified; complete 999-quantile parity and negative scaling.
- [Threaded hybrid evidence](evidence/hybrid-l40s-20261008/index.json): four archived records, 12 external artifacts verified; matched CPU query-input baseline.
- [Fully gathered process evidence](evidence/gathered-process-data-l40s-20261008/index.json): four archived records, 12 external artifacts verified; preserves both worker-return and final-gather times.
- [Full-cohort process equivalence](evidence/fullcohort-process-data-l40s-20261008/index.json): five archived records, 13 external artifacts verified; additive audits establish complete 16,384-row equivalence.

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

Verify every snapshot from the repository root, using a Python environment that can import the study scripts:

```sh
python - <<'PY'
from pathlib import Path
from research.multigpu.collect_evidence import verify

indexes = sorted(Path("research/multigpu/evidence").glob("*/index.json"))
for index in indexes:
    print(index.parent.name, verify(index, external=True))
print(f"Verified {len(indexes)} snapshots")
PY
```

## Reproduction and failure log

The initial CP prototype (`8e22f29c8`) had three passing CPU tests exercising real 2/4-rank Gloo groups, both model ICL blocks, MHA/GQA, broadcast batches, empty/uneven shards, cache backing-storage release, and invalid topology. Those initial tests established CPU correctness only; later CUDA/NCCL tests and pretrained measurements are reported above. See the implementation's `research/multigpu/context-parallel.md` for its current exact scope.

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

Expected harness artifacts are `result.json`, predictions, utilization samples, and a profiler trace. Native complete graph batches are prepared once, with sampling time recorded separately.

The process-DP throughput findings use the following explicit worker settings. Repeat each command with one GPU and a fresh output directory for the matched executor reference. Relational inputs remain the same previously prepared complete batches.

```sh
python research/multigpu/tabular_process_bench.py \
  --data /path/to/covertype --output /path/to/tabular-dp4-result \
  --task classification --size large --gpus 4 --threads 1 \
  --context 1024 --queries 2048 --batch-size 256 \
  --estimators 4 --estimator-batch-size 4 --precision bfloat16 \
  --warmups 1 --repeats 3 --seed 1729 \
  --source-commit "$(git rev-parse HEAD)"
python research/multigpu/relational_process_bench.py \
  --workload /path/to/fixed-workload --output /path/to/relational-dp4-result \
  --gpus 4 --threads 8 --estimators 4 --estimator-batch-size 1 \
  --dtype bf16 --warmups 1 --repeats 3 --seed 1729 \
  --source-commit "$(git rev-parse HEAD)"
```

These example commands describe the current interface; the archived source/runner hashes identify the exact measured implementation. Later versions add separately timed CPU gather fields and must not retroactively change the original timer definition.

The independent CP probe uses random weights and is a kernel/scaling diagnostic, not a quality benchmark:

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
