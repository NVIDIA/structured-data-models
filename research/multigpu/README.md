# Kumo multi-GPU investigation

Status: investigation in progress, 2026-10-08. Implementation baseline: `842c408fe`. This report separates measured results from architectural expectations. Blank result cells mean that a measurement has not arrived; they do not mean zero, parity, or success.

The goal is practical multi-GPU inference for both `KumoTabular` and `KumoRelational`, including throughput, latency, capacity, prediction quality, and a lightweight integration into SDM. The implementation recommendations are in [integration.md](integration.md).

## Approach matrix

| Approach | Work placement | Expected benefit | Main limitation / correctness requirement | Current owner and evidence |
|---|---|---|---|---|
| Query data parallelism (DP) | Complete, fixed query batches across persistent replicas | High aggregate throughput; no attention collectives | Full model and fitted context replicated; preserve batch boundaries and complete relational neighborhoods | `data_parallel_impl`, `cb92ac35d` + `a52ee76db`; ten CPU tests passed including actual model hybrid; GPU measurements pending |
| Ensemble parallelism (EP) | Fixed logical estimator IDs across replicas | Reduce member work and member cache per GPU; small output gather | Same recipes, member RNG, class alignment, output reduction, and cache residency as reference | Initial two CUDA failures fixed by `c4ddd7f15`; both CUDA tests then passed on integrated `6c3f1e22e`; scaling measurements pending |
| Cached context parallelism (CP) | ICL KV rows across ranks; queries replicated | Larger retained context and faster long-context attention | Full fit still replicated; stable softmax reduction, global length scaling, uneven shards; collectives every layer | `context_parallel_impl`, commit `8e22f29c8`; 2/4-rank CPU tests passed, GPU measurements pending |
| Layer/stage placement | Row encoder, GNN, and/or ICL layers on different devices | Parameter/cache capacity; potential pipeline overlap across query batches | Single-query latency can worsen; cache ownership and transferred activations must follow stages | `model_parallel_impl`, `70db412fa` + `80587a987` + `8f293c8f4`; 14 CPU tests passed, 16 CUDA tests pending; current implementation is sequential stage placement, not overlapped pipelining |
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
| Host 2 | Spot four-L4 host, acquired for CP/placement; detailed topology pending | Pending runtime receipt | Separate hardware ladder; do not combine raw speedup denominators with Host 1 |

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
| KumoTabular | DP / CP / stages / tuned EP | 1 / 2 / 4 | Pending | — | — | — | — | — | Pending |
| KumoRelational | Public baseline | 1 | Pending | — | — | — | — | — | Pending |
| KumoRelational | EP / DP / CP / stages | 1 / 2 / 4 | Pending | — | — | — | — | — | Pending |

The first native fit took 122.751 seconds; a repeated native E4 fit with estimator batch size one took 1.327 seconds. Thus the cold first fit is not an appropriate denominator for an EP fit speedup. First-use compilation/library initialization remains a hypothesis for the cold delay. The result's CPU-offloaded cache is explicitly recorded; EP comparisons require their resident one-GPU control. Independent quality review confirmed the baseline metrics and stable repeat outputs.

EP batch p50 was 147.96 / 135.90 / 155.55 ms at 1/2/4 GPUs. Scaling efficiency was 54.4% at two GPUs and 23.8% at four. All four saved prediction arrays, including native, are byte-identical (`92cb62ec…`). EP resident cache storage distributes as 489.00 / 244.50 / 122.25 MiB per GPU, while replicated parameters make aggregate device memory increase. Summed per-rank peaks are an upper bound on simultaneous aggregate use, not a synchronized aggregate trace.

The EP cache's 489 MiB unique backing storage exceeds the public CPU cache's 342 MiB, despite identical predictions; retained view backing allocations versus compact CPU copies are under investigation. CPU dispatch/launch overhead and short member work are candidate explanations for poor scaling, pending profiles. The next experiments compare native estimator batching, larger contexts/batches/E8, batched EP, and process-based execution. The historical TabFM 1.876× result does not predict these Kumo results.

The tuned one-GPU comparison uses the same large model, Covertype rows, E4, context 1,024, query 2,048, query batch 256, and BF16 autocast. Only estimator batching changes:

| Native estimator batch | Median rows/s | Batch p50 | Fit seconds, warmed environment | Accuracy | Log loss | Max probability difference vs estimator batch 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,758.63 | 145.36 ms | 1.327 | 0.764160 | 0.576045 | 0 |
| 2 | 2,984.01 | 85.76 ms | 1.253 | 0.764160 | 0.575996 | 0.008804 |
| 4 | 4,007.14 | 63.82 ms | 1.255 | 0.764160 | 0.575978 | 0.005607 |

Predicted class labels are unchanged for all 2,048 rows. Estimator-batched BF16 outputs are not byte-identical; mean absolute probability differences are about 0.000263 and 0.000259. These quality changes must remain visible when comparing speed against the sequential member path. Best measured native batching is 2.28× the repeated unbatched native run and 2.13× the initial two-GPU EP. Multi-GPU follow-ups must include local estimator batching before claiming a useful advantage.

## Retained evidence

The [initial tabular evidence index](evidence/initial-tabular-20261008/index.json) archives all seven small raw result JSON files in the repository. It binds predictions, row/target identity arrays, and available telemetry to SHA-256 hashes and their original local paths. The index and all 34 original artifacts passed verification at collection; weights and raw datasets are excluded. Large artifacts must remain in the local `.kumo-multigpu-20261008` result store or be copied to a durable user-selected location before that local store is removed. EC2 teardown does not remove these downloaded local files.

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

The native relational harness exposes the following commands (paths must point to the installed runner revision and the actual RelBench cache). Execution evidence is still pending:

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

These commands assume execution from a Git checkout. For an exported source archive, pass its recorded commit explicitly instead of invoking `git rev-parse`. The coordinator's integrated local validation currently reports 43 CPU tests passed and 22 CUDA tests skipped; this is not a GPU validation result.

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
| Assume cached CP solves fit OOM | Rejected by code inspection | Aki branch computes full fit then shards cached ICL KV | Measure fit peak separately; investigate full-fit sharding |
| Slice relational neighbors by query row indices | Invalid graph transformation | Related-table row spaces differ and shared neighbors cross task rows | Freeze complete sampled graph per batch |
| Suspected singular relationship metadata keys | Valid supported alias; no defect | Quality reviewer inspected `sdm/relational/data.py` and `task.py` | No change needed |
| Initial EP CUDA stream tests | Failed: unavailable `TableTensor._tensors()` | Initial host test log: two failed, eight passed | Fixed by `c4ddd7f15`; both CUDA tests passed on `6c3f1e22e` |
| CP full-model runner evidence audit | Six reporting/boundary gaps found | Cache residency, requested/actual graph sizes, silent truncation, eager input memory, failed-run artifacts, repeat parity | `250cae105` adds explicit evidence; inputs remain GPU-resident consistently across native/LSE/CP |
| Eight-GPU/NVSwitch capacity | Unavailable under observed regional Spot quota | Coordinator reports 64-vCPU regional limit | Use separate four-L40S and four-L4 ladders; no NVLink conclusion |
