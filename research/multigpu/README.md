# Kumo multi-GPU investigation

Status: investigation in progress, 2026-10-08. Implementation baseline: `842c408fe`. This report separates measured results from architectural expectations. Blank result cells mean that a measurement has not arrived; they do not mean zero, parity, or success.

The goal is practical multi-GPU inference for both `KumoTabular` and `KumoRelational`, including throughput, latency, capacity, prediction quality, and a lightweight integration into SDM. The implementation recommendations are in [integration.md](integration.md).

## Approach matrix

| Approach | Work placement | Expected benefit | Main limitation / correctness requirement | Current owner and evidence |
|---|---|---|---|---|
| Query data parallelism (DP) | Complete, fixed query batches across persistent replicas | High aggregate throughput; no attention collectives | Full model and fitted context replicated; preserve batch boundaries and complete relational neighborhoods | `data_parallel_impl`, `cb92ac35d` + `a52ee76db`; ten CPU tests passed including actual model hybrid; GPU measurements pending |
| Ensemble parallelism (EP) | Fixed logical estimator IDs across replicas | Reduce member work and member cache per GPU; small output gather | Same recipes, member RNG, class alignment, output reduction, and cache residency as reference | `ensemble_impl`, `d1775c0f3` + `f74f2cc30`; eight CPU tests passed, two CUDA tests pending |
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

No new Kumo GPU measurements have been supplied to this report yet.

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
| KumoTabular | Public baseline | 1 | Pending | — | — | — | — | — | Pending |
| KumoTabular | EP / DP / CP / stages | 1 / 2 / 4 | Pending | — | — | — | — | — | Pending |
| KumoRelational | Public baseline | 1 | Pending | — | — | — | — | — | Pending |
| KumoRelational | EP / DP / CP / stages | 1 / 2 / 4 | Pending | — | — | — | — | — | Pending |

## Reproduction and failure log

The CP prototype (`8e22f29c8`) has three passing CPU tests exercising real 2/4-rank Gloo groups, both model ICL blocks, MHA/GQA, broadcast batches, empty/uneven shards, cache backing-storage release, and invalid topology. This establishes distributed CPU correctness within those tests. It does not establish CUDA kernel compatibility, model quality, or GPU speedup. See the implementation's `research/multigpu/context-parallel.md` for its exact scope.

The EP implementation (`d1775c0f3` + `f74f2cc30`) has eight passing CPU tests, including reduced real KumoTabular modules with 12-class ECOC, class shuffling, exact member codebooks/predictions across 1/2/4 replicas, and refit. Two CUDA tests await hardware execution. Independent review found and the owner fixed replica-initialization stream dependencies and output tensor stream lifetime; a CPU pass cannot validate those CUDA fixes. Its public documentation is `research/multigpu/ensemble.md` and the independent comparison checklist is `research/multigpu/quality.md` (`bf3a602e8`).

The tabular runner (`af6d89791` + `10159e909`) provides the following initial comparison. Repeat for native estimator batch sizes that fit memory, then `--mode ensemble --gpus 1`, `2`, and `4`, each with a fresh output directory. CPU-complete outputs and profiler-only passes are separate from throughput measurements.

```sh
python research/multigpu/tabular_bench.py \
  --data /path/to/covertype --output /path/to/tabular-native \
  --task classification --size large --mode native --gpus 1 \
  --estimators 4 --context 1024 --queries 2048 --batch-size 256 \
  --repeats 3 --precision bfloat16 --profile
```

The native relational harness exposes the following commands (paths must point to the installed runner revision and the actual RelBench cache). Execution evidence is still pending:

```sh
python research/multigpu/relational_bench.py prepare \
  --raw /path/to/rel-hm --output /path/to/fixed-workload \
  --context 1024 --queries 2000 --batch-size 250
python research/multigpu/relational_bench.py run \
  --workload /path/to/fixed-workload --mode native --gpus 1 \
  --estimators 4 --repeats 3 --output /path/to/native-result --profile
python research/multigpu/relational_bench.py run \
  --workload /path/to/fixed-workload --mode ensemble --gpus 2 \
  --estimators 4 --repeats 3 --output /path/to/ep2-result --profile
```

Expected harness artifacts are `result.json`, predictions, utilization samples, and a profiler trace. Native complete graph batches are prepared once, with sampling time recorded separately. The independent CP probe uses random weights and is a kernel/scaling diagnostic, not a quality benchmark:

```sh
torchrun --standalone --nproc-per-node=4 research/multigpu/context_probe.py \
  --family tabular --context 4096 --queries 128 --channels 512 \
  --heads 8 --kv-heads 2 --layers 4 --dtype bfloat16
```

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
