# Native KumoRelational multi-GPU measurements

The relational benchmark consumes RelBench database tables as `RelationalData`, samples complete temporal two-hop subgraphs, and passes the resulting `RelatedTables` directly into `KumoRelational`. It does not flatten neighborhoods into tabular features.

## Workloads and fixed semantics

| Workload | Purpose | Target | Context ladder | Query workload |
|---|---|---|---|---|
| `rel-f1/driver-position` | Fast regression validation and scaling controls | `position` | 1,024 and 4,096 TRAIN observations | All 499 VAL observations, batches of 125 |
| `rel-hm/user-churn` | Large relational database and classification | `churn` | 1,024, 4,096, and 10,000 TRAIN observations | 2,000 initial VAL observations, then larger VAL prefixes |

The dataset owner provides the deterministic TRAIN row permutation. Every context size takes a prefix of that permutation. Queries preserve the original VAL order. Sampling uses exactly `[16,16]`, temporal strategy `last`, and the observation timestamp. No target-derived lags are added. Only TRAIN labels enter fitting; query tables contain entity IDs and timestamps. Validation labels are stored in a separate file and loaded for scoring after measured inference finishes. TEST is unused.

The sampler constructs disjoint per-observation neighborhoods with `__example__` included in relationship keys. Each prepared query batch is the indivisible unit for data parallel scheduling. Batch boundaries, related rows, key values, and graph identities remain the same across GPU counts. Splitting related tables by arbitrary row ranges would violate this contract.

## Comparison arms

| Arm | Execution | Main comparison |
|---|---|---|
| Native | Public model fit/predict on one GPU | User-facing baseline, including the native E>1 CPU cache offload policy |
| Ensemble 1 | Resident executor on one GPU | Control for executor/cache policy before attributing gains to distribution |
| Ensemble 2/4 | Same member recipe and RNG plan over two/four GPUs | Placement scaling with identical sampled graphs |
| Data 1 | One fitted replica, original query batches | Scheduler overhead control |
| Data 2/4 | Full identical fitted replicas, whole batches distributed | Throughput scaling without repartitioning graphs |
| Context/model/hybrid | Added only when model adapters support real relational inputs | Record exact implementation and compare to its corresponding one-GPU control |

Native preprocessing currently runs on CUDA. Ensemble preprocessing runs once on CPU, with an explicit seeded CPU generator and stable member seeds. Therefore native-versus-ensemble output differences may come from preprocessing/RNG policy and must be reported separately from placement parity. The ensemble 1/2/4 comparison preserves that policy. Data replicas use identically seeded device generators; raw parity must still be checked rather than assumed.

## Measurements

Each arm runs in a fresh process. Device synchronization bounds model loading, context fit/cache, warmup, every native/ensemble batch, and each full measured pass. At least one warmup pass and three measured passes are required. Prediction arrays and table column labels are retained. Throughput reports each repetition, not only the best run. Data-parallel per-worker times are service times excluding queue delay; synchronized whole-pass time is the throughput denominator.

The preparation report separates database tensorization, sampler construction, context sampling, and query sampling. Inference timings include CPU-to-GPU inputs, model prediction, and GPU-to-CPU outputs, but reuse prepared neighborhoods. Thus inference throughput is explicitly conditioned on prepared graphs. Adding previously measured sampler time to prediction time is an estimate, not a measured pipelined end-to-end result.

Memory reports load, fit, and prediction allocator peaks per GPU; fit and prediction reset the peak counters independently. Host maximum RSS and 200 ms `nvidia-smi` utilization/memory/power observations supplement allocator numbers. Native caches report tensor bytes. A separate optional profiler pass contains row embedding, GNN, and ICL ranges; the profiled pass is excluded from throughput results.

Classification scoring includes AUROC, log loss, and accuracy. Regression includes median MAE/RMSE plus nonfinite quantile counts and quantile crossing fraction. Raw predictions support maximum/mean error and rank/argmax agreement comparisons. Model revision, package versions, source commit, context IDs, query IDs, and sampled graph hashes accompany each result.

## Commands

Run from the repository root with `PYTHONPATH=.`. Preparation requires CPU `pyg-lib` compatible with the installed PyTorch, pandas, and pyarrow; scoring uses scikit-learn.

```bash
python research/multigpu/relational_bench.py prepare \
  --raw /data/rel-f1 --task driver-position \
  --entity driverId --entity-table drivers --time date --target position \
  --problem regression --context 1024 --queries 499 --batch-size 125 \
  --output /results/workload-f1-c1024-b125

python research/multigpu/relational_bench.py prepare \
  --raw /data/rel-hm --context 1024 --queries 2000 --batch-size 250 \
  --output /results/workload-hm-c1024-b250

python research/multigpu/relational_bench.py run \
  --source-commit SOURCE_COMMIT \
  --workload /results/workload-hm-c1024-b250 \
  --mode native --gpus 1 --estimators 4 --repeats 3 --warmups 1 \
  --output /results/hm-native-e4

python research/multigpu/relational_bench.py run \
  --source-commit SOURCE_COMMIT \
  --workload /results/workload-hm-c1024-b250 \
  --mode ensemble --gpus 4 --estimators 4 --repeats 3 --warmups 1 \
  --profile --output /results/hm-ensemble4-e4
```

No result is claimed in this protocol. Successful measurements, failures, OOM boundaries, exact hardware, and scaling tables belong in the result report after execution.
