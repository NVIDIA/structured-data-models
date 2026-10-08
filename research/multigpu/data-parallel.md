# Query data parallelism for Kumo models

## Implemented scope

`query_parallel.py` provides two research executors over identical, independently fitted predictors. `QueryParallel` uses one persistent thread per replica. `ProcessQueryParallel` uses one spawned process per replica and a caller-supplied factory to construct and fit it. Both accept indivisible prepared `QueryBatch` objects and return CPU `QueryResult` objects in the original batch order. No gradient synchronization or DDP wrapper is needed for inference.

`data_parallel_adapter.py:factory(args, replicas)` adapts explicitly loaded native Kumo replicas to the benchmark runner. It copies the complete incoming generator state separately into each device-local generator before fitting, preserves the same context and recipe, and exposes `predict_batches`. Calling `predict` on a single batch intentionally uses one worker: this technique improves independent-batch/request throughput, not latency of one indivisible batch.

The source reference was the existing `examples/kumo/relational/multi_gpu.py` example (commit `6aa9f4f14d203026c89eacbc945696bfac233a75`). That example already gets the important graph and scheduling semantics right. The present prototype generalizes the contract to both Kumo families, adds threaded and spawned execution, preserves observation IDs, and composes with the ensemble executor.

## Correctness contract

All replicas must have identical checkpoint parameters, model configuration, context targets, fitted recipe/member plan, precision and context-fitting RNG state. A caller-supplied custom recipe must use its explicit generator rather than process-global randomness. Reproducing only the initial random seed after partially consuming a generator is insufficient; the adapter copies `get_state()` before each fit. Generator device type must agree with the input/model device type, because CPU and CUDA RNG algorithms differ.

Every scheduling unit contains the exact original task rows and complete sampled neighborhood. The executor never slices related tables using task row positions and never changes batch boundaries. This is necessary for `TaskGraph.from_input`: task rows must map to distinct entity rows, and task assignments propagate through nonoverlapping neighborhoods. The materializer must retain per-observation composite keys such as `__example__`, temporal filtering, relationship schema and task links. Repeated entities at different prediction times need separate observation IDs and neighborhood copies. Arbitrary repartitioning of one raw relational graph is not an equivalent workload.

Input storage must be CPU resident and remain unchanged until completion. Worker calls explicitly enter inference mode and autocast, because these contexts are thread-local. A single-thread queue protects each model's fitted state from concurrent access, including multiple submission callers. CPU output copies synchronize completion, and output row count and cross-batch column schemas are checked. Futures propagate worker errors. Models cannot be shared between worker slots. External concurrent calls or refits on the same model remain unsupported.

Callbacks are not exposed by this research scheduler. Models or processors that mutate global state, consume untracked randomness at prediction time, or depend on the order of query batches require additional auditing. Preserving the batch contents is necessary but does not prove such custom behavior safe.

## Running the threaded adapter

```python
from research.multigpu.query_parallel import QueryBatch, QueryParallel

# replicas have already been identically fitted on their respective devices.
# batches are sampled/materialized exactly once, before varying GPU count.
with QueryParallel(replicas, ["cuda:0", "cuda:1"]) as executor:
    outputs = executor.predict([
        QueryBatch(tuple(row_ids), task_table_cpu, related_tables_cpu)
        for row_ids, task_table_cpu, related_tables_cpu in batches
    ])
# outputs retain input batch order, row IDs and prediction column order.
```

For tabular-only data, omit `related_tables_cpu`. For spawned workers, supply an importable module-level factory `(worker, device) -> fitted_model`, instantiate `ProcessQueryParallel(factory, devices)`, and call `ready()` before warm timing. Use the usual `if __name__ == "__main__":` guard. The factory must use the same seed/state and context for every worker, rather than adding rank to the seed. `num_threads=1` limits child CPU intra-op oversubscription.

The actual runner factories live in `process_factories.py`: `TabularProcessFactory(data, context, task, size, estimators, seed, precision)` reads only the prepared TRAIN arrays, while `RelationalProcessFactory(graphs, target, task, estimators, seed, precision, num_hops)` fits the exact prepared context graph. Both load pretrained weights within each spawned process. Call `executor.memory(reset_peak=True)` after fit/warmup and `executor.memory()` after a prediction pass to obtain each child's allocator statistics. The parent's `torch.cuda.memory_allocated()` does not measure child allocations. External host/process RAM and device utilization sampling remains necessary.

The standalone runners use the same prepared artifacts as the single-process runners:

```bash
python -m research.multigpu.tabular_process_bench \
  --data /path/to/prepared-tabular --task classification --size large \
  --gpus 4 --context 1024 --queries 2048 --batch-size 256 --estimators 4 \
  --seed 1729 --precision bfloat16 --threads 1 --warmups 1 --repeats 3 \
  --output /path/to/new-result --source-commit REVISION

python -m research.multigpu.relational_process_bench \
  --workload /path/to/prepared-relational --gpus 4 --estimators 4 \
  --dtype bf16 --seed 1729 --threads 1 --warmups 1 --repeats 3 \
  --output /path/to/new-result --source-commit REVISION
```

Both scripts record startup/import/checkpoint/fit together as `spawn_load_fit_s`, then warmup separately, then repeated full-pass wall times, CPU predictions, worker service times, child allocator peaks and `nvidia-smi` telemetry. They terminate worker models before opening validation labels. Worker `max_rss_bytes` is a lifetime host-memory peak; summing worker peaks is not necessarily the simultaneous node peak and can double count shared pages. Compare process 1/2/4 at identical `--threads`, then consider an additional matched node-wide CPU-thread budget; a process-versus-thread speed difference alone does not prove a GIL cause.

## Hybrid 2 DP × 2 EP on four GPUs

Create and fit two independent `EnsembleParallel` groups, one on GPUs `[0,1]`, the other on `[2,3]`. Both groups must represent the same full ensemble member plan. Pass the two group executors to `QueryParallel`, with input devices `cuda:0` and `cuda:2`. Whole query batches alternate between groups, while members within a batch execute concurrently on the group's two GPUs. Construct/finalize the groups before creating the outer executor and close the outer executor before closing the groups.

This composition has two copies of the full ensemble cache across the node, rather than four copies for pure 4-way DP. Each GPU still has its own complete model parameters. The expected advantage is lower per-GPU member-cache pressure than pure DP and higher batch throughput than single-group EP. It introduces extra CPU recipe work and two independent group coordinators; GPU measurements, parity checks and cold-start costs must establish whether it is actually better.

The runnable adapter is `data_parallel_adapter.py:hybrid_factory(args, replicas)`. It accepts the same benchmark factory interface as native DP, creates two equal contiguous replica groups, fits identical recipes with the same cloned generator and `member_seed`, and exposes `predict_batches`. Four supplied replicas produce the intended 2 × 2 layout; two replicas produce two singleton resident ensemble groups, a useful resident-cache DP control. Closing the adapter drains outer work before closing both inner executors. CPU tests verify exact output parity for the actual KumoTabular model with a reduced random-weight architecture; they exercise the nested scheduler, not GPU scaling.

Set `args.recipe_device="cpu"` and pass a CPU fit generator when comparing against a CPU-preprocessed resident EP reference. The adapter then preserves CPU recipe execution while explicitly propagating CUDA autocast into both nested worker levels. The default keeps preprocessing on the group's first GPU and requires a CUDA generator. Changing CPU/CUDA recipe backends can change sampling or preprocessing numerics, so those configurations are separate references, not pure placement comparisons.

### Optional graph replay inside process DP

`tabular_process_bench.py --backend graph` composes the existing one-GPU `GraphEnsembleParallel` with each persistent query worker; `native` remains the default. Each worker fits the same complete resident ensemble and member seed. Graph replay executes one member at a time, so the report explicitly sets its effective estimator batch size to 1 rather than pretending that a native `--estimator-batch-size 4` flag batches graph members. Compare against both matching graph-one-GPU execution and the best tuned native baseline.

All fixed batches, including an irregular final batch, run during warmup. Child reports retain graph count, cumulative capture time/events and allocator peaks. The runner rejects any new capture during timed prediction passes. Worker shutdown explicitly closes the nested graph executor before ending the process, and output gathering retains the same CPU-complete timing boundary and all-repeat archives as native process DP. The optional backend is a separate source revision and does not change the frozen native eight-GPU study.

CPU validation passes the existing process contracts and explicit nested-executor cleanup. CUDA tests cover 1/2/4 devices with batches 256/1024 plus a partial final batch, prediction tolerance against the same native member plan, stable graph counts across three repeats and ownership of already-returned outputs. These CUDA cases are skipped on CPU-only machines; their presence is not a measured graph-DP scaling result.

Use GraphDP1 as the exact graph-scaling oracle. Native-versus-graph prediction equivalence is a separate empirical check: ensemble recipe/member RNG consumption can differ for larger class counts or regression target transformations. The small random-weight, three-class CUDA fixture does not establish generic equivalence for those cases. The planned pretrained Covertype comparison uses all seven classes, a fixed 8192-row cohort and batch size 256, with native estimator-batch-4 controls at each GPU count and all prediction repeats retained.

### Measured graph replay plus process DP

The follow-up was executed on the replacement four-L40S Ohio Spot host with source `ae4e5e549`: pretrained KumoTabular large, Covertype TRAIN context 1024, E4, Q8192, query batch 256, seed 1729, BF16, one CPU thread per process, one complete warmup and three measured passes. The six CUDA contract cases passed on actual 1/2/4 GPU configurations in 55.06 seconds before the pretrained measurements. Both algorithms consume prepared CPU batches and return fully ordered, concatenated CPU outputs inside the reported timing boundary.

| GPUs | Native estimator-batch-4 rows/s | Graph process DP rows/s | Graph speedup over GraphDP1 | Graph/native speedup |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3744.04 | 6074.73 | 1.000× | 1.623× |
| 2 | 7505.81 | 12072.97 | 1.987× | 1.608× |
| 4 | 14341.62 | 22667.07 | 3.731× | 1.581× |

These are medians of fully gathered throughput, not worker-service sums. GraphDP4 retained 93.28% of ideal four-GPU throughput relative to GraphDP1. Independent audits checked every saved prediction repeat: GraphDP2/4 were bitwise identical to GraphDP1, and the shared 2048-row graph prefix matched the earlier GraphEP1 result exactly. Native DP predictions likewise matched the native estimator-batch-4 reference. Graph versus native was **not** bitwise identical: maximum probability difference 0.006434143, mean absolute difference 0.00025358, and six argmax changes among 8192 rows. The net accuracy difference was one fewer correct graph prediction: native accuracy 0.7532958984375 versus graph 0.753173828125; native log loss 0.5902259349822998 versus graph 0.5902121663093567. The predefined BF16 numerical gate passed. These differences are between execution/batching/cache variants, not a quality change caused by increasing GraphDP worker count.

Each graph worker captured exactly four graphs during warmup; capture counts and event lists remained unchanged through all timed passes. Capture took 0.439–0.478 seconds per worker and overlapped across workers. Spawn/import/checkpoint/fit took 4.78–5.28 seconds for graph DP versus 4.98–5.47 seconds for native DP on already staged, warm hosts. Cold graph capture therefore remains an explicit startup cost rather than hidden warm inference work.

| Per-worker steady-state prediction memory | Native estimator batch 4 | Graph resident member execution |
| --- | ---: | ---: |
| Peak allocated GPU memory | 1438.18 MiB | 1316.65 MiB |
| Peak reserved GPU memory | 3168 MiB | 1802 MiB |
| Lifetime host peak RSS | 2114–2119 MiB | 1714–1719 MiB |

The graph configuration also changes cache residency and executes members individually instead of native estimator batching. Its improvement must not be attributed solely to fewer kernel launches from this pair alone; earlier matched resident eager/graph measurements provide that isolation. Every graph-DP worker still duplicates the full fitted resident ensemble, so this small-context memory result does not establish improved large-context capacity. Raw outputs, every repeat, independent audit sidecars and CUDA-test XML are retained under `.kumo-multigpu-20261008/results/graph-dp/`.

The table resets GPU peak counters after warmup. Across startup plus warmup, allocated high-water was 2268.21 MiB for native and 1811.49 MiB for graph; reserved high-water was 3168 and 1950 MiB respectively. These full-phase values must accompany steady-state peaks when reasoning about whether a deployment fits in memory.

### Larger-query-batch saturation control

A final matched control kept the same source, host, TRAIN context, ensemble, precision and CPU-IPC/final-gather contract while using query batch 1024 and the same full 65536-row validation cohort at each GPU count. This prevents treating the batch-256 result as a globally tuned throughput ceiling.

| GPUs | Native estimator-batch-4 rows/s | Graph process DP rows/s | Graph/native speedup |
| --- | ---: | ---: | ---: |
| 1 | 7520.65 | 11207.14 | 1.490× |
| 4 | 29545.50 | 41649.49 | 1.410× |

GraphDP4 scaled 3.716× from GraphDP1 (92.91% efficiency); native scaled 3.929×. All three repeats at four GPUs were bitwise equal to their own one-GPU method reference. Native CPU-IPC predictions additionally matched the same-batch Ohio-local preloaded-pipe reference exactly, confirming transport equivalence without conflating those timing contracts. Graph versus native had maximum probability difference 0.010723293 and mean absolute difference 0.000251088; the combined BF16 absolute/relative gate passed. Fifty argmax labels changed, with a net four fewer correct graph predictions among 65536 rows. Native accuracy/log loss were 0.7591094970703125/0.5810317993164062; graph values were 0.7590484619140625/0.5810284614562988. The independent paired log-loss confidence interval included zero. Different cohort sizes prohibit comparing these aggregate quality metrics directly against the earlier 8192-row cohort as a GPU-induced change.

Graph workers again captured exactly four graphs each before timing and never recaptured during prediction. Steady-state allocated peaks were 1321.52 MiB graph versus 2234.49 MiB native. However, **reserved** peaks were higher for graph: 3652 versus 3168 MiB. Startup-plus-warmup allocated high-water remained 1811.49 versus 2268.21 MiB. Thus this result supports faster warm inference with lower active allocation on this workload, not an unconditional reduction in memory reservation or a universal capacity advantage. All four new runs and their all-repeat audit sidecars use the `resume-graphdp-b1024-*` prefix under the same local raw-results directory.

## Memory and timing expectations

Let `W` denote one model's weights, `C(E)` the total fitted cache for E members, and `A(B)` the active workspace for a fixed query batch B. These are conceptual live-data terms, not allocator peak predictions.

| Layout | Node weight storage | Node fitted member cache | Typical per-GPU cache | Unit accelerated |
| --- | ---: | ---: | ---: | --- |
| 1 GPU | W | C(E) | C(E) | baseline |
| 4 DP | 4W | 4C(E) | C(E) | 4 independent batches |
| 4 EP | 4W | C(E) | about C(E)/4 | 1 batch over E members |
| 2 DP × 2 EP | 4W | 2C(E) | about C(E)/2 | 2 batches, each split by members |

Native SDM `ICLModel.fit` currently copies member caches to pinned CPU memory when fitting more than one context/member on CUDA. Consequently native DP also duplicates pinned host caches and causes repeated host-to-device cache prefetch on prediction. A faster resident EP result could reflect avoided offload as well as parallel execution. Reports therefore need both the native one-GPU baseline and a one-GPU resident executor with the same cache policy as the parallel arm. Explicitly report host RAM, pinned cache bytes, allocated/reserved VRAM and peak device memory; reporting only VRAM can hide native DP's duplication.

Fit is duplicated in the native DP adapter and currently runs sequentially to keep the cold path simple and deterministic. The measured fit time must include all replica fits. A shared recipe fitted once plus device-local context execution, or parallel fits with isolated generators, could reduce this cost later. Copying a fitted `ICLModel` blindly between devices is unsafe because recipe state, caches and streams have independent placement/lifetime requirements.

| Scheduler | Benefit | Cost or limitation |
| --- | --- | --- |
| Threads | Shared CPU prepared tables, low submission overhead, no pickle/IPC; CUDA kernels release Python execution while running | Python recipe/graph work can contend for the GIL; process-global extensions may not be thread-safe |
| Spawned processes | Independent interpreters, predictable model/device ownership, avoids Python GIL contention across workers | Interpreter/import/checkpoint/fit startup; repeated host data; pickling and IPC of relational metadata; output transfer; potentially duplicated preprocessing |
| Distributed preloaded processes | Load common materialized files locally once and send batch indices; avoids per-request graph IPC | Needs a stronger dataset/index lifecycle than this minimal generic prototype; existing torchrun example is the starting point |

The parent wall time around the entire `predict(batches)` call is the throughput measurement. `QueryResult.seconds` measures worker service from input transfer through CPU output, excluding queue wait and, for processes, IPC. Summing service times is not a wall-clock measurement. Batch p50/p95 needs queue-inclusive timestamps for a service workload; per-worker execution latency should be labeled separately. A one-batch call is the latency baseline, while many pre-fixed batches establish throughput scaling. Sample/materialize once for kernel comparisons, then separately measure complete sampler + preprocessing + inference runs to expose CPU bottlenecks.

All arms must return outputs to CPU inside the same timing boundary for the principal comparison. Omitting output copies only for EP/native gives them a different contract, especially for 999-quantile regression. Report cold load + fit and warm prediction independently, and include warmup, multiple repeats, device synchronization, per-GPU peaks and exact observation order.

## Tests and current evidence

Run `PYTHONPATH=. python -m pytest research/multigpu/test_query_parallel.py` from the checkout. The tests cover irregular batch sizes and nonmonotonic observation IDs, 1/2/4/8 CPU worker counts (including idle workers), whole related-table delivery, mixed numerical/categorical inputs, concurrent callers on one replica, accidental shared model rejection, empty input, spawned-process serialization, and real small-architecture KumoTabular fit/predict parity against the identical sequential recipe/member plan. Random weights are used for this CPU model test; it tests execution semantics and does not measure pretrained prediction quality.

The CPU contract tests passed with real KumoTabular output parity at zero absolute/relative tolerance. Integrated DP/EP regression tests passed 22 cases with two CUDA-only cases skipped locally. The actual process GPU experiments below were subsequently executed; CPU tests alone do not establish those performance claims.

## Measured process DP on four L40S GPUs

All measurements used the same `g6e.12xlarge` Spot host with four L40S GPUs, PyTorch 2.9.1+cu130, source `1686803e4`, BF16 autocast and seed 1729. Each configuration received one warmup and three complete measured passes. The tabular model was pretrained KumoTabular large; relational was pretrained KumoRelational with exact prepared `[16,16]` temporal-last neighborhoods. Checkpoint, TRAIN context, batch boundaries and member recipe were fixed across GPU counts.

The original process timing includes parent submission, input IPC, host-to-device transfers, prediction, device-to-host copies and returned CPU batch outputs. It excludes the final parent `TableTensor` concatenation. This small but real boundary difference must remain visible when comparing against native runners that include final concatenation. Later runner revision `c8df1be76` additionally records `gathered_output_repeats_s` and `gathered_rows_per_s`; it leaves the original worker-return measurement unchanged. Do not retroactively rename the original timings as fully gathered output.

| Workload | Estimator batching | Query rows / batch | 1 GPU rows/s | 2 GPUs rows/s | 4 GPUs rows/s | 4/1 speedup | Four-GPU efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Covertype, C1024, E4 | 1 | 2048 / 256 | 1780.50 | 3418.27 | 6650.11 | 3.735× | 93.4% |
| Covertype, C1024, E4 | 4 | 2048 / 256 | 3665.05 | — | 13990.88 | 3.817× | 95.4% |
| Rel-HM user-churn, C1024, E4 | 1 | 2000 / 250 | 1186.19 | 2216.50 | 3859.33 | 3.254× | 81.3% |

The table reports median throughput across three passes, not the best repeat. Covertype workers used one intra-op CPU thread each; the relational workers used eight each, matching the configured per-process value of the earlier native/threaded run. Consequently four relational processes permit up to 32 intra-op CPU threads, whereas the threaded executor shares one process configured with eight. This is a viable implementation comparison, not a controlled attribution of all improvement to the GIL.

A paired timing-boundary control using source `e9114b664` repeated tuned Covertype C1024/E4, estimator batch 4, Q2048/B256 on one and four GPUs. Fully ordered and concatenated CPU output achieved **3659.40 versus 13822.14 rows/s**, a **3.777× speedup and 94.43% efficiency**. The extra parent order validation and concatenation took 0.324–0.666 ms across the six passes. The same runs retain worker-return medians of 3661.89 and 13869.80 rows/s. This paired control closes the output-boundary caveat for the new comparison; it does not rewrite the earlier measurements.

Independent comparison of saved output arrays, ordered IDs, class columns and hashes passed. Covertype process DP with estimator batch size 1 exactly matched the native size-1 reference at every GPU count: accuracy 0.76416015625, log loss 0.5760445594787598. The tuned size-4 process variants exactly matched the tuned native size-4 reference: accuracy 0.76416015625, log loss 0.5759778022766113. The small difference between estimator batching settings is BF16 numerical behavior; use the matching batching reference rather than treating it as a GPU-placement quality change.

Rel-HM process DP exactly matched the native predictions at all three GPU counts: accuracy 0.808, log loss 0.46684929728507996, AUROC 0.6619477128615511 with explicit positive class 1. The original prediction column order was `["1", "0"]`; scoring must follow semantic class labels rather than assuming column 1 is the positive class.

The earlier threaded relational executor achieved about 1350 rows/s on one GPU and 1212 on four. Its typical batch service time grew from about 0.185 seconds to 0.80–0.84 seconds when four threads ran concurrently. Process isolation improved four-GPU throughput to 3859 rows/s. This is consistent with shared Python/runtime contention in the threaded path; CPU-thread allocation and separate CUDA runtimes are also changed, so a GIL-only causal claim would be too strong. Input IPC makes the process one-GPU reference slower than its threaded counterpart, while parallelism more than compensates at four GPUs.

| Workload | Peak allocated VRAM per GPU during prediction | Child host lifetime peak per process | Spawn + import + checkpoint + fit |
| --- | ---: | ---: | ---: |
| Covertype estimator batch 1 | 1314.6 MiB | about 2120 MiB | 5.09–5.52 s |
| Covertype estimator batch 4 | 1438.2 MiB | about 2115 MiB | 5.01–5.47 s |
| Rel-HM estimator batch 1 | 457–470 MiB | about 2390–2560 MiB | 5.15–5.66 s |

Startup used already staged checkpoints and warmed filesystem/compiler caches and parallel replica construction; it is not comparable to the first cold machine invocation. Host-memory figures are per-child lifetime maxima. Their sum is neither a simultaneous node peak nor unique resident memory because shared library pages may be counted multiple times. Every DP replica retains its own complete fitted ensemble/cache, as expected.

### Weak scaling

The weak-scaling control kept 4096 queries per GPU, Covertype C1024/E4, query batch 256 and estimator batch 4. Total query rows therefore differed across configurations; quality differences between those full cohorts are not GPU-induced quality deltas.

| GPUs | Total queries | Aggregate rows/s | Rows/s per GPU | Per-GPU throughput retained |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4096 | 3782.40 | 3782.40 | 100% |
| 2 | 8192 | 7344.95 | 3672.47 | 97.09% |
| 4 | 16384 | 14682.72 | 3670.68 | 97.05% |

Independent audits verified exact predictions and identities over the shared 4096-row prefix and the common 2048-row tuned-reference prefix. A subsequent one-GPU full-16384-row reference using source `e9114b664` produced byte-identical predictions to the four-GPU weak-scaling run across **all 16384 observations**. Both have accuracy 0.75860595703125 and log loss 0.5799540281295776. Its raw outputs are retained separately as `process-tabular-fullcohort-e4-ebs4-c1024-q16384-b256-g1-t1`; the original prefix-only audit records are preserved rather than retroactively expanded. The updated runner also records final concatenation, so compare quality directly but choose matching timing fields for any additional strong-scaling calculation.

Raw results and independent audit sidecars are under `.kumo-multigpu-20261008/results/process-dp/` outside the repository. Each directory retains `result.json`, `predictions.npy`, query IDs where applicable, targets where applicable, and sampled `nvidia-smi` telemetry. Naming records model, context, query count, estimator batching, GPU count and CPU threads.

## Multi-host protocol

`ssh_query_worker.py` runs one persistent process per physical GPU. `multihost_query_bench.py` starts local or SSH peers, waits for every model's READY message, verifies identical context hashes and exact query hashes/IDs, then dispatches whole fixed batches. The global timer uses only the coordinator's monotonic clock and includes command dispatch, worker computation, result encoding/network transport, decoding and validated ordered gathering. Worker clocks are never compared or summed to estimate global throughput. Input batches are staged before timing, which is explicitly a preloaded-query inference measurement rather than network ingress throughput.

The cluster JSON supplies shared `source`, `python`, `data`, `hf_cache`, task-only SSH `identity`, and `workers` entries with `name`, optional remote `host`, and physical CUDA `device`. `--worker-indices 0 1 2 3` or `4 5 6 7` selects each four-GPU group; `--workers 8` selects the combined node set. Each worker records observed GPU UUID/model/memory, host CPU count, framework version, fitted-context hashes and process memory; duplicate physical UUIDs are rejected. Optional `--weights` allocates intact batches using measured worker capacities, minimizing assigned rows divided by capacity with deterministic ties. CPU protocol/shard tests passed 27 cases, including empty worker assignments, irregular batches, capacity weighting and corrupted row/class/output schemas.

An attempted homogeneous eight-L4 experiment could not obtain the four additional singleton Spot workers across the searched east-region capacity pools. No eight-L4 performance claim is supported. The planned alternative combines the existing four L4s and four L40S GPUs across regions. This must be labeled heterogeneous multi-host execution. Compare each four-GPU group with the combined run on identical query rows, and treat the sum of separately measured group throughputs only as an optimistic capacity reference, not as an observed eight-GPU throughput or a homogeneous scaling denominator.

**Execution status:** the mixed-host GPU protocol has not been launched or measured. Source `0361a707f` and `cluster-mixed8-0361a707f.json` were staged on the intended L4 coordinator while other agents held the GPU leases. Before those leases cleared, the execution environment changed to restricted networking with no approval path, and the operator reported SSH `EPERM`. No remote restart or network-policy workaround was attempted. There is therefore no mixed-eight-GPU throughput, latency, memory or prediction-quality result. The 27 CPU protocol tests establish transport/sharding contracts only. Resume with a one-local-GPU smoke test, then matched four-L4 and four-L40S group runs and eight-worker equal/weighted assignments on identical Q65536/B1024/C1024/E4 workloads once ordinary access and explicit leases are restored.

## Approaches intentionally rejected or deferred

- Wrapping a fitted model in training DDP: there are no gradients to synchronize, and it does not solve input/neighborhood partitioning or fitted-state placement.
- Splitting every related table by the task row slice: relational tables have different lengths, duplicate neighborhood copies and graph keys; positional slicing changes the modeled graph.
- Sharing one mutable fitted model between inference threads: native prediction owns transfer streams and recipe execution state. Each worker receives a distinct replica and a serial queue.
- Seeding by rank: that evaluates different ensembles and confounds prediction quality with placement.
- Refitting a separate context per query shard: that changes the estimator and can silently change class/category mappings and temporal scope.
- Forking after CUDA initialization: the process prototype uses spawn and constructs the model in its child.
- Claiming single-query latency improvement from DP: the prototype schedules entire existing batches. Additional GPUs accelerate concurrent batches only.
- Treating more CPU workers as measured GPU scaling: CPU worker tests establish ordering and state isolation only.

## Proposed SDM integration

Keep scheduling outside the model families. A future public `QueryParallel` could accept explicit independently fitted predictors and indivisible structured batches, preserving the same API for tabular and relational models. First establish multi-GPU parity and performance across native and resident cache baselines; then move the small generic executor into `sdm.models` or a narrowly scoped runtime module. Device-aware fitted recipe export/import would enable fit-once replication without copying streams and eliminate duplicated recipe fitting. A persistent distributed example can remain the process deployment path until evidence justifies core runtime complexity.
