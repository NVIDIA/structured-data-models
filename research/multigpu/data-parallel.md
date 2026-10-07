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

The CPU contract tests passed with real KumoTabular output parity at zero absolute/relative tolerance. CUDA performance, native KumoRelational parity, real-task quality and hybrid scaling are owned by the GPU runners and must be filled from their saved measurements. No GPU speedup is claimed solely from this implementation or the CPU tests.

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
