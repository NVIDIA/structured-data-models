# TabFM multi-GPU inference

Status: design investigation, 2026-08-26

## Recommendation

Implement estimator/ensemble parallelism first as a generic execution layer around `ICLModel`. It matches SDM's existing ensemble representation, requires no change to TabFM's numerical operations, partitions rather than replicates estimator caches, and follows the approach used by TabPFN's current multi-device inference engine.

Treat data parallelism as two separate cases:

- Independent tasks or requests should use one ordinary `ICLModel` per process/device. This needs documentation and examples more than a new core abstraction.
- Query-row parallelism for one fitted task should come later and only for models that explicitly declare query rows independent. It requires replicated weights and caches and therefore improves throughput but not model or context capacity.

Keep context parallelism experimental until the first two paths are benchmarked. It is the only listed approach that can make a single long context fit when its row-dependent KV cache exceeds one device, but it changes every distributed attention site, requires collectives in the model core, and does not shard all of TabFM's cache. PyTorch's context-parallel API is still documented as unstable and is newer than SDM's `torch>=2.7` floor.

The approaches are composable in the long term. A two-dimensional device mesh could partition estimators on one dimension and context or query rows on the other, but that should not be the first implementation.

## Decision matrix

| Approach                         | Partitioned unit                | Weights per GPU | Cache placement for one task                                          | Communication                                        | Best use                                         |                   Priority |
| -------------------------------- | ------------------------------- | --------------: | --------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------ | -------------------------: |
| Ensemble parallel                | Estimator IDs                   |    Full replica | Partitioned by estimator                                              | Gather small member outputs for postprocessing       | `num_estimators >= num_devices`                  |                          1 |
| Data parallel, independent tasks | Whole tasks or requests         |    Full replica | Partitioned by task                                                   | None in the model path                               | Throughput across datasets/users                 | 1, primarily documentation |
| Data parallel, one fitted task   | Query rows                      |    Full replica | Replicated on every participating GPU                                 | Concatenate query outputs                            | One estimator with very large query sets         |                          2 |
| Context parallel                 | Context rows at attention sites |    Full replica | Row-dependent KV sharded; column-dependent cache initially replicated | Ring/all-gather attention at every distributed layer | A single context/cache that does not fit one GPU |            3, experimental |

Tensor or pipeline parallelism is orthogonal. It would be needed if a model replica itself did not fit on one GPU; none of the three approaches above shards the 1.64B model parameters.

## What exists in SDM today

The current code already has most of the logical boundaries needed for ensemble parallelism:

- `ICLModel._forward_call`, `fit`, and `_predict_call` create estimator members through `RecipeExecution`, run members serially, preserve member order, and apply target inversion and output reduction once.
- `RecipeExecution.fit_transform` and `transform` materialize `MemberContext` and `MemberQuery` objects for each estimator.
- `Cache` contains independent estimator caches under integer keys. With multiple CUDA estimators, `ICLModel.fit` currently moves each cache to pinned CPU memory, while `_predict_call` prefetches one cache at a time to the single input device.
- `EnsembleTable` and `ReduceEstimators` already make the estimator dimension explicit and preserve the logical member order across preprocessing and postprocessing.
- TabFM's `_forward` receives an ordinary member context/query and an optional member-local `Cache`; it does not need to know which worker ran it.

The main missing boundary is that preprocessing, member execution, and postprocessing are all owned by one `ICLModel` instance and one device. Multi-device execution should factor those phases without teaching the TabFM wrapper about workers, threads, processes, queues, or device selection.

There is also prior work on the remote `feature/tabiclv2-estimator-batching` branch that groups compatible estimator members for a single model call. Its `ContextGroup`, `QueryGroup`, `_forward_group`, and stable member-ID reconstruction are useful patterns. A multi-device implementation should reuse or supersede those execution boundaries rather than create a second incompatible member scheduler.

## TabFM memory and computation shape

The current SDM architecture has 1,647,783,197 parameters for regression and 1,639,442,458 for classification. Parameter storage alone is approximately:

| Dtype     | Regression | Classification |
| --------- | ---------: | -------------: |
| FP32      |   6.14 GiB |       6.11 GiB |
| BF16/FP16 |   3.07 GiB |       3.05 GiB |

The published safetensors files are approximately 6.59 GB for regression and 6.56 GB for classification, consistent with FP32 storage. Autocast reduces activation compute dtype but does not by itself reduce resident parameter storage.

For one estimator at `C` transformed columns and `R` context rows, the dominant KV cache storage is:

```text
row embedding: 2 × 6 layers × C × 256 inducing points × 256 channels
ICL block:     2 × 24 layers × R × 2,048 channels
```

The factor of two is key plus value. In BF16/FP16 this is approximately:

```text
row embedding cache = 1.5 MiB × C
ICL cache           = 0.1875 MiB × R
```

Examples per estimator:

| Shape             | Row embedding cache | ICL cache | Total before small metadata |
| ----------------- | ------------------: | --------: | --------------------------: |
| `C=100, R=1,000`  |            0.15 GiB |  0.18 GiB |                    0.33 GiB |
| `C=500, R=10,000` |            0.73 GiB |  1.83 GiB |                    2.56 GiB |
| `C=500, R=50,000` |            0.73 GiB |  9.16 GiB |                    9.89 GiB |

FP32 doubles these cache figures. Activations and allocator workspace are additional.

This explains the scheduling priorities:

- Ensemble parallelism distributes independent copies of the full cache, so it improves both latency and aggregate cache capacity when there are multiple estimators.
- Query data parallelism must make each GPU able to answer against the same fitted context, so it duplicates the cache and is unattractive under memory pressure.
- Row context parallelism shards the `R`-dependent ICL cache, but the `C`-dependent row-embedding cache remains replicated unless a later design also shards columns.

## Approach 1: estimator/ensemble parallelism

### Semantics

Estimator members are independent until `RecipeExecution.transform_output`. The scheduler assigns stable member IDs to devices, runs each device's assigned members, gathers outputs onto the input/output device, restores member order, then invokes the existing target inversion and output recipe.

For `E` estimators and `G` GPUs, use a dynamic work queue or a deterministic round-robin initial assignment with one worker per GPU. A dynamic queue handles preprocessing-dependent member costs and `E % G != 0`; result slots keyed by member ID preserve deterministic reduction order.

Each GPU holds one model replica. Each estimator cache belongs to the device that built it when the resident-cache policy is selected. CPU offload remains available for cases where multiple caches assigned to one GPU do not fit.

### Public shape

Do not add `devices=` to `TabFM` or allow a single `torch.nn.Module` to pretend it lives on several devices. Keep ordinary models single-device and compose them with a generic runner. A working API shape is:

```python
models = [
    sdm.models.TabFM(
        task="classification",
        checkpoint_path=checkpoint_path,
        device=device,
    )
    for device in ("cuda:0", "cuda:1")
]

model = sdm.inference.EnsembleParallel(models)
model.fit(x_context, y_context, num_estimators=8)
prediction = model.predict(x_query)
```

The exact name is not important yet. Accepting explicit replicas avoids hidden 1.6B-parameter deep copies and keeps checkpoint loading and dtype decisions visible. A model-factory convenience can be considered after the execution contract is stable.

The wrapper should preserve the normal `forward`, `fit`, `predict`, and `clear` behavior. Output should return to the original query device to preserve SDM's current device contract.

### Internal refactor

Split the current base-model flow into three reusable phases:

1. Prepare: fit or apply one `RecipeExecution`, producing stable `(member_id, MemberContext, MemberQuery)` work items on the caller/coordinator device.
2. Execute: run a single member or compatible member group against a supplied model replica and member-local cache.
3. Finalize: restore logical member order, invert regression targets, and run the output recipe on the coordinator device.

The execution layer should own only PyTorch model execution. `RecipeExecution` remains the authority for ensemble preprocessing state and output reconstruction. This keeps leakage-sensitive preprocessing and stochastic schedules identical whether execution is serial, batched on one GPU, or distributed across GPUs.

The fitted state needs a generic placement record such as `(member_id, worker_id, cache)`. Do not encode a device into `Cache` itself: `Cache` should remain a tensor container, while the scheduler owns placement and offload policy.

### Concurrency

A same-process implementation can use one long-lived Python worker thread per CUDA device. PyTorch releases the GIL while CUDA work runs, and a worker can establish its device and stream once. Return a CUDA event with each result so the coordinator waits on producer work without imposing device-wide synchronization.

This is the mechanism used by current TabPFN: it maintains a model copy per device, dispatches ensemble members through a thread pool, associates each estimator with an explicit KV cache, and gathers results in estimator order. Relevant upstream sources are [`parallel_execute.py`](https://github.com/PriorLabs/TabPFN/blob/fb157711adac138e0eae588e6a4e570ac28f8d25/src/tabpfn/parallel_execute.py) and [`inference.py`](https://github.com/PriorLabs/TabPFN/blob/fb157711adac138e0eae588e6a4e570ac28f8d25/src/tabpfn/inference.py).

SDM should copy the execution idea, not TabPFN's estimator-specific API. In particular, the SDM runner must preserve `TableTensor`, `RecipeExecution`, arbitrary output recipes, classification column alignment, regression inverse transforms, and related-table capability.

### Determinism and callbacks

Run all recipe sampling on the coordinator before dispatch so a fixed generator produces the same member definitions and order as serial execution. If a model core consumes randomness, draw one seed per member centrally and construct a device-local generator in the worker; a single CUDA generator cannot be shared across devices.

Lifecycle callbacks should run on the coordinator. Phase one should reject callbacks that request gradients because cross-device differentiable execution, callback thread safety, and gathered autograd graphs are outside an inference-only feature. Ordinary PyTorch hooks on replicas remain possible.

### Output reduction

Gather all member outputs in phase one and call the existing output recipe once. Although `ReduceEstimators(method="mean")` could be optimized into per-device partial sums and one small reduction, arbitrary processors may legally run before the reducer. A distributed-reducer protocol is an optional later optimization, not a requirement for correctness.

## Approach 2: data parallelism

### Independent tasks and requests

This is embarrassingly parallel and already composes with SDM's single-device API: launch one process per GPU, create one model per local device, and route complete tasks to ranks. There is no model-path collective and no shared mutable cache. A `torchrun` example can demonstrate the pattern without adding fleet, request-routing, or serving abstractions to the library.

`torch.nn.DataParallel` is not a good fit. It scatters the leading tensor dimension, replicates modules during forward, and does not understand `RecipeExecution`, `TableTensor` ensemble members, or the persistent state created by `fit`. `DistributedDataParallel` is designed primarily around gradient synchronization and adds no useful inference scheduling for independent requests.

SDM may eventually expose a small batching utility for multiple independent ICL tasks, but it should remain tensor-centric and must not become a service scheduler.

### Query rows for one fitted task

Given a frozen fitted context, TabFM query rows are independent: row embedding attends each row to context-derived inducing states, row attention operates within a row, and the ICL block attends queries to cached context rather than to other queries. Therefore a query table can be split on dimension `-2`, evaluated in parallel, and concatenated in original row order.

This path has three important constraints:

- Every participating GPU needs a model replica and every estimator cache, so memory grows with the number of GPUs rather than shrinking.
- Query independence is a model property, not a safe assumption for every future `ICLModel`; add an explicit capability before a generic runner shards query rows.
- Splitting `RelatedTables` consistently with task rows is non-trivial. A generic implementation needs a public row-selection operation that preserves task links and relational schemas; TabFM itself does not use related tables.

For one estimator, query parallelism can scale a very large prediction batch. With several estimators, ensemble parallelism should normally consume the devices first because it avoids cache replication. Query sharding becomes useful when `E < G`, when one member dominates latency, or when independent task throughput is unavailable.

The first implementation should build on cached `fit`/`predict`, not one-shot `forward`: build one canonical estimator cache, copy it to each worker device, transform each query chunk with the already-fitted recipe, run it, and concatenate results before output processing. Add automatic chunk sizing only after explicit sizes are benchmarked.

## Approach 3: context parallelism

### Where context rows participate

Context-row sharding crosses two TabFM stages:

1. In each of six induced column-attention layers, learned inducing queries attend over all context rows. The context row tokens can be sharded, but attention must combine all shards before each rank applies the output block to its local rows.
2. In each of 24 ICL layers, local row queries attend over all current context K/V rows. During cached prediction, query rows attend over the sharded per-layer KV cache.

Cell embedding is row-local. Row-wise attention is also row-local because its sequence dimension is columns/readout tokens. The difficult operations are the 30 cross-row attention sites above.

### Distributed attention algorithm

For local query shards and context K/V shards, ring attention is the natural execution shape: keep local queries stationary, rotate K/V shards through the process group, and update the online softmax accumulator for each shard. The output remains sharded like the query. During `fit`, each rank records only its local context K/V slice; during `predict`, those slices stay resident and rotate or contribute partial attention without being permanently all-gathered.

An all-gather-KV prototype is simpler and useful as a numerical baseline, but it does not solve peak context memory at the attention site. A true capacity feature must avoid materializing the full K/V on every rank.

PyTorch provides `torch.distributed.tensor.experimental.context_parallel`, which patches `torch.nn.functional.scaled_dot_product_attention` and uses ring/all-gather rotation strategies. Its [official tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) shows that outputs remain sequence-sharded until explicitly unsharded. The API needs to be prototyped against TabFM's cross-attention shapes: its examples focus on Q/K/V sharded along the same self-attention sequence, while TabFM often has different query and context lengths.

### SDM integration boundary

Context parallelism belongs below TabFM's public wrapper:

- Add a distributed attention backend at the `sdm.nn.SDPA`/`Attention` boundary rather than branching on devices inside `TabFM`.
- Pass a process group or `DeviceMesh` explicitly through an execution context; do not rely on an implicit global group in generic modules.
- Represent sharded cache state explicitly. `KVCacheEntry` currently contains dense local tensors with no global sequence length or shard metadata, and `Cache.to()` assumes ordinary device moves.
- Teach `RowEmbedding` and `ICLBlock` where the row shard is introduced and where an output remains sharded. The outer `TableTensor` API should see a normal gathered prediction only at the final boundary.

Do not land imports from PyTorch private `_attention` modules. SDM currently supports `torch>=2.7`, while the documented context-parallel API is experimental and evolving. Start with an example/prototype pinned to a supported PyTorch version; either wait for a suitable public stable API, raise the minimum version deliberately, or implement a small SDM-owned ring backend after measurements justify the maintenance cost.

### Limits

Context parallelism replicates model weights on every rank. With row-only sharding it also replicates the approximately `1.5 MiB × C` BF16 row-embedding cache. At `C=500`, that is a 0.73 GiB per-estimator floor before the row-dependent ICL cache, activations, and allocator workspace.

It also adds communication to 30 attention layers. Small and medium contexts may slow down because MLP, projection, row-attention, and cell-embedding work remains local or replicated while collectives are added. Context parallelism should be selected for capacity first and only advertised for speed if benchmarks show a crossover.

## Proposed implementation sequence

### Phase 0: measurement and execution boundary

- Add a benchmark that reports synchronized `fit`, first `predict`, warm `predict`, peak allocated/reserved GPU memory, host cache size, output dtype/device, and numerical delta from serial execution.
- Refactor member preparation/execution/finalization in `ICLModel` without changing public behavior.
- Reconcile this refactor with the compatible-estimator grouping work on `feature/tabiclv2-estimator-batching`.
- Add an explicit TabFM parameter/cache size helper or benchmark-side calculation so placement decisions use observed shapes rather than checkpoint-size guesses.

### Phase 1: generic ensemble parallel runner

- Accept explicit same-configuration model replicas, one per worker device.
- Execute stable estimator work items with one long-lived worker per CUDA device.
- Support both resident-device and pinned-CPU estimator cache placement.
- Gather member outputs to the query device and reuse existing target/output processing unchanged.
- Preserve serial behavior for one worker and when `num_estimators=1`.
- Initially support inference-only callbacks and deterministic coordinator-side recipe sampling.

### Phase 2: query/data parallelism

- Publish a one-process-per-GPU example for independent tasks.
- Add a model capability for query-row independence.
- Implement cached query-row sharding for capable non-relational models, with cache replication explicit in the API and metrics.
- Extend to relational models only after row selection of `RelatedTables` has a correct public contract.

### Phase 3: context-parallel prototype

- Prototype distributed cross-attention with two NCCL ranks and compare full outputs and gradients-disabled inference against dense SDPA.
- Apply it first to the 24-layer `ICLBlock`, measure the reduced row-dependent cache, then cover induced column attention.
- Keep row-embedding cache replicated initially and report that memory floor.
- Promote the feature only after multi-GPU CI, long-context benchmarks, uneven-shard handling, and failure cleanup exist.

## Verification plan

Functional cases:

- Classification and regression, one-shot and cached paths.
- `num_estimators` values smaller than, equal to, and not divisible by device count.
- Default recipe plus a custom output recipe that does work before `ReduceEstimators`.
- Stable member order, class-column order, and regression target inversion.
- Resident and CPU-offloaded caches, repeated prediction, `clear`, serialization, and worker exception propagation.
- One-device execution matching the existing code path.
- Autocast with FP32 and BF16 model storage called out separately.

Benchmark matrix:

| Workload         | Estimators |  Columns | Context rows |    Query rows |    GPUs |
| ---------------- | ---------: | -------: | -----------: | ------------: | ------: |
| Ensemble-heavy   |          8 | 100, 500 |      1k, 10k |            1k | 1, 2, 4 |
| Query-heavy      |          1 | 100, 500 |          10k | 1k, 10k, 100k | 1, 2, 4 |
| Context-capacity |          1 |      500 |    10k, 50k+ |            1k | 1, 2, 4 |

Report cold model load separately from fit and warm predict. Use CUDA events or explicit synchronization around timing. Record per-GPU peak memory, aggregate host memory, cache transfer volume, and interconnect type because PCIe and NVLink change the crossover points.

The current GPU CI job exposes one GPU, so it cannot validate real multi-GPU scheduling or NCCL context parallelism. Unit tests can exercise scheduling with logical CPU workers and mock events, but phase 1 needs at least one periodic two-GPU integration job; context parallelism needs a two-rank NCCL job.

## Open decisions

- Whether the first public runner accepts explicit replicas only or also a model factory.
- Whether resident cache placement is opt-in, opt-out, or selected by an explicit memory policy.
- Whether model-core RNG is part of the cross-device reproducibility contract or only recipe/member determinism is guaranteed.
- Whether compatible estimator batching should happen within each device worker in phase 1 or follow as a separate optimization.
- Which PyTorch version first offers a public context-parallel API suitable for TabFM cross-attention without private imports.
- Whether BF16/FP16 checkpoint conversion should be improved before replicating TabFM, since current published checkpoint storage is FP32-sized.

## Sources

- [SDM `ICLModel`](../sdm/models/base.py)
- [SDM `RecipeExecution`](../sdm/processing/execution.py)
- [SDM TabFM model](../sdm/models/tabfm/model.py)
- [SDM TabFM row embedding](../sdm/models/tabfm/row_embedding.py)
- [SDM TabFM ICL block](../sdm/models/tabfm/icl.py)
- [SDM attention implementation](../sdm/nn/attention.py)
- [TabPFN multi-device inference at `fb157711`](https://github.com/PriorLabs/TabPFN/blob/fb157711adac138e0eae588e6a4e570ac28f8d25/src/tabpfn/inference.py)
- [TabPFN parallel executor at `fb157711`](https://github.com/PriorLabs/TabPFN/blob/fb157711adac138e0eae588e6a4e570ac28f8d25/src/tabpfn/parallel_execute.py)
- [PyTorch context parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [Published TabFM checkpoint and architecture card](https://huggingface.co/google/tabfm-1.0.0-pytorch)
