# Ensemble parallel inference

The prototype `sdm.models.EnsembleParallel` works with the current `ICLModel` member execution contract, including KumoTabular and KumoRelational. It replaces neither model architecture nor preprocessing recipes. Callers create identical evaluation replicas on chosen devices; the executor shares one fitted recipe and assigns member `i` to replica `i % GPU_count`. Each replica retains only its members' caches on its device.

```python
import torch
from sdm.models import EnsembleParallel, KumoTabular

replicas = [KumoTabular(task="classification", device=f"cuda:{i}")
            for i in range(2)]
with EnsembleParallel(replicas) as model:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.fit(x_context, y_context, num_estimators=8,
                  generator=torch.Generator(device=x_context.device).manual_seed(1729),
                  member_seed=1729)
        prediction = model.predict(x_query)
    print(model.cache_bytes)
```

For KumoRelational, pass `related_tables` to both fit and predict and forward `num_hops=2` to fit. Sample query neighborhoods once upstream and supply the same ordered samples to every comparison arm. The executor moves transformed related tables together with the task rows.

## Semantics and fairness

- Recipe fitting and query transformation execute once on the input device. All devices therefore receive the same member preprocessing plan.
- Model randomness uses an explicit device-local generator seeded `member_seed + member_id`. Seeds remain stable under 1/2/4-device placement, including relational random edge embeddings and tabular ECOC codebooks. This is a distinct RNG policy from public `ICLModel.fit`, which consumes one generator sequentially. Use one replica of this executor as the exact stochastic serial reference.
- Outputs return to the input device in logical member order. Regression target transforms are inverted separately for each member before the recipe output reduction. Classification output columns are handled by the existing recipe/container semantics.
- Current public `ICLModel.fit` offloads multiple member caches to pinned CPU memory and `predict` prefetches them. This prototype retains all caches on their owning GPU. Report public serial, serial-resident executor, and 2/4-GPU executor separately; otherwise offload removal is confounded with GPU scaling.
- Public `estimator_batch_size` can batch compatible tabular members. Measure its best feasible setting as another single-GPU baseline. This first prototype executes members separately; related-table members already execute separately in the public implementation.

## Execution and boundaries

One long-lived host thread and dedicated CUDA stream serve each replica. Thread-local inference mode and autocast state are explicitly propagated. Workers wait on the caller stream's ready event; fit/predict synchronously join all workers and synchronize their streams before returning. Failure also drains all submitted workers, and failed refits leave the executor unfitted. This boundary deliberately favors correctness and comparable timing over asynchronous serving integration.

The caller must preserve replica weights and devices and avoid overlapping calls to the same executor. No training, callbacks, model-internal sharding, CPU cache offload, dynamic load balancing, multiprocessing serialization, or automatic checkpoint replication is included. `close()` joins the workers; use the context manager. Input preprocessing can remain a bottleneck and related-table transfer can repeat for members sharing tables. All preprocessed contexts initially remain on the input device, so GPU0 preprocessing peak can limit context scaling even when distributed caches would fit. Parameters are replicated, so aggregate parameter memory grows with GPU count. Single-member latency cannot improve through ensemble distribution.

## Validation status

Initial local tests: eleven CPU tests pass, two two-CUDA-device tests skipped on the development host. CPU tests cover uneven member assignment across 1/2/4 replicas, independent stochastic cached state, member order, regression inversion, repeated prediction, failed-refit invalidation/recovery, and duplicate replica rejection. Real KumoTabular modules with reduced dimensions exercise 7-class native-API parity, 12-class ECOC, shuffled class vocabularies, identical per-member codebooks, and exact predictions across 1/2/4 replicas and repeated fits. CUDA tests cover a non-default caller stream, inherited autocast context, non-default replica initialization, changing caller streams between fit and prediction, and CPU output from CUDA replicas. GPU quality and scaling measurements are pending and must be attached before performance claims.

The earlier TabFM branch supplied the independent-member scheduling idea. This implementation uses current recipe/member APIs and avoids importing that branch's large state-validation framework. Threads avoid model/data serialization and exploit PyTorch CUDA operations releasing the GIL; if CPU preprocessing or graph work limits scaling, process-based execution is a separate measurable alternative.

## Combining GPU distribution with member batching

`research.multigpu.batched_ensemble.BatchedEnsembleParallel` is a separate research adapter using current SDM `_batch_slices`, `_stack_context`, and `_forward_batch`. Its `fit(..., estimator_batch_size=2)` groups compatible preprocessed members and distributes those groups across replicas. The batch size is capped at the members-per-replica ceiling so that `None` does not put the entire ensemble on one GPU. E8 with four GPUs and batch size two therefore runs one two-member batch per GPU. Fit caches remain grouped; prediction replays the same grouping and restores original member ordering before target inversion and output reduction.

KumoTabular classification with at most ten classes and regression have deterministic model execution, making this grouping safe apart from ordinary batch-kernel floating-point differences. ECOC above ten classes requires separate per-member seeded generators, which a single batched generator cannot reproduce; this adapter falls back to single-member execution there. KumoRelational also falls back because related-table members cannot be stacked by the current native helpers. A general batched stochastic interface would need explicit per-member RNG inputs or recorded codebooks, not a silently changed sampling sequence.

Eighteen CPU parity tests cover regression, seven-class classification, twelve-class ECOC fallback, 1/2/4 replicas, and batch sizes two/maximum. Predictions pass `rtol=atol=1e-5`, with identical classification argmax. This adapter is awaiting GPU throughput, memory, and quality measurements; it has no demonstrated performance benefit yet.

The first GPU test attempt exposed use of a nonexistent `TableTensor._tensors()` accessor in stream bookkeeping. Replacing it with the supported `TableTensor.record_stream()` fixed both CUDA tests; both then passed on the GPU host. This implementation failure is separate from measured model prediction quality.

Main-thread `torch.profiler` does not collect worker CPU ranges in the tested PyTorch 2.9.1 configuration: a local probe recorded only preprocessing and finalization. Named worker ranges exist for supported profilers, but complete GPU/device coverage must be checked. Synchronized wall timing and utilization measurements remain valid; use Nsight Systems for complete threaded CUDA tracing when available.
