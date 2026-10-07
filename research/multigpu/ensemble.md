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

The caller must preserve replica weights and devices and avoid overlapping calls to the same executor. No training, callbacks, model-internal sharding, CPU cache offload, dynamic load balancing, multiprocessing serialization, or automatic checkpoint replication is included. `close()` joins the workers; use the context manager. Input preprocessing can remain a bottleneck and related-table transfer can repeat for members sharing tables. Parameters are replicated, so aggregate parameter memory grows with GPU count. Single-member latency cannot improve through ensemble distribution.

## Validation status

Initial local tests: five CPU tests pass, one two-CUDA-device test skipped on the development host. CPU tests cover uneven member assignment across 1/2/4 replicas, independent stochastic cached state, member order, regression inversion, repeated prediction, failed-refit invalidation/recovery, and duplicate replica rejection. CUDA test covers a non-default caller stream and inherited autocast context. GPU quality and scaling measurements are pending and must be attached before performance claims.

The earlier TabFM branch supplied the independent-member scheduling idea. This implementation uses current recipe/member APIs and avoids importing that branch's large state-validation framework. Threads avoid model/data serialization and exploit PyTorch CUDA operations releasing the GIL; if CPU preprocessing or graph work limits scaling, process-based execution is a separate measurable alternative.
