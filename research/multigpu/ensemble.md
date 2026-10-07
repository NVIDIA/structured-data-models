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

Initial local tests: eleven CPU tests pass. CPU tests cover uneven member assignment across 1/2/4 replicas, independent stochastic cached state, member order, regression inversion, repeated prediction, failed-refit invalidation/recovery, and duplicate replica rejection. Real KumoTabular modules with reduced dimensions exercise 7-class native-API parity, 12-class ECOC, shuffled class vocabularies, identical per-member codebooks, and exact predictions across 1/2/4 replicas and repeated fits. Both two-device CUDA tests subsequently passed on L40S GPUs, covering a non-default caller stream, inherited autocast context, non-default replica initialization, changing caller streams between fit and prediction, and CPU output from CUDA replicas. Measured pretrained-model findings appear below and in the central results report.

The earlier TabFM branch supplied the independent-member scheduling idea. This implementation uses current recipe/member APIs and avoids importing that branch's large state-validation framework. Threads avoid model/data serialization and exploit PyTorch CUDA operations releasing the GIL; if CPU preprocessing or graph work limits scaling, process-based execution is a separate measurable alternative.

## Combining GPU distribution with member batching

`research.multigpu.batched_ensemble.BatchedEnsembleParallel` is a separate research adapter using current SDM `_batch_slices`, `_stack_context`, and `_forward_batch`. Its `fit(..., estimator_batch_size=2)` groups compatible preprocessed members and distributes those groups across replicas. The batch size is capped at the members-per-replica ceiling so that `None` does not put the entire ensemble on one GPU. E8 with four GPUs and batch size two therefore runs one two-member batch per GPU. Fit caches remain grouped; prediction replays the same grouping and restores original member ordering before target inversion and output reduction.

KumoTabular classification with at most ten classes and regression have deterministic model execution, making this grouping safe apart from ordinary batch-kernel floating-point differences. ECOC above ten classes requires separate per-member seeded generators, which a single batched generator cannot reproduce; this adapter falls back to single-member execution there. KumoRelational also falls back because related-table members cannot be stacked by the current native helpers. A general batched stochastic interface would need explicit per-member RNG inputs or recorded codebooks, not a silently changed sampling sequence.

Eighteen CPU parity tests cover regression, seven-class classification, twelve-class ECOC fallback, 1/2/4 replicas, and batch sizes two/maximum. Predictions pass `rtol=atol=1e-5`, with identical classification argmax. GPU results below demonstrate a two-device throughput improvement on the larger workload, but also a four-device numerical failure when local member batch size changes.

The first GPU test attempt exposed use of a nonexistent `TableTensor._tensors()` accessor in stream bookkeeping. Replacing it with the supported `TableTensor.record_stream()` fixed both CUDA tests; both then passed on the GPU host. This implementation failure is separate from measured model prediction quality.

Main-thread `torch.profiler` does not collect worker CPU ranges in the tested PyTorch 2.9.1 configuration: a local probe recorded only preprocessing and finalization. Named worker ranges exist for supported profilers, but complete GPU/device coverage must be checked. Synchronized wall timing and utilization measurements remain valid; use Nsight Systems for complete threaded CUDA tracing when available.

## Persistent process comparison

`research.multigpu.process_ensemble:factory` creates one persistent spawned Python process per GPU. Parent preprocessing remains shared and ordered; each process owns model weights and caches for its assigned logical members. Every member retains the same `member_seed + member_id` as the thread executor. It transports transformed tables and predictions through CPU shared memory, including GPU-to-host and host-to-GPU copies in end-to-end timing. This avoids CUDA IPC ownership assumptions at the cost of measurable transport overhead. Constructor timing includes moving supplied replicas to CPU, process startup, and reloading onto child GPUs.

The process arm isolates Python interpreter contention but is not automatically faster. Compare process EP1 against thread EP1 to quantify fixed IPC costs before interpreting process EP2/4 scaling. `model.memory(reset_peak=True)` returns each child's PID, device, allocator counters, and unique retained cache storage. Main-process allocator counters alone exclude child GPU allocations. Two local CPU process tests pass exact stochastic prediction parity and repeated-fit/clear behavior. One/two-device CUDA process tests passed on source `4e1c5c33d` in 5.30 seconds. The first CUDA test attempt failed before worker initialization because the fixture's `test.models` import collided with the Python standard-library `test` package; moving the fixture to the importable `research.multigpu._test_models` namespace fixed that test-harness failure.

## Cache compaction hypothesis and experiment

The first real E4 resident run reported 512,754,868 bytes of unique GPU cache storage versus 358,614,196 logical tensor bytes. Native CPU-offloaded caches used the latter size. Structural inspection on reduced KumoTabular modules identified row-encoder column-attention value tensors: a value view retains the full fused key/value projection allocation after normalized keys move into a new allocation. For example, `row_embedding.col_block0.value` had shape `(6,8,4,4)`, stride `(256,32,4,1)`, offset 16, 3,072 logical bytes, and 6,144 backing bytes. This is not the ICL grouped-query head slice; that path already calls `.contiguous()`.

`research.multigpu.compact_ensemble:factory` is an optional resident-cache arm. It clones only view groups whose backing storage is larger than the sum of uniquely referenced view bytes, preserves repeated aliases, and leaves fully referenced shared storage untouched. Every clone occurs after fit on the member's owning stream. `compaction_s` is synchronization-aware added fit time; `compaction_storage_bytes` records before/after unique bytes per member. End-to-end fit and memory peaks must include this phase. CPU tests verify exact tensor values, retained aliasing, storage savings, and avoidance of unnecessary copies.

On the pretrained large E4/C1024 workload, compaction reduced unique resident cache storage from 512,754,868 to 358,614,196 bytes (30.1%) with identical prediction hashes. Added compaction wall time was 3.23 ms. Prediction peak allocation fell from approximately 1.716 to 1.561 GB, but fit peak remained 1.900 GB and allocator reservation remained 2.047 GB. This is a retained-cache/prediction-memory optimization, not a demonstrated fit-capacity increase. Median throughput was 1,817 rows/s versus the earlier uncompact resident 1,730; those separately run measurements do not establish a robust 5% latency improvement. A separate C16384 workload already had equal logical and physical cache sizes, so compaction correctly did nothing.

## Autocast lifetime and graph replay

The [initial Nsight analysis](ensemble-profile.md) found 7,840 extra FP32-to-BF16 conversion kernels in four-GPU resident inference. `research.multigpu.persistent_autocast:factory` holds one outer autocast scope on each worker across fit/predict calls. The ordinary executor's per-call context is nested inside that scope, so leaving a call does not end the worker's outer scope. This tests whether repeated cast-cache invalidation explains the extra kernels. It deliberately fixes precision for the executor lifetime and rejects a mismatched caller autocast setting; replica weights remain immutable. Close exits each context on its owning thread. This is a measurement hypothesis, not a proven speedup or default API change.

`research.multigpu.graph_ensemble:factory` captures neural replay separately for each fitted member, query shape, and precision. Recipe processing stays eager on the parent. Cached target classes are copied to CPU solely for output-column metadata, avoiding `.tolist()` on a CUDA tensor inside capture. Static numerical query buffers are copied before replay, and returned outputs are cloned so later replays cannot overwrite earlier predictions. Different remainder shapes create different graphs; refit clears all captures.

Graph capture currently supports only CUDA KumoTabular members with fully numerical transformed queries. Unsupported input forms fail explicitly. Capture records warmup/capture durations and shapes in `capture_events`; `graph_capture_s` sums worker durations and can exceed wall time because captures overlap. The first prediction shape incurs cold setup and additional graph memory. Warm every measured shape before claiming steady-state latency, and record `graph_count` before and after the timed interval to detect accidental cold captures. Internal autocast caching is disabled during graph capture so weight-conversion operations and their storage belong to the graph instead of depending on temporary cached casts from a finished autocast scope. One/two-device CUDA tests on actual reduced KumoTabular modules passed on source `4e1c5c33d` in 2.52 seconds, including changed query values, a distinct remainder shape, output lifetime, and refit invalidation. Pretrained scaling remains a separate required measurement. Since graph replay also moves class-label metadata to CPU, any speedup combines capture effects with removal of that metadata's per-call device synchronization.

Persistent autocast did not materially improve the small workload: median throughput was 1,753 rows/s on one GPU and 1,660 on four versus earlier ordinary-thread values of 1,730 and 1,645. Prediction hashes were identical. It retained approximately 267 MB of additional cast weights per GPU after prediction. These approximately 1% timing differences are not evidence of a useful optimization; paired traces must determine whether conversions were actually removed before interpreting the rejected performance hypothesis.

## Findings available so far

The small E4/C1024/query-batch256 workload did not scale through naive threaded EP: approximately 1,730 rows/s on one resident replica, 1,883 on two, and 1,645 on four. Predictions were byte-identical. Native estimator batching four reached approximately 4,007 rows/s, so comparing only against unbatched native would materially understate the strongest baseline. Trace evidence shows short kernels and little multi-GPU compute overlap; interpreter isolation, reduced launch counts, and precision lifetime are the next experiments.

Larger E8/C4096/query-batch1024 batched EP produced about 4,014 rows/s on one resident GPU and 7,368 on two, with exact predictions in that initial comparison. Four GPUs reached about 5,981 rows/s but changed effective local member batch size and failed the predefined numerical tolerance against the batch-eight reference. That four-GPU throughput is not a valid same-semantics scaling claim. Matched local batching controls, repeated measurements, memory accounting, and the complete quality results belong in the central results report before integration recommendations are finalized.
