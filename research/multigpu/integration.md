# SDM multi-GPU integration design

Inspection baseline: `842c408fe`, 2026-10-08. Recommendations below are architectural inferences from the source, not measured performance claims.

## Model boundaries that matter

`KumoTabular` performs cell embedding, alternating induced column attention and row attention, then a dataset-level ICL transformer. Its medium and large variants cache only two KV heads for query ICL attention, across 24 ICL layers. A cached-context implementation therefore has less KV memory to save than full multi-head attention would suggest. Query rows are independent after the context-derived state is fitted. During fit, context rows interact through induced column attention and ICL context attention; naive row splitting and averaging is not equivalent.

`KumoRelational` constructs context/query task graphs, derives per-table features and relative time, embeds every related table, runs GNN message passing, and feeds task readouts to the shared TabICLv2 ICL block. Table encoders are separable only after deterministic target ownership and preprocessing are established. The GNN draws and caches random edge-type embeddings; all equivalent executions must use the same values. Query graph preparation and GNN cost are unaffected by ICL-only CP.

The current public fitted path offloads multi-estimator CUDA caches to pinned CPU and overlaps their reload with compute. A device-resident executor must document that policy change. Model parameters, fit recipes, fitted caches, and logical estimator identities have distinct lifecycles; treating a fitted model as an ordinary movable module loses this distinction.

For one ordinary classification/regression member, ICL KV storage is approximately `2 × layers × context_rows × KV_heads × head_width × bytes_per_element`. These are architectural estimates at BF16/FP16 (two bytes), excluding parameters, row-encoder caches, graph state, allocator overhead, hierarchical class branches, and transient fit activations:

| Model | ICL layers / KV heads / head width | KV at 1,024 context | KV at 16,384 context | Ideal four-way retained KV at 16,384 |
|---|---|---:|---:|---:|
| KumoTabular small | 12 / 8 / 64 | 24 MiB | 384 MiB | 96 MiB/rank |
| KumoTabular medium | 24 / 2 / 64 | 12 MiB | 192 MiB | 48 MiB/rank |
| KumoTabular large | 24 / 2 / 64 | 12 MiB | 192 MiB | 48 MiB/rank |
| KumoRelational | 12 / 8 / 64 | 24 MiB | 384 MiB | 96 MiB/rank |

Actual measurements must report cache tensor dtype and backing storage. A four-way CP run does not quarter total model memory: if sharded ICL KV accounts for fraction `f` of live memory, ideal total-memory ratio is `(1-f) + f/4`, before communication buffers. Likewise, if ICL attention is fraction `a` of elapsed time, even zero-overhead infinite attention scaling cannot exceed `1/(1-a)`. This makes phase attribution necessary before investing in more invasive partitioning.

A meta-device census on `842c408fe` gives the following exact parameter counts. The FP32 column multiplies count by four and is not a measured GPU peak. Each row initializes one task, not both classification and regression networks.

| Model | Classification parameters | FP32 parameter bytes | Regression parameters | FP32 parameter bytes |
|---|---:|---:|---:|---:|
| KumoTabular small | 27,458,266 | 109,833,064 | 28,466,231 | 113,864,924 |
| KumoTabular medium | 61,485,274 | 245,941,096 | 62,492,087 | 249,968,348 |
| KumoTabular large | 213,668,250 | 854,673,000 | 215,683,191 | 862,732,764 |
| KumoRelational | 29,915,010 | 119,660,040 | 30,908,383 | 123,633,532 |

All are below one GiB of FP32 weights per task, making weight capacity alone a weak reason to introduce tensor parallelism/FSDP on 24–48 GiB GPUs. Caches, row activations, and sampled graphs need direct measurement. Parameter share is not compute share: the ICL block contains roughly 95% of large KumoTabular and 88% of relational classifier parameters, while row-encoder cost grows with data rows and columns.

Reproduce without loading pretrained weights or allocating GPU storage:

```python
from sdm.models import KumoRelational, KumoTabular

for task in ("classification", "regression"):
    models = [KumoTabular(task=task, size=size, pretrained=False, device="meta")
              for size in ("small", "medium", "large")]
    models.append(KumoRelational(task=task, pretrained=False, device="meta"))
    for model in models:
        count = sum(parameter.numel() for parameter in model.parameters())
        print(task, type(model).__name__, count, count * 4)
```

## Minimal public surface

Keep inference placement independent of model construction and preprocessing recipes. Use the existing `ICLModel.fit` / `predict` contracts and explicit replicas or process groups. Avoid introducing a cluster manager, service scheduler, mandatory configuration schema, or cloud dependency into SDM.

| Boundary | Proposed responsibility | Must remain independent |
|---|---|---|
| Generic ensemble executor | Assign logical members; retain their fitted state; preserve ordered finalization; forward thread-local autocast/inference state | Kumo-specific layers and cloud launch |
| DP example/runner | One persistent model per process; identical fit; fixed whole-batch assignment; indexed result gathering | Gradient DDP machinery and model internals |
| Attention context scope | Explicit group and global KV length; stable partial-attention reduction; shard metadata | Sampling, recipes, estimator scheduling |
| Model-specific stage plan | Move complete modules and associated caches; transfer activations at defined boundaries | Distributed dataset APIs |
| Experiment harness | Hardware setup, immutable input choices, timings, profiler, targets/metrics, failures | Public core model API |

A reusable core abstraction is warranted only once it has at least two real users. Query DP can remain a short `torchrun` example because it requires no model-internal synchronization. Use NCCL for GPU tensor collectives; CPU control or small output metadata can use a CPU-capable process group. Do not wrap inference in DDP just to replicate parameters: DDP is primarily a gradient synchronization mechanism, as described by [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html).

## Recommended progression

1. Establish fixed-input one-GPU public, batched-estimator, and executor baselines, including cache residency and model-core RNG parity.
2. Measure EP and DP on 1/2/4 GPUs. EP is useful when E is large enough to fill devices; DP is useful when independent batches or requests are plentiful. Include E1 and E8 so empty EP workers and independent query scaling are visible.
3. Add cached ICL CP on the same contexts, then longer contexts. Compare single-batch latency and memory, not only aggregate rows/s. Communicate global context length once and cache it, instead of `.item()` and all-reduce in every layer.
4. Profile phase shares. If relational table embedding dominates, prototype per-table placement before graph partitioning. If row encoder fit dominates, cached ICL CP will not solve the bottleneck.
5. Test EP × DP (for example two E4 groups across four devices) and EP × CP (two groups, each with two context shards) only after each component is independently correct.
6. Pursue tensor, full-fit context, graph, or pipeline parallelism where measured memory/latency limits justify their communication and maintenance costs.

## Attention recombination and pitfalls

For shard `s`, let `L_s` be its attention log-sum-exp and `O_s` its normalized output. With `m = max_s L_s`, the global result is `sum_s exp(L_s-m) O_s / sum_s exp(L_s-m)`. This preserves full softmax attention mathematically. Floating-point reduction order still changes, so bitwise parity is not a universal requirement. Aki's branch implements a maximum reduction and two sum reductions; its use of a private efficient-attention operator needs an explicit version/kernel compatibility test.

The communication is proportional to query output state, not KV length: approximately one maximum and two sums over query/head statistics/output each attention call. At 24 layers, collective latency can outweigh local work for short contexts, especially on PCIe hosts. GQA must preserve head mapping without permanently expanding the stored KV cache. Global-length-dependent query scaling must use the original context length. Empty shards need a neutral contribution (`L=-inf`, zero numerator); no-rank-data cases need a defined output or explicit rejection. Preserve query-self/diagonal contributions exactly once when an attention variant includes them.

[Ring Attention](https://arxiv.org/abs/2310.01889) describes another exact route: circulate KV blocks while processing distributed sequence blocks. It is a useful reference for distributed fit, but is more invasive than replicated-query reduction for cached inference. [FlashAttention](https://arxiv.org/abs/2205.14135) supplies the tiled exact-attention/normalizer foundation; neither paper proves performance for SDM's specific small-width tabular layers.

## Other viable approaches

| Approach | Concrete SDM implementation path | Decision criterion |
|---|---|---|
| Tensor parallel MLP/projections | Apply row/column sharding to Linear projections, preserve packed QKV/head layouts, all-reduce output projections | Single-member parameter/compute bottleneck and fast interconnect; first verify DTensor coverage of custom tensor boundaries |
| ICL layer placement | Contiguous layer groups with layer-local KV caches; query activations move between groups | Capacity benefit immediately; throughput benefit requires overlapping multiple microbatches |
| Encoder → ICL pipeline | Put row encoder (and relational GNN) on one device, ICL on another; retain each stage's fitted caches locally | Measured stage balance and enough queued queries; avoid claiming a serial stage split is pipeline speedup |
| Table encoder fan-out | Prepare task/table inputs once; dispatch independent table encodings; gather in original table order before GNN | Several similarly expensive tables and transfer cheaper than saved encoding |
| Graph partition | Assign node owners, exchange source/halo states per hop, combine exact segment sufficient statistics, gather task readouts | Graph state exceeds a device or GNN dominates; preserve min/max/moment aggregation and duplicate-edge semantics |
| CPU cache/offload/prefetch | Existing baseline path, improve residency scheduling separately | Memory fit and transfer/computation overlap; not multi-GPU by itself |
| Parameter sharding/offload | Load/gather layer parameters when needed | Parameter capacity problem; usually poor first choice when inference caches/activations dominate |
| Quantized/mixed-precision replicas | Same distributed method with smaller weights/cache where supported | New quality/numerical arm; do not attribute precision savings to parallelism |

PyTorch offers [DTensor tensor parallel styles](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel) and [pipeline stage/microbatch support](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html). These are reference APIs, not mandatory new dependencies or evidence of compatibility with SDM custom kernels. Pin the deployed PyTorch version before implementation; latest documentation may describe a newer version than the EC2 environment.

## High-value correctness and scale tests

| Test | Bug it can detect |
|---|---|
| E=1, E=5 over 2/4 GPUs; G>E | Dropped/duplicated members and empty workers |
| Different preprocessing groups and per-member class order | Wrong member routing or class-column reduction |
| Classification >10 classes and regression quantiles | ECOC/tree branch cache/RNG changes and output shape truncation |
| Independent fit, repeated predict, clear, refit with another context | Stale caches, rank lifetime and mutation bugs |
| Nontrivial untrained weights or pretrained weights | Zero-initialized residual branches can make an incorrect implementation appear equal |
| CPU/Gloo process tests plus real CUDA/NCCL tests | Distributed API correctness versus actual stream/device/kernel behavior |
| CP uneven context, fewer rows than ranks, varying lengths, GQA, FP32/BF16 | Empty-shard NaNs, local-length scaling and head alignment |
| Relational shared neighbors, disconnected nodes, duplicate edges, temporal cutoff, varying hops | Graph boundary/target ownership changes hidden by disconnected toy graphs |
| Fixed graph/hash and batches at 1/2/4 GPUs | Speedup caused by changing sampled work |
| Public CPU-offloaded versus resident serial versus parallel | Residency gains misreported as multi-GPU scaling |
| Tiny/medium/large query batches and short/long context | Collective launch overhead, pipeline bubbles, and saturation boundaries |
| Worker exception or Spot interruption | Hangs, partial outputs treated as complete, stale distributed process groups |

Per-rank random seeding alone is insufficient for EP: logical members must retain their seeds when they change ranks. DP replicas should match full fit state. CP participants must agree on input ordering, query tensors, model state, ensemble branch, and collective order. Hierarchical classification can introduce variable control flow; all ranks in a context group must traverse the same branches.

## Profile-directed follow-up experiments

These are proposed experiments, not implemented or measured optimizations. Choose an experiment after the corresponding signal appears in traces; measure it against the same existing baseline before combining it with another change.

| Observed signal | Focused experiment | Evidence needed to accept it |
|---|---|---|
| CP slower even on one rank with the LSE path | Replace or specialize the LSE attention backend; preserve GQA without materializing repeated KV heads | Native SDPA vs LSE1 kernel latency, temporary bytes, and full-model prediction tolerance |
| CP rank-local attention fast but NCCL dominates | Reduce/fuse collective launches, larger query batches, compare PCIe to NVLink/NVSwitch | Collective time/bytes and synchronization stalls; fixed batch/context and identical useful outputs |
| CP peak dominated by full fit | Shard fit's induced-column reductions and ICL attention, or fit once then scatter caches | Fit peak/latency and distributed fit parity; fitting once does not solve that rank's fit OOM |
| EP GPU idle while CPU transforms inputs | Cache transformed query members when legal; separate preprocessing and execution; compare persistent processes | CPU phase time, GPU idle intervals, transfer bytes, no state fitted on validation |
| EP scaling plateaus when estimators per GPU are small | Batch compatible estimators locally, then distribute batches | Best native estimator batching vs EP batching, same member plan and available memory |
| DP replicas dominated by immutable input copies | Share read-only host buffers or stage batches per worker; keep model/fit state local | Host RSS/transfer reductions without changing sampled graphs or moving labels into inputs |
| Relational encoder dominated by multiple similar tables | Table fan-out with stable per-table state and gather order | Per-table cost balance, transfer overhead, target/RNG/cache parity |
| One relational table dominates | Split its cached query rows, preserving fitted encoder state, and gather before unchanged GNN | Row encoder speedup and graph parity; avoid independently re-fitting row subsets |
| GNN dominates with large boundary-connected graph | Exact destination partition with source halos or merged sufficient statistics | Hop-wise state parity, boundary bytes, load balance, memory/latency crossover |
| Sequential placement reduces memory but idles devices | Overlap independent query microbatches through explicit stage queues | Pipeline occupancy, end-to-end p50/p95, steady-state throughput, bounded live buffers |

The current CP prototype expands reduced KV heads via `repeat_interleave` before its efficient-attention kernel, while native attention requests GQA directly. For KumoTabular large this turns two KV heads into sixteen during each call. The retained-cache formula remains correct, but temporary memory and memory traffic are different. A poor CP result on that path is evidence about the complete prototype/kernel combination; it does not by itself reject the mathematical context split.

Hybrid EP × DP should partition complete estimators inside each request and complete batches across request groups. Local averaging followed by averaging group outputs is only equivalent for a linear, equally weighted reducer; SDM regression can use trimmed estimator averaging after inverse target transforms. Preserve the original global estimator reduction rather than assuming associativity.
