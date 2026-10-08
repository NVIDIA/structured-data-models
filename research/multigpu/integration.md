# SDM multi-GPU integration design

Inspection baseline: `842c408fe`, 2026-10-08. Architecture descriptions follow source inspection; the outcome-directed recommendations below also use the explicitly scoped measurements in [README.md](README.md).

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

The study branch contains experimental executors, adapters, cloud-operation notes, runners, profilers, and evidence retention. Their combined size is not a proposed production API. Any upstream change should be a separately reviewable extraction of a small reusable capability with behavior tests, rather than a wholesale merge of the investigation harness.

The coordinator has published a separate scoped candidate, [`feature/ensemble-parallel`](https://github.com/NVIDIA/structured-data-models/tree/feature/ensemble-parallel), at [`75bd74af2`](https://github.com/NVIDIA/structured-data-models/commit/75bd74af21d7541e7ef042871af5e3f076203a17), directly based on `842c408fe`. Its four files contain the generic resident threaded executor, export, focused tests and usage documentation—not CP, process/graph adapters, cloud orchestration or the full study. The caller constructs distinct identical evaluation replicas; preprocessing/reduction run once, member cache ownership is stable, calls are synchronous and must not overlap on one executor. This is an experimental opt-in contract, not automatic native estimator batching or a guarantee of speedup. Process/graph benchmark gains must not be attributed to this smaller branch. No PR is implied by the confirmed branch push.

| Boundary | Proposed responsibility | Must remain independent |
|---|---|---|
| Generic ensemble executor | Assign logical members; retain their fitted state; preserve ordered finalization; forward thread-local autocast/inference state | Kumo-specific layers and cloud launch |
| DP example/runner | One persistent model per process; identical fit; fixed whole-batch assignment; indexed result gathering | Gradient DDP machinery and model internals |
| Attention context scope | Explicit group and global KV length; stable partial-attention reduction; shard metadata | Sampling, recipes, estimator scheduling |
| Model-specific stage plan | Move complete modules and associated caches; transfer activations at defined boundaries | Distributed dataset APIs |
| Experiment harness | Hardware setup, immutable input choices, timings, profiler, targets/metrics, failures | Public core model API |

A reusable core abstraction is warranted only once it has at least two real users. Query DP can remain a short `torchrun` example because it requires no model-internal synchronization. Use NCCL for GPU tensor collectives; CPU control or small output metadata can use a CPU-capable process group. Do not wrap inference in DDP just to replicate parameters: DDP is primarily a gradient synchronization mechanism, as described by [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html).

### Candidate extraction boundaries

| Candidate | Potential destination | Required before promoting from research |
|---|---|---|
| Cache residency and stream lifetime semantics | Existing `Cache` / `ICLModel` fitted execution | One documented default-preserving policy; unique backing-storage accounting; clear/refit/device-lifetime tests; avoid separate model copies for each policy |
| Resident ensemble execution | One generic model executor shared by Tabular and Relational | Reuse existing recipe/member batching and aggregation, preserve stable logical member RNG, reduce duplicated lifecycle code, validate actual GPU stream/failure behavior, demonstrate useful target workloads |
| Partial attention and stable distributed recombination | Small optional `sdm.nn` primitives | Explicit group/global length/topology metadata, GQA and empty shards, version-tested backend, full model tolerances/quality, clear limitations on fit/compile/training |
| Query data parallel usage | One short public example with `torchrun` | Fixed batch identities and full relational neighborhoods, local fit per rank, indexed output merge, clear initialization and teardown |
| Layer placement hooks, process factories, hybrid adapters | `research/multigpu` until interfaces converge | Replace hooks with explicit stages/cache ownership if adopted; demonstrate capacity/pipeline benefit and interactions with public fitted APIs |
| Benchmark runners, EC2 launch operations, result collector and raw evidence | Research/benchmark documentation and scripts | Remain optional and out of import-time model code; preserve reproduction and failure history |

The initial small-context measurements favor tuning existing estimator batching before adding a new default execution mode. They do not justify enabling EP or CP automatically. Explicit opt-in research modes allow capacity and workload-dependent benefits to be evaluated without burdening every model-family wrapper.

The resumed pretrained ladder adds a useful research result: graph replay reaches 6,398 rows/s on one L40S and 11,146 on four, with exact native-batch1 outputs; four GPUs scale 1.742× over graph1. Process EP scales 2.724× but is only 1.109× tuned native. Graph execution also changes class-metadata synchronization, so the whole gain cannot be assigned to fewer launches alone. Keep static-shape capture, warmup, output lifetime, refit invalidation and memory controls explicit. Neither adapter is included in the published minimal candidate.

Those paired profiles now show direct `cudaLaunchKernel` calls dropping from 39,832 to 944 for graph1, while executed kernels increase. This supports dispatch amortization, not less GPU arithmetic or a GIL-only conclusion. Profile-derived kernel-active fractions are not hardware occupancy; use uninstrumented timings for throughput.

Native relational process EP subsequently scales 1.944× from one to four workers on its fresh same-host, final-gather-included workload, preserving every output repeat. Full-model FP32 resolves the tested F1 CP numerical screen but is not faster than its FP32 native reference; all-gather remains slower than single-rank resident attention on the tested 16k workload. These additions support keeping throughput, precision policy and retained-cache capacity as separate decisions. They do not justify automatically selecting CP or a process executor.

Destination-block GNN is not ready for default integration: 23 actual-checkpoint GPU cases fail hidden-output gates, even though its exploratory E8/C16k H&M final probabilities pass the looser task-level screen. A 28.2% fit-peak reduction with essentially unchanged/slightly slower throughput is a measured memory–numerics tradeoff, not an exact replacement. Keep native execution default and retain both levels of validation.

### What the measured results prioritize

1. **Throughput with many independent batches: persistent process DP; optionally capture supported static tabular shapes.** The larger-batch Covertype graph-DP1/4 control reaches 11,207/41,649 rows/s including final ordered CPU gathering: 3.716× matched graph scaling and 1.410× same-shape native-DP4. The separate batch256 ladder scales 3.731× and improves 1.581× over its native-DP4. Predictions are byte-identical within each backend across GPU counts; graph versus native estimator-batch4 passes the unchanged BF16 gate but is not exact (50 label flips/65,536 rows at batch1,024). Keep backend/batching numerical policy explicit, with no quality-benefit claim. Graph lowers live prediction allocation here but increases reserved memory at the larger batch. This is not a universal graph recommendation or maximum throughput claim. Native batching/process DP is the simpler generic option and already scales 3.254× on relational H&M in the original worker-return timer. Publish a small persistent-worker/`torchrun` example with complete query batch ownership, not another distributed model class. Optional graph support needs bounded shape-cache growth, warmup/capture accounting, output lifetime, refit invalidation and fallback contracts; preserve startup, replicated fit state, IPC and final gather costs. Neither process/graph backend is in the minimal published resident-EP candidate.
2. **A single wider ensemble: batch compatible local members, then use resident EP.** Covertype E8 gives 1.446× on two GPUs at fixed tuned local batch-two execution, with exact predictions. The earlier 1.836× comparison also changes local batch width; native batch-two is a stronger baseline than native batch-eight. Four-way output is exact to native batch-two but fails the original batch-eight numerical gate, isolating a batching-shape policy difference. Reuse native batching and output finalization rather than preserving a duplicate single-member loop as production architecture.
3. **Capacity without new attention arithmetic: sequential stage/layer placement.** The L4 tabular layer split approximately halves prediction peak per device with exact predictions and a small throughput penalty. This supports an explicit capacity option, not a throughput promise or an overlapped pipeline claim. Stage/cache ownership should be formalized before hook-based research code becomes public.
4. **Retained long-context memory: keep CP experimental.** Resident 16k GQA caches shrink as expected, but both tested collective variants remain slower than native/single-rank execution on PCIe. F1 BF16 regression fails the full-quantile gate; FP32 local partial attention also fails its matched single-rank comparison. Full-model FP32 subsequently passes every strict quantile comparison but is not faster than its FP32 reference and costs about20% throughput versus BF16 native. Preserve these distinct precision controls; do not auto-select CP by GPU count or treat aggregate quality proximity as numerical equivalence.
5. **More invasive splits remain profile-gated.** Tensor parallelism, exact graph partition, table fan-out, full-fit context sharding, and overlapped pipelines have design paths but no measured implementation result in this report. Weight sizes alone do not justify tensor sharding. Large H&M graph/activation pressure may justify fit or encoder work; use measured phase memory and time, not architectural estimates, to select the next implementation.

These rankings concern the measured PCIe L40S/L4 hosts and selected workload sizes. They are not a general rejection of CP on faster fabrics or a proof that process DP always wins single-request latency. Original process timers exclude final CPU concatenation; the later tabular control explicitly includes it. California Housing's exact-output EP slowdown and the small threaded hybrid's negative scaling show why opt-in, measured policies are preferable to automatic placement by GPU count.

Build the process-DP guidance on the existing `examples/kumo/relational/multi_gpu.py` rather than introducing a second orchestration stack. Add explicit fixed-batch identities, the relational complete-neighborhood contract, a tabular local-batching example, failure propagation, and gathered-output timing. Keep the reusable contract at the model/fit/predict boundary: a process factory can remain example-specific without requiring every future SDM model family to implement a serving protocol.

The completed heterogeneous cross-region controls reinforce that boundary. Balanced mixed2→8 query DP scales 3.970× with fixed per-row GPU-family ownership. Capacity-weighted assignment adds 28.7% at eight GPUs, but changes ownership and therefore some floating-point outputs; every row must be checked against its assigned-family reference. Preloaded input, network output and coordinator location are explicit benchmark policies. Keep fleet discovery, SSH transport and weighted host scheduling in examples/research, not inside `KumoTabular`, `KumoRelational` or a mandatory SDM distributed runtime.

## Recommended progression

1. Establish fixed-input one-GPU public, batched-estimator, and executor baselines, including cache residency and model-core RNG parity.
2. Measure EP and DP on 1/2/4 GPUs. EP is useful when E is large enough to fill devices; DP is useful when independent batches or requests are plentiful. Include E1 and E8 so empty EP workers and independent query scaling are visible.
3. Add cached ICL CP on the same contexts, then longer contexts. Compare single-batch latency and memory, not only aggregate rows/s. Communicate global context length once and cache it, instead of `.item()` and all-reduce in every layer.
4. Profile phase shares. If relational table embedding dominates, prototype per-table placement before graph partitioning. If row encoder fit dominates, cached ICL CP will not solve the bottleneck.
5. Test EP × DP (for example two E4 groups across four devices) and EP × CP (two groups, each with two context shards) only after each component is independently correct.
6. Pursue tensor, full-fit context, graph, or pipeline parallelism where measured memory/latency limits justify their communication and maintenance costs.

## Attention recombination and pitfalls

For shard `s`, let `L_s` be its attention log-sum-exp and `O_s` its normalized output. With `m = max_s L_s`, the global result is `sum_s exp(L_s-m) O_s / sum_s exp(L_s-m)`. This preserves full softmax attention mathematically. Floating-point reduction order still changes, so bitwise parity is not a universal requirement. Aki's branch implements a maximum reduction and two sum reductions; its use of a private efficient-attention operator needs an explicit version/kernel compatibility test.

The communication is proportional to query output state, not KV length: one maximum and numerator/denominator sums over query/head statistics/output each attention call. The current prototype packs the two sums into one buffer, using two all-reduce launches per call rather than Aki's original three; an optional all-gather variant exchanges all partial outputs before a local merge. At 24 layers, collective latency can outweigh local work for short contexts, especially on PCIe hosts. GQA must preserve head mapping without permanently expanding the stored KV cache. Global-length-dependent query scaling must use the original context length. Empty shards need a neutral contribution (`L=-inf`, zero numerator); no-rank-data cases need a defined output or explicit rejection. Preserve query-self/diagonal contributions exactly once when an attention variant includes them.

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

The initial CP efficient-attention path expands reduced KV heads via `repeat_interleave`, while native attention requests GQA directly. For KumoTabular large this turns two KV heads into sixteen during each call. The retained-cache formula remains correct, but temporary memory and memory traffic differ. A poor result on that path is evidence about the complete prototype/kernel combination; it does not by itself reject the mathematical context split.

The follow-up Flash LSE variant (`926f6dc3f`) is now implemented and passed multi-rank CUDA tests. PyTorch 2.9.1's [CUDA attention implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/transformers/cuda/attention.cu) returns the Flash attention log-sum-exp, and its [dispatch constraints](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) explicitly support GQA for Flash while the efficient backend requires matching head counts. This motivated using Flash for supported BF16/FP16 inputs without KV expansion; the version-sensitive private API remains a research boundary. Initial short-context probes still showed distributed overhead, so long-context measurements must determine its practical value.

Hybrid EP × DP should partition complete estimators inside each request and complete batches across request groups. Local averaging followed by averaging group outputs is only equivalent for a linear, equally weighted reducer; SDM regression can use trimmed estimator averaging after inverse target transforms. Preserve the original global estimator reduction rather than assuming associativity.
