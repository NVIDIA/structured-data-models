# Cached context parallelism for Kumo

Prototype branch: `research/kumo-multigpu-context-20261008`, based on `842c408fe`.

## Supported mathematical split

KumoTabular's ICL transformer and KumoRelational's shared TabICLv2 ICL transformer perform query-to-context attention during fitted-cache replay. The queries are replicated on every rank and each rank retains a contiguous shard of every layer's key/value context. Local attention produces an output and log-sum-exp normalizer. A global maximum followed by a packed numerator/denominator sum produces the globally normalized attention result before the output projection, residual, normalization, or MLP. Reducing after those nonlinear operations would be incorrect.

`origin/aki/cp` (`499485af5`) supplied the design precedent. This implementation additionally handles empty/uneven shards and broadcast batch dimensions, uses FP32 reduction for half precision, combines numerator and denominator in one collective, records the global context length once per fitted layer, and rejects incompatible cache reuse. Global query scaling must use the original context length: local lengths change both KumoTabular LogScale and relational QASSMax semantics. No host `.item()` is required in attention.

## Interface

```python
import torch
import torch.distributed as dist
from sdm.nn.context_parallel import context_parallel

with torch.inference_mode(), context_parallel(dist.group.WORLD):
    model.fit(context, target)  # same full context on every rank
    predictions = model.predict(query)  # same query batches on every rank
```

For KumoRelational, pass the normal related tables and hop configuration to fit/predict. This scope only changes the final ICL attention. Table embedding, preprocessing, graph sampling/message passing, model parameters, and non-ICL caches remain replicated. Each rank must use identical weights, random preprocessing plans, sampled neighborhoods, and batch order. Outputs are replicated; count query throughput once, not once per rank.

Fit still computes the full context on every GPU, then copies only each local ICL cache shard. `clone()` is deliberate: an already-contiguous slice can keep the full backing allocation alive. This reduces resident ICL cache storage, not full-context fit compute or transient peak memory. KumoTabular medium/large retain only two query KV heads, so their ICL caches are already relatively compact; collective overhead may dominate.

## Supported and excluded cases

- Supported: inference-only cached replay, variable batch shapes via broadcasting, MHA/GQA, empty and uneven context shards, float32 and BF16 kernel paths, CPU dense correctness reference, CUDA efficient attention with returned LSE.
- Explicitly rejected: masks and valid-length tensors while distributed ICL attention is active; sharded cache replay without matching world/rank topology; ordinary unsharded cache replay inside a parallel scope; one-shot KumoTabular calls without a fitted cache; autograd-enabled scopes.
- Unsupported research boundaries: torch.compile; distributed context fitting; arbitrary query partitions within the same group; graph partitions; training; serializing and moving caches to a different rank topology. CPU offload via the existing Cache tensor traversal preserves the scalar topology metadata, but offload timing needs measurement.
- CUDA LSE uses PyTorch's private efficient attention operator, also used by Aki. This is a version-sensitive prototype dependency. Native SDPA and one-rank efficient/LSE baselines must both be measured to distinguish kernel choice from distribution.

## Validation and measurement

`test/nn/test_context_parallel.py` launches real 2-rank and 4-rank Gloo process groups and, when GPUs are present, 2/4-rank NCCL groups under FP32 and BF16. It checks zero/one/uneven context lengths, MHA and GQA, broadcast batches, KumoTabular and relational ICL prediction parity, actual backing-storage release, mask rejection, and topology errors. Output projections are randomized so the zero-initialized residual defaults cannot make a broken attention path pass. Local result: 3 tests passed and 4 CUDA cases skipped; existing Kumo/TabICLv2 ICL regression suite plus the original CP tests passed 19 with 10 CUDA skips. Distributed CUDA evidence is pending.

`research/multigpu/context_probe.py` is a torchrun microbenchmark of native SDPA, single-rank efficient/LSE, and distributed cached ICL. It records fit time/peak, cache bytes, synchronized repeated prediction times/peak, and prediction differences. Its random weights establish kernel behavior and scalability only; dataset quality must come from pretrained full-model runner measurements.

```sh
torchrun --standalone --nproc-per-node=4 research/multigpu/context_probe.py \
  --family tabular --context 4096 --queries 128 --channels 512 \
  --heads 8 --kv-heads 2 --layers 4 --dtype bfloat16 --output /tmp/cp-probe
```

Requested sweep: ranks 1/2/4, contexts 1024/4096/16384/32768, MHA (`--kv-heads 8`) versus GQA (`--kv-heads 2`), float32/BF16, both families. Use context/queries small enough for initial smoke and measure all ranks' latency; the slowest rank sets throughput. No GPU speedup or accuracy claim has been established yet.

`research/multigpu/context_model_bench.py` adds pretrained full-model comparisons using the team's fixed tabular arrays and native relational sampled graphs. Run `--mode native` and `--mode lse` with one torchrun process each; run `--mode context` with 2/4 processes. Keep all other parameters fixed. It saves every prediction repeat, query IDs, per-rank timings/peaks, resident ICL versus total cache bytes, quality metrics, and optional per-rank traces. Validation targets are opened after prediction only. Query throughput uses each repeat's slowest rank and counts replicated output rows once. Inputs and graphs are resident on each GPU for all three modes; this is a controlled execution comparison, not host-to-GPU streaming throughput.

```sh
HF_HUB_OFFLINE=1 torchrun --standalone --nproc-per-node=4 \
  research/multigpu/context_model_bench.py --family tabular \
  --data /data/covertype --output /results/covertype-cp4 \
  --mode context --size small --context 1024 --queries 2048 \
  --batch-size 256 --estimators 4 --repeats 3 --profile
```

## Further viable extensions

The prototype distributes only fitted-cache replay. Fit attention could also shard KV while retaining replicated queries/hidden rows, recombining every layer before residual and MLP. That partitions quadratic attention work but communicates full context-sized outputs and still replicates row embeddings/MLPs. A more complete distributed fit would partition query rows too, exchanging KV or circulating KV blocks while maintaining online softmax. It must preserve globally fitted preprocessing and induced-column attention, relational sampling/graph boundary semantics, hierarchical class routing, and estimator seeds. Splitting raw context into independent model fits and averaging their predictions is a different estimator, not exact context parallelism.

Query/data parallelism can wrap CP groups: split independent query batches across groups while sharing each group's context shards. Ensemble parallelism can assign distinct estimator groups similarly. These hybrids trade less cross-GPU replication against collective traffic. A useful next decision is whether long-context attention dominates total model time after the replicated preprocessing/row/GNN stages; the full-model trace determines that.
