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

`test/nn/test_context_parallel.py` launches real 2-rank and 4-rank Gloo process groups. It checks zero/one/uneven context lengths, MHA and GQA, broadcast batches, KumoTabular and relational ICL prediction parity, actual backing-storage release, mask rejection, and topology errors. Output projections are randomized so the zero-initialized residual defaults cannot make a broken attention path pass. Local result: 3 tests passed in 13.01 seconds on CPU; distributed CUDA evidence is pending.

`research/multigpu/context_probe.py` is a torchrun microbenchmark of native SDPA, single-rank efficient/LSE, and distributed cached ICL. It records fit time/peak, cache bytes, synchronized repeated prediction times/peak, and prediction differences. Its random weights establish kernel behavior and scalability only; dataset quality must come from pretrained full-model runner measurements.

```sh
torchrun --standalone --nproc-per-node=4 research/multigpu/context_probe.py \
  --family tabular --context 4096 --queries 128 --channels 512 \
  --heads 8 --kv-heads 2 --layers 4 --dtype bfloat16
```

Requested sweep: ranks 1/2/4, contexts 1024/4096/16384/32768, MHA (`--kv-heads 8`) versus GQA (`--kv-heads 2`), float32/BF16, both families. Use context/queries small enough for initial smoke and measure all ranks' latency; the slowest rank sets throughput. No GPU speedup or accuracy claim has been established yet.
