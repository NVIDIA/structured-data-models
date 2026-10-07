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
- Experimental `context_parallel(group, kernel="flash")` / benchmark `--kernel flash` uses PyTorch's FlashAttention LSE output without repeating GQA KV heads. It requires CUDA FP16/BF16; FP32 is explicitly rejected. PyTorch's [FlashAttention integration](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp) exposes separate query and KV head counts. Separate 2/4-rank BF16 parity tests passed on L4; full-model regression quantile failures below remain a limitation despite kernel-level parity.

## Validation and measurement

`test/nn/test_context_parallel.py` launches real 2-rank and 4-rank Gloo process groups and, when GPUs are present, 2/4-rank NCCL groups under FP32 and BF16. It checks zero/one/uneven context lengths, MHA and GQA, broadcast batches, KumoTabular and relational ICL prediction parity, actual backing-storage release, mask rejection, and topology errors. Output projections are randomized so the zero-initialized residual defaults cannot make a broken attention path pass. The initial nine-case CPU/CUDA suite passed on L4. After adding precision/reduction variants, local CPU validation passed five cases with ten CUDA cases skipped; the added CUDA cases require the separately recorded follow-up run.

`research/multigpu/context_probe.py` is a torchrun microbenchmark of native SDPA, single-rank efficient/LSE, and distributed cached ICL. It records fit time/peak, cache bytes, synchronized repeated prediction times/peak, and prediction differences. Its random weights establish kernel behavior and scalability only; dataset quality must come from pretrained full-model runner measurements.

```sh
torchrun --standalone --nproc-per-node=4 research/multigpu/context_probe.py \
  --family tabular --context 4096 --queries 128 --channels 512 \
  --heads 8 --kv-heads 2 --layers 4 --dtype bfloat16 --output /tmp/cp-probe
```

Requested sweep: ranks 1/2/4, contexts 1024/4096/16384/32768, MHA (`--kv-heads 8`) versus GQA (`--kv-heads 2`), float32/BF16, both families. Use context/queries small enough for initial smoke and measure all ranks' latency; the slowest rank sets throughput. Completed cells are reported below; unmeasured cells are not implied by this sweep description.

## First measured results: four L4 GPUs

Source `55dba6bb5`, PyTorch 2.9.1+cu130, four NVIDIA L4 GPUs on one Spot host. These GPUs use PCIe without NVLink. A100/NVSwitch capacity was denied by the organization policy, so these measurements say nothing about NVLink scaling. All nine correctness cases passed on the GPU host, including 2/4-rank FP32 efficient attention and BF16 efficient/Flash attention, in 142.79 seconds. The first test attempt failed before execution because Linux's standard-library `test` namespace conflicted with pytest importlib's pickled module name; retrying with prepend mode passed. Tests now launch workers directly through torchrun to remove that namespace dependency.

The attention microprobe used random weights, four ICL layers, channels 512, eight query heads/two KV heads, context 1,024, query 128, and seven timed BF16 replays. It isolates attention implementation effects and is not a quality benchmark.

| Replay kernel | Native SDPA | One-rank LSE | CP4, slowest-rank median | CP4 / native speed |
|---|---:|---:|---:|---:|
| Efficient | 3.021 ms | 3.296 ms | 4.681 ms | 0.645x |
| Flash, native GQA | 3.117 ms | 3.271 ms | 4.568 ms | 0.682x |

Both four-rank microprobes had maximum logit difference 0.0009765625 versus native SDPA. Flash one-rank output was identical to native. These are short-context measurements; the collectives cost more than the partitioned attention work saves.

The first full-model workload used pretrained KumoTabular **small** (12 ICL layers, MHA), Covertype, fixed TRAIN context 1,024, fixed VAL query 2,048, E4, batch 256, seed 1729, FP32 parameters/BF16 autocast, and three timed passes. Query IDs and input hashes matched across arms. Default E4 fitted caches reside on CPU, so the cache columns below describe CPU storage; this is not a GPU-resident capacity claim.

| Arm | Unique rows/s, median | Accuracy | Log loss | ICL cache/rank | Total cache/rank | Prediction allocated peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| Native SDPA, 1 GPU | 3,082.25 | 0.756836 | 0.586389 | 96 MiB | 145.00 MiB | 265.63 MiB |
| Efficient LSE, 1 GPU | 3,085.89 | 0.755859 | 0.586496 | 96 MiB | 145.00 MiB | 265.63 MiB |
| Efficient CP, 2 GPUs | 2,648.21 | 0.755859 | 0.586468 | 48 MiB | 97.00 MiB | 241.13 MiB |
| Efficient CP, 4 GPUs | 2,576.37 | 0.754395 | 0.586577 | 24 MiB | 73.00 MiB | 228.88 MiB |
| Flash LSE, 1 GPU | 2,960.10 | 0.756836 | 0.586389 | 96 MiB | 145.00 MiB | 265.63 MiB |
| Flash CP, 2 GPUs | 2,604.96 | 0.756348 | 0.586461 | 48 MiB | 97.00 MiB | 241.13 MiB |
| Flash CP, 4 GPUs | 2,538.58 | 0.754395 | 0.586535 | 24 MiB | 73.00 MiB | 228.88 MiB |

Flash LSE1 predictions were byte-identical to the native baseline. Maximum probability differences were 0.007744/0.008601 for efficient CP2/CP4 and 0.009326/0.009801 for Flash CP2/CP4. The corresponding class-decision changes were 4/5 and 3/7 out of 2,048. This is tolerance-based equivalence, not exact prediction identity. The small changes in validation quality are measured outcomes, not evidence of an accuracy improvement or regression across datasets.

Each Flash CP batch generated 96 NCCL all-reductions per rank (two per layer × 12 layers × four estimators). CPU launch annotations summed to 5.03 ms for CP2 and 4.94 ms for CP4. The initial torch.profiler traces did not contain CUDA kernel events despite requesting them, so these values are **CPU launch spans**, not GPU communication time. Nsight follow-up is required for a compute/communication breakdown. Prediction timing itself used explicit CUDA synchronization and CPU-completed output copies. Cold first-fit time (8.27 seconds) versus subsequent runs (~1 second) is confounded by runtime/kernel cache warmup; no fit acceleration is claimed.

Raw evidence directories are `cp-probe4-c1024-{efficient,flash}`, `cp-covertype-small-c1024-e4-{native1,lse1,context2,context4}`, and `cp-covertype-small-c1024-e4-flash-{lse1,context2,context4}` under the shared experiment artifact root.

## Large model and resident cache at 16k context

Source `1686803e4`, same L4 host, pretrained KumoTabular **large** (24 ICL layers, 16 query heads, two cached KV heads), Covertype, context 16,384, query 2,048, E4, batch 256, three timed BF16 passes. Every arm moves its fitted cache to its assigned GPU once before warmup; transfer time and fit-plus-transfer transient peak are separately recorded. This isolates resident GPU cache scaling from repeated CPU-to-GPU transfers.

| Arm | Unique rows/s | Accuracy | Log loss | Resident ICL cache/rank | Total GPU cache/rank | Prediction allocated peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| Native SDPA, 1 GPU | 1,887.25 | 0.914551 | 0.225671 | 768 MiB | 1,080.00 MiB | 2.1895 GiB |
| Flash LSE, 1 GPU | 1,824.62 | 0.914551 | 0.225671 | 768 MiB | 1,080.00 MiB | 2.1895 GiB |
| Flash CP, 2 GPUs | 1,525.92 | 0.914551 | 0.225536 | 384 MiB | 696.00 MiB | 1.8164 GiB |
| Flash CP, 4 GPUs | 1,492.31 | 0.914551 | 0.225547 | 192 MiB | 504.00 MiB | 1.6284 GiB |

The one-rank Flash LSE control remains byte-identical to native. CP2/CP4 maximum probability differences are 0.009093/0.008519. Resident CP4 lowers prediction allocated peak by 25.6%, but throughput is 0.791x native. This demonstrates memory reduction without a speedup at this context length and topology. It does not solve the replicated fit peak. Evidence: `cp-covertype-large-c16384-e4-resident-flash-{native1,lse1,context2,context4}`.

## Native relational regression and a numerical failure

RelBench `rel-f1/driver-position`, native sampled `[16,16]` graph input, context 1,024, all 499 ordered validation queries in batches of 125, E4, same L4 host and three BF16 timed passes. The full graph encoder, relational operations, and final ICL head execute normally; this is not a flattened feature surrogate. Default fitted caches reside on CPU. The runtime lacks cuDF and emits CPU string-sort/join fallback warnings, so those upstream CPU costs are part of every arm.

| Arm | Unique rows/s | MAE | RMSE | Prediction peak/rank |
|---|---:|---:|---:|---:|
| Native SDPA | 364.23 | 3.414711 | 4.214993 | 338.86 MiB |
| Flash LSE, 1 GPU | 362.35 | 3.414711 | 4.214993 | 338.40 MiB |
| Flash CP, 2 GPUs | 361.15 | 3.418062 | 4.219112 | 313.90 MiB |
| Flash CP, 4 GPUs | 376.00 | 3.416702 | 4.217447 | 301.65 MiB |

Native and Flash LSE predictions are identical. All predictions are finite, all 999 quantiles remain monotonic, and each arm is repeat-stable. However, **BF16 CP fails the full-quantile numerical gate** `abs(actual-native) <= 0.01 + 0.05*abs(native)`: CP2 fails 938/498,501 entries and CP4 fails 915. Do not label these full-model results numerically equivalent merely because median predictions and aggregate MAE are close. CP2's worst tolerance excess is row 133, q002: native -0.268077 versus CP -0.354751. CP4's is row 63, q007: native 0.165292 versus CP 0.078619. The maximum absolute differences (~0.20224) occur in other, large-valued tail quantiles and do not identify the worst relative failures.

The diagnostic `--kernel efficient_fp32` preserves the ordinary BF16 Q/K/V input quantization but computes local normalized attention outputs and the global merge in FP32, delaying rounding until the normal output projection. It does **not** restore full-model numerical parity. Relative to the original BF16 native output, one-rank FP32-partial already fails 771 entries, CP2 fails 792, and CP4 fails 786. Relative to its own one-rank control, CP2/CP4 still fail 98/93 entries. Maximum errors versus BF16 native decrease from about 0.20224 to 0.14446, but this does not establish that partial-output rounding is the sole cause. Tolerances remain unchanged.

| FP32-partial arm, otherwise BF16 | Unique rows/s | MAE | RMSE | Gate failures vs BF16 native |
|---|---:|---:|---:|---:|
| One-rank LSE | 336.76 | 3.414720 | 4.215018 | 771 |
| CP2 | 365.78 | 3.414638 | 4.214971 | 792 |
| CP4 | 369.30 | 3.414846 | 4.215114 | 786 |

Evidence: `cp-f1-c1024-e4-flash-{native1,lse1,context2,context4}` and `cp-f1-c1024-e4-fp32partial-{lse1,context2,context4}`. A separate full-model FP32 comparison is needed to distinguish mathematical distributed-attention correctness from BF16 model sensitivity.

## Native H&M classification at 16k context

Source `5c8d19806`, native `rel-hm/user-churn` graphs with two-hop `[16,16]` sampling, actual context 16,384, 4,096 validation rows in eight batches of 512, E4, GPU-resident fitted caches, three BF16 timing passes on the same four-L4 host. These context graphs include roughly 319k related nodes; the full graph encoder remains replicated. Inputs and all query graphs reside on each rank for every arm. Existing raw CLI fields contain historical defaults; `input_identity.workload` and actual batch lengths are authoritative. The runner now records effective configuration separately from the original requested CLI.

| Arm | Unique rows/s | Accuracy | Log loss | AUROC | ICL cache/rank | Prediction peak/rank |
|---|---:|---:|---:|---:|---:|---:|
| Native SDPA | 1,061.85 | 0.807617 | 0.464014 | 0.662424 | 1,536 MiB | 2.3700 GiB |
| Flash LSE1 | 1,057.77 | 0.807617 | 0.464049 | 0.662432 | 1,536 MiB | 2.3700 GiB |
| Flash CP4 | 1,048.54 | 0.807617 | 0.464047 | 0.662425 | 384 MiB | 1.2450 GiB |

CP4 reduces prediction allocated peak by 47.5%, with throughput 0.987x native: memory relief without measured acceleration. Maximum probability differences are 0.008284 for LSE1 and 0.016804 for CP4; unlike the smaller workloads, even this native/LSE1 pair is not byte-identical. Both pass the fixed BF16 probability gate; CP4 changes no class decisions. No claim of improved prediction quality follows from the negligible metric differences. Evidence: `cp-hm-c16384-e4-resident-flash-{native1,lse1,context4}`.

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
