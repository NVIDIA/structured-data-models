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

## CUDA profile and collective overhead

An independent Nsight Systems CUDA/NVTX trace of resident-cache Covertype large C16k/E4/CP4 captured all four CUDA processes, all four fit ranges, and all 32 expected prediction batch ranges. `context_trace.py` scopes kernel/copy/API records by both owning process and its NVTX intervals, avoiding double-counting the other ranks' work inside overlapping ranges. The original report, SQLite export, diagnostics, and derived JSON are retained as `cp-large16k-nsys.*` and `cp-large16k-nsys-analysis.json`.

| Rank | Eight-batch traced wall | GPU kernel interval union | Kernel-active fraction | NCCL kernel sum/count |
|---|---:|---:|---:|---:|
| 0 | 1.894 s | 1.346 s | 71.1% | 0.892 s / 1,536 |
| 1 | 1.894 s | 1.401 s | 74.0% | 0.960 s / 1,536 |
| 2 | 1.894 s | 1.386 s | 73.2% | 0.941 s / 1,536 |
| 3 | 1.894 s | 0.695 s | 36.7% | 0.250 s / 1,536 |

The count matches two all-reductions × 24 ICL layers × four estimators × eight batches. Rank 0's 768 Flash split-KV attention kernels total only 64.30 ms. Non-NCCL kernels total roughly 0.44–0.45 seconds per rank, whereas collective kernel times vary substantially: NCCL time includes waiting for peers and cannot be interpreted as pure transfer bandwidth. GPU kernel-active fraction measures the presence of a running kernel, not SM occupancy. CUDA launch APIs add roughly 352–368 ms of CPU spans per rank; these may overlap GPU work and must not be added to GPU durations to reconstruct wall time. There are no frontend-versus-ICL nested NVTX scopes, so a full module-level split is unavailable from this trace.

The resident-cache run copies only 64 bytes H2D and 59,744 bytes D2H per rank inside the prediction ranges; recorded D2D copies total 12 MiB. Their summed CUDA copy durations are under 1.1 ms per rank. These copy records do not account for traffic generated inside NCCL kernels. NCCL initialization confirms `SHM/direct/direct` ring transport and `isAllDirectP2p 0`, consistent with this host's disabled peer access. Fit remains replicated: no NCCL kernels occur inside fit, and each GPU is kernel-active for about 7.7–7.8 of the traced 9.6–9.7 seconds.

Profiling perturbs execution: the traced prediction pass is about 1.894 seconds versus about 1.37 seconds in the unprofiled timing matrix. Use the latter for throughput claims. Nsight also emits missing-NVTX/CUDA warnings for auxiliary processes; the expected four worker traces and named ranges are present, and diagnostics remain in the derived record.

This profile motivates the separately controlled `reduction="all_gather"` experiment: one packed output/LSE collective per ICL layer, followed by a stable local merge, instead of MAX then SUM all-reductions. It retains the default algorithm unchanged. For Q256/H16/D64 FP32 communication, the local packed tensor is 1,064,960 bytes and the CP4 gathered tensor is 4,259,840 bytes. An idealized ring sends about 3.19 MB per rank for all-gather versus 1.62 MB across the two all-reductions; actual transport traffic was not measured. Thus fewer launches trade against higher traffic and a world-size-scaled temporary buffer, not a free reduction in communication.

## Further viable extensions

The prototype distributes only fitted-cache replay. Fit attention could also shard KV while retaining replicated queries/hidden rows, recombining every layer before residual and MLP. That partitions quadratic attention work but communicates full context-sized outputs and still replicates row embeddings/MLPs. A more complete distributed fit would partition query rows too, exchanging KV or circulating KV blocks while maintaining online softmax. It must preserve globally fitted preprocessing and induced-column attention, relational sampling/graph boundary semantics, hierarchical class routing, and estimator seeds. Splitting raw context into independent model fits and averaging their predictions is a different estimator, not exact context parallelism.

Query/data parallelism can wrap CP groups: split independent query batches across groups while sharing each group's context shards. Ensemble parallelism can assign distinct estimator groups similarly. These hybrids trade less cross-GPU replication against collective traffic. A useful next decision is whether long-context attention dominates total model time after the replicated preprocessing/row/GNN stages; the full-model trace determines that.

## Historical interruption boundary

This subsection records the first session's evidence boundary. The resumed measurements below supersede its CUDA-correctness, FP32, and all-gather deferrals; they do not retroactively change the earlier BF16 failures.

The final proposed GPU block was staged as `run-context-final.sh` against source `4e1c5c33d`, but **was not launched**. Before its GPU lease began, the environment changed to network-restricted execution; a read-only SSH check failed with `Operation not permitted`. No alternate network route or permission bypass was attempted. Therefore the following remain unmeasured, not failed benchmark results:

- Added CUDA correctness cases for `efficient_fp32` and `all_gather`; the earlier nine-case GPU suite passed, and the expanded local CPU suite passed five cases with ten CUDA skips.
- Full-model FP32 F1 native versus CP2. Its predeclared FP32 numerical gate is `atol=1e-5, rtol=1e-4`; the looser BF16 gate must not substitute for it.
- Controlled resident large-C16k all-reduce versus all-gather CP2/CP4 throughput and quality. The all-gather option is CPU-tested only and should not be promoted as a validated CUDA optimization.
- Resident small-MHA C32k native/LSE1/CP2/CP4. No long-MHA crossover or absence of a crossover is established by the available short-MHA and 16k-GQA measurements.
- CP8 and NVLink/NVSwitch coverage. Hardware/policy limitations prevented those configurations; PCIe/shared-memory results do not establish their scaling.

The local evidence contains 21 completed pretrained full-model arms plus two random-weight microprobes, with result JSON and saved predictions, and the completed Nsight report/SQLite/log/derived analysis. The last CP GPU profiling process exited successfully, and all corresponding downloads completed before the restriction. No final-block CP process was started. Subsequent remote process state and instance teardown cannot be verified from this restricted session; cloud ownership and cleanup remain with the coordinating operator.

Current integration recommendation: keep context sharding experimental. It provides measured resident-cache and prediction-peak memory relief on both Kumo families, but no convincing throughput improvement in the measured matrix. BF16 relational regression fails the declared all-quantile screen, and the FP32-partial alternative does not repair it. Ensemble/data parallelism or placement may be preferable depending on the workload; context fitting, graph encoding, and preprocessing are still replicated here.

## Resumed full-FP32 isolation

After access was restored, source `c0fb64fcc` ran on a newly provisioned four-L4 Spot host in Frankfurt, with PyTorch 2.9.1+cu130. Models and the retained F1 graph were independently SHA-verified before execution. The four added CUDA cases—2/4-rank `efficient_fp32` and 2/4-rank Flash `all_gather`—passed in 110.15 seconds, including their uneven/empty shard and model tests. This closes the earlier CUDA-testing deferral for these variants. Each completed benchmark directory was downloaded before starting the next timed arm.

The F1 experiment uses the same native C1024 graph, all 499 validation queries, batch125, E4, seed1729, public CPU-cache offloading, and three saved prediction passes. Its graph SHA256 is `23c439d0ff2771af5fcaed8735607d1fe762ee2236bb5451eceda7cfdf1adcb2`. FP32 applies to the complete model, not just partial attention; parameters and fitted floating caches are FP32, matmul precision is `highest`, and CUDA matmul TF32 is disabled. cuDNN TF32 remains enabled as recorded; this was not a blanket backend-disable experiment. Native and LSE1 use the same hardware/runtime and partial kernel control.

| Precision / arm | Unique rows/s | MAE | RMSE | Prediction peak/rank | ICL cache/rank (CPU) |
|---|---:|---:|---:|---:|---:|
| BF16 native | 377.43 | 3.414711 | 4.214993 | 338.86 MiB | 96 MiB |
| FP32 native | 303.73 | 3.401794 | 4.201006 | 416.39 MiB | 192 MiB |
| FP32 efficient LSE1 | 299.96 | 3.401794 | 4.201006 | 416.39 MiB | 192 MiB |
| FP32 efficient CP2 | 300.31 | 3.401794 | 4.201006 | 367.39 MiB | 96 MiB |

The complete FP32 CP2 output passes the original FP32 gate (`atol=1e-5, rtol=1e-4`) across all 498,501 quantile entries: maximum absolute difference is 1.335144e-5 and mean absolute difference is 8.991849e-7. Native versus LSE1 is byte-identical. This supports the distributed algorithm's numerical validity under full FP32 for this workload; it does not make the failed BF16 or partial-FP32 results pass. CP2 delivers 0.989x FP32-native throughput and 0.796x BF16-native throughput. Full-model FP32 therefore restores this numerical gate at a substantial precision-policy cost, without a demonstrated latency gain. Single cold-fit timings are retained but not interpreted as steady-state fit speedups because startup/compiler cache state differs across first uses.

Evidence: `cp-resume-20261008-a1/f1-c1024-e4-{bfloat16-native1,float32-native1,float32-lse1,float32-context2}`, including all three prediction arrays, effective configuration, numeric-backend flags, and per-rank wall/memory records. This is a new-host matched comparison; the earlier L4 measurements remain historical, not cross-host controls.

| F1 arm | Cold load | Cold fit | Fit allocated peak/rank | Total fitted cache/rank (CPU) |
|---|---:|---:|---:|---:|
| BF16 native | 3.203 s | 13.407 s | 779.24 MiB | 129.81 MiB |
| FP32 native | 0.770 s | 4.977 s | 1,100.21 MiB | 259.63 MiB |
| FP32 LSE1 | 0.780 s | 3.320 s | 1,100.21 MiB | 259.63 MiB |
| FP32 CP2 | 0.794 s | 3.308 s | 1,100.21 MiB | 163.63 MiB |

These startup observations are maxima across ranks and include first-use effects; the first BF16 arm's longer fit does not establish that FP32 is faster at fitting. FP32-native doubles the fitted floating cache and raises fit peak 41.2% and prediction peak 22.9% relative to BF16-native. FP32 CP2 halves ICL cache relative to FP32-native, but total CPU cache still exceeds BF16-native because replicated non-ICL tensors remain FP32. The quality audit independently confirmed the strict gate and repeat/rank consistency for all three prediction passes.

## Resumed one-collective comparison

On the same new four-L4 host and source, the complete Covertype large C16k/E4/Q2048/B256 resident-cache ladder compares native SDPA, single-rank Flash LSE, and Flash CP2/CP4 with either default two-all-reduce merging or packed all-gather. Each arm has three saved prediction passes; each pass uses the slowest rank and counts validation rows only once.

| Arm | Unique rows/s | Slowest-rank pass range | Prediction peak/rank | ICL GPU cache/rank |
|---|---:|---:|---:|---:|
| Native SDPA | 1,853.19 | 1.1041–1.1059 s | 2,242.03 MiB | 768 MiB |
| Flash LSE1 | 1,807.12 | 1.1279–1.1345 s | 2,242.03 MiB | 768 MiB |
| CP2 all-reduce | 1,500.81 | 1.3636–1.3702 s | 1,860.03 MiB | 384 MiB |
| CP2 all-gather | 1,541.44 | 1.3201–1.3289 s | 1,860.03 MiB | 384 MiB |
| CP4 all-reduce | 1,489.72 | 1.3689–1.3758 s | 1,667.53 MiB | 192 MiB |
| CP4 all-gather | 1,525.77 | 1.3383–1.3456 s | 1,667.53 MiB | 192 MiB |

All-gather's nominal throughput changes are +2.71% at CP2 and +2.42% at CP4 relative to the corresponding default merge. These three-pass ranges do not overlap within this ordered run, but there are no randomized arm-order or independent run replications: treat the differences as small measured changes, not a robust general speedup. Neither approach beats native or LSE1. All-gather CP4 remains only 0.823x native throughput. Its larger temporary does not raise this workload's whole-prediction peak because other allocations dominate; that observation does not remove its world-size-scaled temporary or higher communication volume.

All six arms have accuracy 0.91455078. Native and LSE1 are byte-identical. CP2's reduction variants are also byte-identical, while CP4 all-gather differs from CP4 all-reduce by at most 0.002825916; changing floating-point reduction order is not guaranteed bitwise invariant. Versus native, all CP arms pass the fixed BF16 numerical screen and change two class decisions. Maximum probability differences are 0.009093225 (both CP2 arms), 0.008518979 (CP4 all-reduce), and 0.008754194 (CP4 all-gather). Log losses are 0.22567126 native/LSE1, 0.22553590 CP2, 0.22554731 CP4 all-reduce, and 0.22554880 CP4 all-gather; these negligible differences are not evidence of improved predictive quality.

Evidence: `cp-resume-20261008-a1/covertype-large-c16384-e4-resident-flash-{native1,lse1,all_reduce2,all_gather2,all_reduce4,all_gather4}`. All ten resumed pretrained arms and added CUDA test logs were retrieved locally before releasing the GPU lease; a final remote process check found no CP worker or GPU compute process. The implementation remains experimental. Long-MHA C32k, CP8, and NVLink/NVSwitch measurements remain unperformed; full context/graph fit sharding remains future work rather than an implemented capability.
