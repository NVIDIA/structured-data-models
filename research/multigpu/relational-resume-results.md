# Large native relational capacity results

Ten fresh H&M user-churn runs completed on replacement L4 Spot capacity. Native CPU-cache-offloaded inference succeeded at 65,536 context observations and eight ensemble members on one GPU. A two-GPU stage placement also succeeded with resident caches. One-GPU resident ensemble placement failed both with and without bounded GNN statistics, including retries with expandable allocation verified inside the benchmark process. Four-GPU ensemble placement also failed under the default allocator: it distributes members, not each member's full graph computation.

The bounded-statistics prototype reduced native fit peak **allocated** memory by 28–29%, but did not improve throughput or remove the tested resident-capacity barrier. Its general checkpoint numerical gate failed; the task-level comparisons below are experimental, not approval of a universally equivalent implementation.

## Fixed setup and measurement boundary

| Item | Value |
|---|---|
| Host | Replacement Frankfurt EC2 Spot host, four NVIDIA L4 GPUs; 22.04 GiB usable device capacity reported by CUDA |
| Runtime | Python 3.12, PyTorch 2.9.1+cu130, CUDA 13.0, Arrow fallback without cuDF |
| Immutable source | `ae4e5e549`, including the bounded-GNN output-lifetime correction |
| Checkpoint | KumoRelational classifier v1.0.1, revision `2bd603d3d8f25f67a7aaa8579908595e20567e22` |
| Graphs | Native temporal-last disjoint two-hop `[16,16]` `RelatedTables`, not flattened tabular input |
| Context / query | 16,384 or 65,536 TRAIN observations; 4,096 ordered VAL observations in eight complete batches of 512 |
| Model settings | E8, seed 1729, FP32 parameters with BF16 autocast, eight CPU threads |
| Repetitions | One warmup; three synchronized prediction passes on one fitted model per arm |
| Block setting | `--gnn-block-size 0` is unchanged GNN; `16384` bounds destination statistics blocks |
| Timing | Prepared CPU graph inputs through prediction and per-batch CPU output; final concatenation outside the default timer |
| Fit timing | One fresh-process fit per arm; earlier CUDA tests warmed some shared compilation caches |

Sampling and serialized graph preparation are excluded from the reported inference rates. Fit includes recipe, graph/model context processing, and cache creation. Peak allocator counters reset before fit and again before measured prediction. Failures preserve the traceback and allocator peaks, with current allocation sampled after exception unwinding. No failed arm supplies prediction quality or throughput.

The C16 graph has 319,383 nodes. C64 has 1,279,607 nodes, 2,571,320 directed edges, and four edge types. All query graphs have fewer than 16,384 nodes, so the block setting principally changes multi-block context fitting; each query uses one destination block. Within a context, every arm loads identical stored graph bytes. Across C16 and C64, query graph bytes differ slightly because context sampling precedes query sampling, despite identical VAL identities. Cross-context results are not an isolated context-length intervention.

Native preprocessing uses CUDA and the public CPU cache-offload policy. Resident ensemble/stage placement uses shared CPU recipe preprocessing and a placement-independent member RNG plan. Therefore native versus stage is a comparison of policies as well as placement. It is not a same-prediction speedup control.

## Allocator correction

The first eight runs requested `PYTORCH_ALLOC_CONF=expandable_segments:True`, with `PYTORCH_CUDA_ALLOC_CONF` unset. A fresh process on this exact runtime explicitly allocated a CUDA tensor and then inspected `torch.cuda.memory._snapshot()["allocator_settings"]`. It reported `expandable_segments: false` and an empty legacy-key setting. These eight runs are consequently labeled **default allocator**, not verified expandable. The inference about earlier runs uses their recorded matching runtime/environment; the probe was separate, after stage2 and before ensemble4, not a snapshot inside every preceding run.

Original requested-environment fields remain unchanged. The group `provenance.json` and additive `quality-allocator-correction.json` record the correction. The probe receipt SHA256 is `8afef22e1193240c8cf23588debb38e1e05374799077e11794056d7997d6ddf0`.

The final two fresh retries instead unset the newer key and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each wrapper allocated a CUDA tensor, saved actual allocator settings, asserted `expandable_segments: true`, and invoked the benchmark via `runpy` in that same process. Their own `provenance.json` receipts establish effective configuration. Both still failed; neither is omitted from the capacity conclusion.

## Successful runs

Rows/s is the median of all three passes, with min–max retained. Memory is GiB, not decimal GB. Stage memory entries list GPU0 / GPU1 separately.

| Context | Policy | Block | Fit s | Rows/s [min–max] | Fit peak allocated GiB | Fit peak reserved GiB |
|---:|---|---:|---:|---|---|---|
| 16,384 | Native, 1 GPU, CPU cache offload | 0 | 24.729 | 558.20 [555.03–558.25] | 4.326 | 6.992 |
| 16,384 | Native, 1 GPU, CPU cache offload | 16,384 | 24.832 | 549.88 [549.54–551.73] | 3.107 | 6.078 |
| 65,536 | Native, 1 GPU, CPU cache offload | 0 | 101.692 | 435.78 [435.52–435.81] | 16.737 | 19.342 |
| 65,536 | Native, 1 GPU, CPU cache offload | 16,384 | 101.895 | 433.44 [433.42–434.61] | 11.865 | 19.592 |
| 65,536 | Stage placement, 2 GPUs, resident cache | 0 | 99.528 | 512.93 [512.31–513.77] | 16.130 / 12.891 | 18.734 / 12.896 |

All five successes use the corrected default-allocator label. Blocking reduced allocated fit peaks by 28.2% at C16 and 29.1% at C64. C64 reserved peak increased, so a blanket claim of lower GPU memory is misleading. Median throughput ratios were 0.985× and 0.995× respectively; there is no observed speed benefit. Single fit durations are not repeated-fit estimates.

Native prediction peak allocated memory was unchanged by blocking: 1.396 GiB at C16 and 3.697 GiB at C64. Native stored CPU cache storage was 3,273,162,816 bytes and 12,936,839,232 bytes respectively. Stage2 stored 12 GiB of ICL cache on GPU1 plus 51,937,344 bytes of other cache on GPU0; prediction peaks were 0.502 / 12.150 GiB. Logical cache bytes and unique physical storage bytes are separately retained in raw JSON.

Stage2 places row encoding/GNN on GPU0 and the complete ICL block/cache on GPU1. It is not a pipeline-concurrency experiment. Its 512.93 rows/s is 1.177× the native offload rate, but that ratio includes cache policy and recipe/RNG differences; it must not be presented as an exact two-GPU speedup. Native already fits on one GPU with offload. The valid capacity finding is that stage2 supplies a measured resident-cache path where the tested resident ensemble attempts failed.

## Failed resident attempts

All failures below occur during fit at C64/E8. Recorded peaks exclude the allocation that failed. Default and verified-expandable attempts are separate directories and are not silently pooled.

| Policy | GPUs | Block | Effective allocator | Failure allocation and operation | Peak allocated GiB / GPU |
|---|---:|---:|---|---|---:|
| Resident ensemble | 1 | 0 | Default | 6.10 GiB five-statistics output | 16.188 |
| Resident ensemble | 1 | 16,384 | Default | 2.44 GiB full-node LayerNorm | 15.865 |
| Resident ensemble | 4 | 0 | Default | 6.10 GiB five-statistics output; all four devices fail | 16.188 |
| Resident ensemble | 1 | 0 | Verified expandable | 1.22 GiB skip-linear output | 21.015 |
| Resident ensemble | 1 | 16,384 | Verified expandable | 2.44 GiB full-node LayerNorm | 20.376 |

The final verified blocked failure reported 2.19 GiB free for a 2.44 GiB request, 19.43 GiB allocated, and only 157.92 MiB reserved but unused. The verified unblocked attempt reported 585.12 MiB free for a 1.22 GiB request. These are still active-capacity failures, not evidence that merely selecting a different allocator solves the workload. Different failure locations and progress also mean failed-run peak differences are not completed-fit savings measurements.

No same-policy C64 resident ensemble reference completed: EP1 and EP4 failed. Consequently stage2 has no full-output same-policy C64 placement comparison in this batch. Earlier smaller-context placement parity is separate evidence, not a substitute. Bounded full-node normalization/GELU is an unimplemented follow-up hypothesis; no savings or success from it were measured here.

## Prediction quality and numerical limits

All five completed arms have byte-identical repeats within each fitted model. The first-repeat NPY, PT archive, logged quality, and per-repeat arrays use the corrected provenance contract. Metrics below explicitly select churn class `1`.

| Context / policy | Block | AUROC | Log loss | Accuracy | Maximum probability change vs matching native block0 |
|---|---:|---:|---:|---:|---:|
| C16 native | 0 | 0.665599079 | 0.463186711 | 0.808105469 | — |
| C16 native | 16,384 | 0.665978837 | 0.463077426 | 0.808105469 | 0.010768354 |
| C64 native | 0 | 0.667022017 | 0.461203009 | 0.808349609 | — |
| C64 native | 16,384 | 0.667003759 | 0.461197615 | 0.808349609 | 0.008892119 |
| C64 stage2 | 0 | 0.670340667 | 0.460325718 | 0.808105469 | Not a same-policy comparison |

Both blocked/native pairs preserve all hard labels but change probabilities. C64 mean absolute probability change is 0.000949429. Independent audit found the configured final-probability BF16 tolerance passes; its paired log-loss difference was approximately −0.000005398 with bootstrap interval [−0.000110854, +0.000102543], spanning zero. Do not interpret descriptive AUROC or loss differences as a demonstrated quality improvement.

The broader correctness result is less favorable: 57 synthetic CUDA tests passed, but the actual-checkpoint C512 screen had 23 numerical failures and eight passes at unchanged tolerances. Unblocked controls were exact. A focused first-hop diagnostic found bitwise-identical statistics, while changing projection GEMM shape alone changed outputs (FP32 maximum 4.7207e-5; BF16-autocast maximum 0.125). Repeated graph normalization can amplify these differences. Historical CPU singleton-block failures are also retained. Thus task-specific final tolerance does not establish generally equivalent GNN semantics or justify enabling the adapter by default.

## Reproduction and evidence

Successful commands use the existing runner; for example, from the immutable source directory with the recorded offline model cache:

```bash
/opt/pytorch/bin/python -m research.multigpu.relational_bench run \
  --workload /home/ubuntu/kumo-multigpu/results/workload-hm-c65536-b512 \
  --mode native --gpus 1 --estimators 8 --dtype bf16 --threads 8 \
  --warmups 1 --repeats 3 --seed 1729 --gnn-block-size 0 \
  --source-commit ae4e5e549 --output NEW_ATTEMPT_DIRECTORY
```

Exact invocation/environment files are authoritative, including the verified allocator wrapper. Use new attempt directories; do not overwrite outcomes. Model checkpoint SHA256 is `0c2e35cc18303f0d3ddeaf04e9cab6ca3a8bd6ea17195ac0e42d47af4bcb2a2d`. Stored graph identities are:

| Context | `graphs.pt` bytes | SHA256 |
|---:|---:|---|
| 16,384 | 60,365,937 | `79df998020561488bb4cb18e6ec8fc81090e69209ede50eebf9f8d4b8add4f3d` |
| 65,536 | 136,973,041 | `1176d9e75c8f019920cce1f8887d1360f518de254e7850af8a45f9f4dc2183a6` |

All ten outcome directories and sibling logs are downloaded under `.kumo-multigpu-20261008/results/relational-resume/`, named `relational-resume-l4-hm-c{16384,65536}-e8-{native,ensemble1,ensemble4,stage2}-gnn{0,16384}` for the applicable arms above. The two allocator retries append `-expandable-verified`. Successes retain raw predictions for every repeat, quality, timing windows, telemetry and invocation; failures retain tracebacks and memory receipts. Independent audit sidecars are additive. Raw arrays and model bytes are not committed to Git.

These are fresh replacement-host measurements, not recovery of the lost old L40S C16/E8 queue. That queue's outcome remains unknown, as documented in [the earlier relational report](relational_results.md). Graphs use TRAIN context and VAL queries only; evaluation labels were opened after model inference. No TEST data was used.
