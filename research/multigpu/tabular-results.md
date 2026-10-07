# KumoTabular measurements

## Initial small-context strong-scaling experiment

Covertype validation: large KumoTabular checkpoint v1.0.1, 1,024 TRAIN context rows, 2,048 fixed validation queries, 54 numerical features, E4, query batches of 256, BF16 autocast with FP32 parameters. All modes use the same preprocessing seed 1729. Resident ensemble modes additionally use member seeds 1729–1732. Hardware is four L40S GPUs on one EC2 Spot host, with PyTorch 2.9.1+cu130 and Python 3.12.3. Each entry has three timed passes after warmup, and pass timing includes CPU output transfer. Each configuration runs in a fresh process.

| Execution | GPUs | Estimators per model call | Median rows/s | Interpretation |
|---|---:|---:|---:|---|
| Public native, initial process | 1 | 1 | 1,818.93 | Initial baseline |
| Public native, later control | 1 | 1 | 1,758.63 | Repeat after the ensemble ladder |
| Resident ensemble | 1 | 1 | 1,729.77 | Matched resident-cache scaling reference |
| Resident ensemble | 2 | 1 | 1,883.05 | 1.089× versus resident one GPU |
| Resident ensemble | 4 | 1 | 1,644.81 | 0.951× versus resident one GPU |
| Public native, estimator batching | 1 | 2 | 2,984.01 | Existing single-GPU batching is a major control |
| Public native, estimator batching | 1 | 4 | 4,007.14 | More than twice the naive two-GPU ensemble throughput |

This workload does not show useful four-GPU ensemble scaling. The tuned single-GPU baseline is substantially faster than the unbatched multi-GPU prototype. Additional experiments must preserve estimator-batching controls before claiming multi-GPU benefits. Kernel launch, preprocessing, transfer, and scheduling causes remain hypotheses until trace analysis; throughput alone cannot identify which dominates.

Native and resident ensemble one/two/four-GPU predictions are byte-identical, including the repeated native control. Their validation accuracy is 0.76416015625, macro one-versus-rest AUROC 0.9474149901961447, and float32 sklearn log loss 0.5760445594787598. Batched native inference changes numerical results slightly: estimator batch four has the same accuracy, AUROC 0.9474122196492643 and log loss 0.5759778022766113. The raw predictions are retained for a full deviation audit; equal aggregate accuracy does not imply identical predicted labels.

The initial native fit took 122.75 seconds, while its later repetition took 1.327 seconds. Resident fits took 1.075, 1.114, and 1.298 seconds on one, two, and four GPUs. The initial cold fit must not be used to claim a parallel fit speedup. First-use runtime behavior is a plausible explanation, but no specific compiler or driver cause has yet been established.

Resident one-GPU prediction peak allocation was 1.716 GB. Resident four-GPU prediction peak allocation was approximately 1.262 GB per GPU, and parameter copies increased aggregate memory. Native estimator-batch four used 1.767 GB peak allocated and 3.322 GB peak reserved during prediction. These are allocator measurements, not estimates of total process GPU memory.

Raw host output directories are `/home/ubuntu/kumo-multigpu/results/tabular-native-large-e4-c1024-q2048`, `tabular-ep{1,2,4}-large-e4-c1024-q2048`, and `tabular-native-warm-eb{1,2,4}-large-e4-c1024-q2048`. They contain JSON configuration/runtime/timings/memory/metrics, NumPy predictions and row identities/targets, and 200 ms device telemetry. The baseline uses source `9f0d6c765`; subsequent runs use source `6c3f1e22e`, which fixes the ensemble stream-recording API. The standalone runner hash is recorded in every JSON result.

The original ensemble CUDA smoke failed because it called a nonexistent `TableTensor._tensors()` method. Replacing this with the supported `record_stream` operation fixed both caller-stream/autocast and replica-initialization/changed-stream CUDA tests. The fixed tests were observed passing in 0.80 seconds before the GPU ensemble ladder. The failed attempt remains part of the experiment record.

## Larger-context batching controls

Covertype E8 with 4,096 context rows, 8,192 validation queries, and query batch size 1,024 changes the result. Existing single-GPU estimator batching must still be tuned on this exact workload.

| Execution | GPUs | Local estimator batch | Median rows/s |
|---|---:|---:|---:|
| Native | 1 | 2 | 4,433.85 |
| Native | 1 | 4 | 3,777.01 |
| Native | 1 | 8 | 3,219.44 |
| Resident batched ensemble | 1 | 2 | 4,583.76 |
| Resident batched ensemble | 2 | 2 | 6,626.77 |
| Resident batched ensemble | 4 | 2 | 5,875.07 |
| Resident batched ensemble | 1 | 8 | 4,013.81 |
| Resident batched ensemble | 2 | 4 | 7,368.45; reverse-order repeat 7,282.49 |

Holding local estimator batch size at two gives 1.446× two-GPU and 1.282× four-GPU speedups over the resident one-GPU reference. All of these batch-two predictions, including native, are byte-identical. Two GPUs with local batch four are faster, but their reference predictions differ from the batch-two group. The originally observed 1.836× speedup compares local batch eight on one GPU with local batch four on two GPUs; those two configurations produce identical outputs but are not the fastest one-GPU settings tested.

Changing local batch two versus four/eight changes BF16 numerical results enough to fail the predeclared pointwise tolerance at one probability entry: maximum difference 0.0322531, mean difference 0.000526267, six predicted-class changes among 8,192 queries. The batch-two group has accuracy 0.8419189453 and log loss 0.4139244854; the batch-four/eight group has accuracy 0.841796875 and log loss 0.4138842225. Independent paired bootstrap intervals for the log-loss difference include zero. The numerical failure is retained; matching local batch size establishes that device placement itself preserves predictions.

## Regression and nested hybrid

California Housing uses 4,096 TRAIN context rows and 4,096 validation queries, E8, query batch 512, and fixed estimator batch two on every configuration. Every one of the 999 output quantiles is byte-identical across native and resident one/two/four-GPU execution. Median prediction RMSE is 0.4168959066, MAE is 0.2449084520, and there are no quantile-order violations.

| Execution | GPUs | Median rows/s | Peak allocated per GPU, GB |
|---|---:|---:|---:|
| Native | 1 | 2,815.71 | 1.469 |
| Resident batched ensemble | 1 | 2,662.00 | 1.775 |
| Resident batched ensemble | 2 | 2,575.15 | 1.494, 1.482 |
| Resident batched ensemble | 4 | 2,316.35 | 1.303, 1.291, 1.291, 1.291 |

This narrower eight-feature workload with 999 output quantiles does not benefit from the threaded ensemble executor. This is a measured workload limit, not a general claim about regression models.

A four-GPU hybrid with two query workers, each containing a two-GPU E4 ensemble, was measured on the initial Covertype workload. Both it and the resident one-GPU reference begin with CPU query batches and include input/output transfer. The hybrid achieved 1,640.59 rows/s versus 1,763.70 rows/s for the reference, with byte-identical predictions. Nesting query threads around ensemble threads did not improve this workload.

## Targeted memory and launch experiments

The initial Nsight traces capture worker CUDA activity that the main-thread PyTorch profiler misses. Native E4 execution launches 40,664 kernels in its profiled prediction pass; four-GPU unbatched ensemble execution launches 48,552. The extra 7,840 kernels are FP32-to-BF16 casts, rather than cache-packing kernels. Kernel execution occupies only a small part of the profiled pass. These instrumented runs establish launch behavior, not clean-run throughput.

Holding an outer autocast context open on each worker retained approximately 267 MB more memory per GPU but yielded only small throughput differences: 1,753.36 rows/s for one GPU and 1,660.36 for four GPUs on the initial Covertype workload. Predictions remained byte-identical. This intervention does not rescue four-GPU scaling; its cast-count effect still requires the paired Nsight trace.

Compacting retained cache views reduced unique resident storage from 512.75 MB to 358.61 MB and prediction peak allocation from 1.716 GB to 1.561 GB. It took 3.23 ms during fit and preserved prediction bytes. Throughput was 1,816.92 rows/s, but the modest difference from the earlier resident reference requires interleaved repetitions before attribution. Inspection identifies value views into fused column-attention KV buffers as the retained storage; the reduced ICL KV-head selection was already contiguous and was not the cause.
