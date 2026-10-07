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

Next measurements cover larger contexts and query batches, E8, batched resident ensembles, query parallelism, independent-process query workers, hybrid combinations, and separate Nsight traces. Results above establish only this initial configuration.
