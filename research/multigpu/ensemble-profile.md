# Initial ensemble CPU/CUDA trace analysis

The initial small KumoTabular workload is dominated by launching many short CUDA kernels, with very little concurrent GPU computation. This explains why adding thread-based replicas does not provide useful scaling for this case. The evidence supports launch overhead and low concurrency; it does not independently prove that the Python GIL is the sole cause.

Configuration: pretrained KumoTabular large, Covertype seven-class classification, E4, 1,024 context rows, 2,048 query rows, query batches of 256, BF16 autocast with FP32 parameters, L40S host. Native uses estimator batch size one and pinned-CPU member caches. EP4 uses one resident member per GPU. This native profile is a diagnostic control, not the fastest native baseline: the separate clean benchmark found native estimator batch size four substantially faster.

The analysis selects only the explicit `prediction_pass` NVTX range, excludes load/fit, and unions overlapping kernel intervals on each GPU. CUDA API times below are sums of CPU call durations and must not be added to wall time. Trace instrumentation changes timing, so these durations are not clean benchmark throughput results.

| Trace metric | Native, one GPU | Resident EP, four GPUs |
|---|---:|---:|
| Traced prediction range | 1.62963 s | 1.53082 s |
| Kernel launches | 40,664 | 48,552 |
| GPU0 kernel-active union | 199.846 ms, 12.26% | 62.655 ms, 4.09% |
| GPU1 kernel-active union | — | 58.521 ms, 3.82% |
| GPU2 kernel-active union | — | 59.545 ms, 3.89% |
| GPU3 kernel-active union | — | 59.537 ms, 3.89% |
| Range with no GPU compute kernel active | 87.74% | 85.44% |
| Range with at least two GPUs computing | — | 1.085% |
| Range with all four GPUs computing | — | 0.00148% |
| Kernel launch APIs, summed CPU duration | 307.228 ms | 500.530 ms |
| CPU threads issuing kernel launches | 1 | 5: coordinator plus four workers |
| Host-to-device copy bytes | 2,868,913,632 | 1,376,512 |
| Host-to-device copy duration sum | 216.561 ms | 0.215 ms |
| Device-to-host copy bytes | 59,648 | 1,436,288 |
| Device-to-host copy duration sum | 0.385 ms | 0.470 ms |
| Reported peer-copy bytes | 0 | 192 |

“No compute kernel active” does not mean the entire GPU is idle: copy engines can still be active, particularly in the native CPU-cache baseline. The resident executor eliminates almost all cache H2D traffic, but its host launch behavior prevents that saving from translating into effective four-GPU computation overlap.

All 7,888 additional EP kernels are accounted for by demangled-name differences: 7,840 extra float-to-BF16 conversion kernels and 48 integer-add kernels. GEMM, flash-attention, reduction, and other kernel counts match. Therefore the additional kernels should not be described as generic cache-view packing. A likely cause is per-call worker autocast lifetime, which can repeatedly discard cached parameter casts, while native holds an outer autocast context over the prediction pass. `persistent_autocast:factory` tests this hypothesis with a fixed dtype and worker-lifetime outer context; it must demonstrate fewer casts, preserved predictions, and improved clean timing before adoption.

The cache storage issue is separate: row-encoder value views can retain unused fused KV backing storage. `compact_ensemble:factory` tests that memory mechanism. Compaction is not guaranteed to save memory for every context size; a larger-context run may already allocate compact cache tensors.

A read-only copy-size check also found 32 device-to-host copies of 56 bytes in each trace, matching seven int64 class labels across four members and eight query batches. This is consistent with the model's `classes.tolist()` output-label construction. Their total device-copy durations were only 0.053 ms native and 0.043 ms EP4; all `cudaStreamSynchronize` CPU durations summed to 1.77 and 1.79 ms respectively. The metadata synchronization exists, but these traces do not identify it as a dominant part of the approximately 1.5-second inference range. CUDA graph replay also moves this metadata to CPU, so any future graph improvement must not be attributed solely to fewer kernel launches without a separate control.

Coverage checks find GPU kernel records on all four EP devices and launch records from all four workers plus the coordinator. Nsight diagnostics report missing OS scheduling information and several ancillary-process NVTX/CUDA warnings. The requested top-level inference ranges are present, and the four model workers have CUDA coverage, but these traces cannot establish Python thread scheduling or GIL wait fractions. A persistent-process comparison is needed to test interpreter isolation directly.

## Reproduction and source artifacts

Run `python research/multigpu/analyze_nsys.py <trace.sqlite> --phase prediction_pass --output <analysis.json>`. The script reads SQLite in read-only mode, records per-device interval unions, concurrency fractions, demangled kernel counts, per-thread launch counts, API duration sums, copy kinds/bytes, and profiler diagnostics.

Local evidence directory: `/Users/ardrianw/repositories/.kumo-multigpu-20261008/profiles`.

| File | SHA-256 |
|---|---|
| `tabular-nsys-native.sqlite` | `4c844934ad008e55940a4db7ae96de1435ae5565295eac34631d46169b4e370f` |
| `tabular-nsys-ep4.sqlite` | `e38f22c274029a0fac8fbb0af29c3ed2d1dc632ce8b17530291c394a6936c296` |
| `tabular-nsys-native-analysis.json` | `89e493c3d645544e60bc29305673bc6bfcd86933acb766897bb2a23e7a2f9754` |
| `tabular-nsys-ep4-analysis.json` | `d9b15ca7efc6ebee4e4aeee5b65961d1c27516c79aed17a3045814bfafb90d39` |
