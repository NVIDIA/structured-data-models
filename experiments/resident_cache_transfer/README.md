# Avoid unnecessary cache-transfer streams

A GPU prediction with an already-resident cache still created a transfer stream, traversed the cache with `.to(device)`, and waited for that stream. These operations move no data in this case.

PyTorch 2.14 also mishandles this wait during fullgraph compilation: its stream-dependency pass makes a later tensor constant an input of an earlier `control_deps` node, producing:

```
Argument '_tensor_constant9' of Node 'control_deps' was used before it has been defined
```

`upstream_repro.py` reproduces the same graph-ordering error on CPU with a plain FX graph and no SDM imports. The actual GPU case was one-estimator KumoRelational prediction on RelBench driver data; all 51 cache tensors were already on cuda:0. Setting the transfer stream to the current stream still failed because the compiler retained the wait operation.

## Change

Prediction checks whether any cached tensor differs from the query device. It creates a transfer stream and waits only when transfer is needed. If any cache needs copying, the original asynchronous prefetch path remains.

Current-stream acquisition and `tensor.record_stream` are retained for every CUDA prediction. Residency does not establish allocation-stream ownership: a caller can predict on another stream or clear the cache before queued work completes.

The check follows the existing `Cache._tensors()` traversal used by `Cache.to()`. Non-tensor metadata is ignored; empty caches require no transfer; nested or mixed-device caches trigger transfer when any tensor differs. It does not change prediction computations, callbacks, gradient mode, cache offload policy, or output precision.

## Validation

- CPU PyTorch 2.7.1 and 2.14.0: existing base-model/cache suites each report 37 passed, 1 GPU skip, including callback lifecycle and train-mode checks.
- GPU correctness and latency validation of the resident-cache fast path is pending in the coordinated GPU investigation.
- CPU-offloaded/multi-estimator caches retain asynchronous transfer. The underlying PyTorch fullgraph stream bug can still affect that path; this change does not claim to fix the compiler itself.

Source branch is based on frozen integration `c75ba3af6`. The candidate source consists of `86a3f7a4a` plus lifetime correction `27dde06aa` (net 9 additions, 3 removals in `sdm/models/base.py`).
