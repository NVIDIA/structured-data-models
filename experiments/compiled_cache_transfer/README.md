# Order compiled cache copies with their readers

## Problem

On PyTorch 2.14/L4, public fullgraph prediction with two fitted cache batches offloaded to pinned CPU memory executes but returns incorrect probabilities. The maximum observed error is 0.37245, changing between repeated calls. This is not a precision-tolerance issue.

The generated Inductor wrapper shows why. It launches first-estimator computation on the compute stream, then explicitly reuses those cache buffers for the next transfer on another stream:

```python
with default_stream:
    # GPU kernels reading buf47 have been launched here.
    torch.ops.streams.record_stream.default(buf47, 0)
with stream1:
    buf650 = buf47; del buf47  # reuse
    buf650.copy_(arg64_1, True)
```

The wrapper has no compute-to-transfer wait before that overwrite. The first estimator's GPU work can still be reading the buffer. `record_stream` delays allocator reuse after deallocation; it cannot protect an explicit write through the same live buffer. The wrapper repeats this pattern for the transferred cache tensors. This is compiler-generated reuse, not the buffer allocation order in eager prediction.

## Candidate

Use the compute stream for cache transfers during compilation. Keep the existing transfer stream and asynchronous prefetch for eager prediction:

```python
if torch.compiler.is_compiling():
    transfer_stream = compute_stream
elif x.device not in self._transfer_streams:
    # Existing eager stream setup.
```

Compiling only the inner models leaves `predict()` eager, so that usage also retains asynchronous transfers. This preserves CPU cache offloading, cache contents, model calculations, and all `record_stream` lifetime tracking. Within compiled prediction, copies and their readers now execute in stream order; copy/compute overlap is sacrificed. This is an explicit workaround for compiler scheduling, not a numerical adjustment.

Source commit: `f0a5db1e4`, based on `18a4d151a` (the explicit-event fix) and frozen integration `c75ba3af6`. This branch is a separate candidate; its exact GPU replay is pending.

## Confirmed diagnostic

Setting the existing model transfer stream to the current compute stream before compiling passes on the original failing workload: Tabular classification, 256 context rows, 128 query rows, two cache batches, 70 pinned CPU cache tensors, FP32, no autocast, PyTorch 2.14.0+cu130, NVIDIA L4.

| Check | Result |
|---|---|
| Public prediction | `fullgraph=True`, one graph |
| Eight eager calls | Bitwise identical to the first eager output |
| Eight compiled calls | Bitwise identical to the first compiled output |
| All compiled calls versus first eager output | Pass `atol=1e-5, rtol=1e-4`; max difference 1.9968e-6 |
| Warm compiled median, five repetitions | 11.930 ms |
| Peak compiled PyTorch allocation | 169.779 MiB |

The diagnostic also makes eager prediction use the compute stream, so its eager timing is not a measurement of production asynchronous eager prediction. Memory measures PyTorch allocation only. [gpu_diagnostic.json](gpu_diagnostic.json) records artifact checksums and the exact diagnostic configuration.

CPU base/cache tests pass on both 2.7.1 and 2.14: 37 passed, one CUDA test skipped. These protect eager behavior but cannot validate the new CUDA compilation branch. Actual source replay on GPU, other cache counts/shapes, and performance of retained asynchronous overlap remain unconfirmed.
