# Experimental compiled chunk mapping

Review prototype, not a production recommendation. Based on main after #1334 and #1335.

- `sdm/_chunk.py` maps a pure, row-independent tensor transform over bounded chunks using the private PyTorch higher-order `map` API.
- `CellEmbedding` uses it only for compiled inference without extra row inputs. Ordinary eager execution and gradient-enabled execution retain their existing Python loops.
- The assembled result is allocated once, with chunk-sized temporaries inside the loop. A supplied output still needs the assembled temporary and a final copy.
- Short inputs are processed directly. Longer inputs repeat the last row for final-chunk padding and discard padded results. There is no maximum input size or arbitrary cap.
- Output layout is unspecified. Functions that use randomness, mutate state, or mix mapped rows do not satisfy the helper's contract.

## Trace the example

Validated with PyTorch 2.14 on CPU and NVIDIA L4 CUDA. From the repository root:

```sh
PYTHONPATH=. TORCH_LOGS=recompiles,dynamic,graph_breaks python -u examples/torch_compile_chunked_map.py > chunk-map.txt 2>&1
```

The example compares each compiled result with eager, prints its own call markers and graph count, and fails if later calls recompile after three initial versions. Rows 7, 12, 20, 35, 7, 64, and 81 produced graph counts 1, 2, 3, 3, 3, 3, and 3 in the isolated prototype. The initial short-input and multi-chunk specializations remain.

## Measurements from the isolated prototype

These are synthetic CellEmbedding measurements on the same L4, not full KumoTabular API results. Compilation is excluded. Latency is the median of ten synchronized CUDA-event samples after warmup; memory is additional peak allocated above live inputs, weights, and any supplied output. MB is decimal megabytes.

| Workload | Existing eager | Existing compiled loop | Compiled map |
|---|---:|---:|---:|
| 4,096 rows, no supplied output: latency | 2.872 ms | 0.599 ms | 1.090 ms |
| Same: additional peak allocation | 5.54 MB | 8.39 MB | 5.80 MB |
| 10,000 rows, supplied output: latency | 2.031 ms | 2.827 ms | 2.404 ms |
| Same: additional peak allocation | 31.35 MB | 119.54 MB | 123.63 MB |

The smaller case has 16 columns, 16 channels, group size two, eight frequencies, and 256 rows per chunk. The larger case has 32 columns, 64 channels, group size four, 16 frequencies, and 1,024 rows per chunk; its supplied output is a noncontiguous view, and the 81.92 MB target is excluded from each allocation delta.

FP16 autocast and FP16-input comparisons matched the original compiled embedding bitwise on the tested CPU/CUDA configurations. Both versions can differ slightly from eager due to compiled FP16 arithmetic. Direct output-mutation alternatives were rejected after producing incorrect results with batched supplied outputs.

This trades some inference latency and memory behavior for graph reuse across changing chunk counts. It does not promise a universal speedup, training support, CUDA graph capture, or lower first-call compilation time. Validate each additional caller before using the helper elsewhere.
