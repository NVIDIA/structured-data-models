# Explicit events for cache-transfer ordering

PyTorch 2.14 fullgraph GPU prediction fails when its stream-dependency pass moves a use of a tensor constant before that constant is defined:

```
Argument '_tensor_constant9' of Node 'control_deps' was used before it has been defined
```

This occurred in KumoRelational one-hop prediction on RelBench driver data. The constant is introduced by an indexed scalar assignment after prediction waits for its cache-transfer stream. It is a compiler graph-ordering failure, not a data or precision error.

The candidate replaces both waits with their equivalent event sequence:

```python
# Before
compute_stream.wait_stream(transfer_stream)

# After
compute_stream.wait_event(transfer_stream.record_event())
```

The second expression is exactly the implementation of `torch.cuda.Stream.wait_stream` in both installed PyTorch versions. Writing it explicitly chooses the compiler's event handling instead of its faulty `wait_stream` handling. Cache copies remain asynchronous, prefetch overlap remains in place, and `record_stream` lifetime tracking is unchanged. There is no compiler-specific branch or change to model calculations.

## Evidence and limits

`reproduce.py` constructs a plain FX graph with no SDM imports or GPU allocation. On PyTorch 2.14, the compiler pass produces invalid ordering for `wait_stream`; the explicit event graph remains valid. This reproduces the cause independently of the full model.

```bash
python experiments/cache_stream_events/reproduce.py
```

### Confirmed CUDA result

On an NVIDIA L4, PyTorch 2.14.0+cu130, `torch.compile(model.predict, backend="inductor", fullgraph=True, dynamic=True)` passes KumoRelational one-hop prediction on RelBench driver data. Query rows change **4 → 1 → 8 → 4**: three compiled graphs, no graph breaks. Both eager and compiled use FP32 weights with autocast off and TF32 matmul off; all predictions meet the unchanged `atol=1e-5, rtol=1e-4` check. Maximum absolute difference is 2.30e-6; predicted classes agree completely.

| Eight-query-row case | Eager | Compiled |
|---|---:|---:|
| Warm median wall latency | 74.855 ms | 41.013 ms |
| Peak PyTorch allocated memory | 142.190 MiB | 141.863 MiB |

Latency excludes compilation and uses five repetitions after two warmups. This is a small, one-estimator workload; it does not establish larger-workload performance. Memory covers PyTorch allocations, not total device usage or separate cuDF allocations. All 51 cache tensors were already on the GPU.

The standalone `cuda_reproduce.py` also passes with explicit events for resident CUDA tensors and asynchronous pinned-host-to-CUDA transfers, with both graph-break settings, across lengths 32/128/7. It requires exact output equality. The original `wait_stream` fails all four configurations with the same constant-ordering error. **Offloaded multi-cache full-model prediction and actual CUDA validation on PyTorch 2.7 remain pending.**

The measured library is frozen integration `c75ba3af6c231c7bdcd181a50d35cdc7fb53c090` plus source patch `18a4d151a`. The runner is based on `937beb883`, with cache-device reporting and an unused single-stream diagnostic option. [gpu_summary.json](gpu_summary.json) records the source, runner, input, and raw-result checksums plus summarized measurements.

Exact model command on the validation host:

```bash
cd /home/ubuntu/validation/repo-events
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 \
TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
TORCHINDUCTOR_CACHE_DIR=/home/ubuntu/validation/cache/events-fp32-full \
/home/ubuntu/validation/torch214/bin/python \
  /home/ubuntu/validation/run-diagnostics.py \
  --model relational --entry predict --device cuda --autocast off \
  --fullgraph --capture-dynamic-outputs --arm-index 0 \
  --query-indices 1 0 2 1 --source-commit c75ba3af6+18a4d151a \
  --data /home/ubuntu/validation/data/driver-dnf_bundle.pt \
  --checkpoint /home/ubuntu/validation/data/relational-classifier.pt \
  --output /home/ubuntu/validation/results/relational-onehop-predict-fp32-full-events18.json
```

The runner defaults used here are one estimator, two warmups, and five measured repetitions. FX/AOT disk caches were disabled; the normal in-process compiled graphs remained active.

The source delta is commit `18a4d151a`, two replacements in `sdm/models/base.py`, based on frozen integration `c75ba3af6`. The separate resident-cache fast path is not included, keeping attribution clear.
