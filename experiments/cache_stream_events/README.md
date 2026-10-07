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

Actual CUDA validation on 2.14 and 2.7, including offloaded multi-cache prediction, is pending in the coordinated GPU investigation. No end-to-end speed or support claim is made until those runs complete.

The source delta is commit `18a4d151a`, two replacements in `sdm/models/base.py`, based on frozen integration `c75ba3af6`. The separate resident-cache fast path is not included, keeping attribution clear.
