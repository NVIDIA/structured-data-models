import copy
import warnings

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sdm.nn import SDPA, TransformerBlock, _cudnn_varlen
from sdm.testing import onlyCUDA

# Unit tests for the sdm.nn._cudnn_varlen gating and custom op (the
# model-mediated toggle and equivalence tests live in
# test/models/tabiclv2/test_model.py). The eligibility and degrade tests
# are CPU-safe by design but additionally marked `cuda`, so the GPU CI
# job - where `nvidia-cudnn-frontend` is installed - also exercises them
# against the real dependency; the remaining tests run the kernel and
# need CUDA plus the optional package.


@pytest.mark.cuda
def test_cudnn_varlen_shape_eligibility() -> None:
    def shape_ok(
        batch: int = 4,
        head_dim: int = 64,
        num_heads: int = 8,
        num_key_value_heads: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ) -> bool:
        query = torch.empty(batch, 2, num_heads, head_dim, dtype=dtype)
        return _cudnn_varlen._shape_eligible(
            query,
            num_query_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
        )

    assert shape_ok()
    assert shape_ok(dtype=torch.float16)
    # Full precision stays on the boolean-mask path.
    assert not shape_ok(dtype=torch.float32)
    # Grouped-query attention stays on the boolean-mask path.
    assert not shape_ok(num_key_value_heads=2)
    # Head dims must be multiples of eight...
    assert not shape_ok(head_dim=36)
    # ...with 128 the inclusive ceiling (136 is a multiple of eight,
    # isolating the ceiling gate).
    assert shape_ok(head_dim=128)
    assert not shape_ok(head_dim=136)
    # Batch 65535 is the inclusive ceiling.
    assert shape_ok(
        batch=65535,
        head_dim=8,
        num_heads=1,
        num_key_value_heads=1,
    )
    assert not shape_ok(
        batch=65536,
        head_dim=8,
        num_heads=1,
        num_key_value_heads=1,
    )


@pytest.mark.cuda
def test_cudnn_varlen_eligibility_gates() -> None:
    query = torch.empty(4, 32, 8, 64)
    key = torch.empty(4, 48, 8, 64)

    # A disabled flag short-circuits everything.
    _cudnn_varlen.enable_cudnn_varlen(False)
    assert not _cudnn_varlen.eligible(
        query, key, num_query_heads=8, num_key_value_heads=8
    )

    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        # CPU tensors are never eligible (also covers environments
        # without the optional dependency, where enabling is inert).
        assert not _cudnn_varlen.eligible(
            query, key, num_query_heads=8, num_key_value_heads=8
        )
        if torch.cuda.is_available() and _cudnn_varlen.is_available():
            base = query.cuda().bfloat16()
            base_k = key.cuda().bfloat16()
            # Eligibility mirrors serving: inference (no-grad) context.
            with torch.no_grad():
                assert _cudnn_varlen.eligible(
                    base, base_k, num_query_heads=8, num_key_value_heads=8
                )
                # The graph takes dtype and device from ``query``, so a
                # key that differs in either must not be bound to it.
                assert not _cudnn_varlen.eligible(
                    base,
                    base_k.float(),
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
                assert not _cudnn_varlen.eligible(
                    base,
                    base_k.cpu(),
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
                # Grouped-query attention stays on the boolean-mask path.
                assert not _cudnn_varlen.eligible(
                    base, base_k, num_query_heads=8, num_key_value_heads=2
                )
                # Head dims must be multiples of eight (and at most 128).
                assert not _cudnn_varlen.eligible(
                    base[..., :36],
                    base_k[..., :36],
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
                # Head dim 128 is the inclusive ceiling (136 is a
                # multiple of eight, isolating the ceiling gate).
                wide = torch.empty(
                    2, 4, 8, 136, device="cuda", dtype=torch.bfloat16
                )
                assert _cudnn_varlen.eligible(
                    wide[..., :128],
                    wide[..., :128],
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
                assert not _cudnn_varlen.eligible(
                    wide,
                    wide,
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
                # Batch 65535 is the inclusive ceiling.
                flat = torch.empty(
                    65536, 1, 1, 8, device="cuda", dtype=torch.bfloat16
                )
                assert _cudnn_varlen.eligible(
                    flat[:65535],
                    flat[:65535],
                    num_query_heads=1,
                    num_key_value_heads=1,
                )
                assert not _cudnn_varlen.eligible(
                    flat,
                    flat,
                    num_query_heads=1,
                    num_key_value_heads=1,
                )
                # Full precision stays on the boolean-mask path.
                assert not _cudnn_varlen.eligible(
                    base.float(),
                    base_k.float(),
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
            # Gradient-enabled calls stay on the boolean-mask path.
            with torch.enable_grad():
                assert not _cudnn_varlen.eligible(
                    base.clone().requires_grad_(True),
                    base_k,
                    num_query_heads=8,
                    num_key_value_heads=8,
                )
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


@pytest.mark.cuda
def test_cudnn_varlen_build_failure_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A graph-build failure must degrade to the masked fallback inside
    # the op (probed once, negatively cached), never raise mid-serving.
    # Monkeypatching `_Graph` itself (rather than the cuDNN frontend)
    # keeps the simulated failure reachable on every machine: the raise
    # happens inside `_get_graph_or_none`'s probe on CPU and CUDA alike,
    # and the warning match below pins the mock's message so the test
    # fails if the simulated failure ever stops being exercised.
    class _FailingGraph:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("No execution plans support the graph.")

    monkeypatch.setattr(_cudnn_varlen, "_cudnn_fe", object())
    monkeypatch.setattr(_cudnn_varlen, "_enabled", True)
    monkeypatch.setattr(_cudnn_varlen, "_Graph", _FailingGraph)
    # Start from a clean cache so the public stats assertion below sees
    # exactly this test's probe.
    _cudnn_varlen._graph_cache.clear()

    generator = torch.Generator().manual_seed(0)
    query = torch.randn(2, 16, 8, 64, generator=generator)
    key = torch.randn(2, 32, 8, 64, generator=generator)
    value = torch.randn(2, 32, 8, 64, generator=generator)
    seqused = torch.tensor([20, 32], dtype=torch.int32)
    try:
        # Pin the math backend: the fused CPU kernels happen to return
        # contiguous outputs even without the fallback's stride fix,
        # which would make the contract assertions below vacuous.
        with sdpa_kernel([SDPBackend.MATH]):
            with pytest.warns(RuntimeWarning, match="No execution plans"):
                out = _cudnn_varlen.cudnn_varlen_sdpa(
                    query, key, value, seqused
                )
            expected = _cudnn_varlen._masked_fallback(
                query, key, value, seqused
            )
        torch.testing.assert_close(out, expected)
        # Negatively cached: the second call neither warns nor rebuilds.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, seqused)
        assert not record
        # The public stats accessor reports the negatively cached probe
        # as a degrade (0 built, 1 degraded).
        assert _cudnn_varlen.cudnn_varlen_stats() == (0, 1)
        # The fallback must honor the op's fake stride contract: a
        # compiled caller receives the fake's layout and crashes if the
        # real output is a transposed view.
        assert out.is_contiguous()
        fake = torch.empty_like(query)
        assert out.stride() == fake.stride()
    finally:
        # The monkeypatch restores the module globals; drop the
        # negatively cached probe so later tests see a clean cache.
        _cudnn_varlen._graph_cache.clear()


@onlyCUDA
def test_cudnn_varlen_disable_during_inflight_execution() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(0)
    query = torch.randn(
        4, 64, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        4, 130, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        4, 130, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    counts = torch.tensor([130, 77, 1, 100], dtype=torch.int32, device=device)
    # Disabling clears the graph cache, so the stats below see only this
    # test's graph.
    _cudnn_varlen.enable_cudnn_varlen(False)
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        # Build the graph on the default stream, then execute on a second
        # stream and drop the graph while those executions may still be
        # in flight.
        with torch.no_grad():
            expected = _cudnn_varlen.cudnn_varlen_sdpa(
                query, key, value, counts
            )
            torch.cuda.synchronize()
            # The comparison below is only meaningful on the cuDNN graph,
            # not on the masked fallback a failed build would degrade to.
            assert _cudnn_varlen.cudnn_varlen_stats() == (1, 0)
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                outs = [
                    _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, counts)
                    for _ in range(8)
                ]
            _cudnn_varlen.enable_cudnn_varlen(False)
            # Churn the allocator so memory released by the drop would be
            # reused before the queued kernels have finished.
            junk = [
                torch.full((4, 1, 1, 1), 7, dtype=torch.int32, device=device)
                for _ in range(64)
            ]
            stream.synchronize()
        for out in outs:
            assert torch.equal(out, expected)
        del junk
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


@onlyCUDA
def test_cudnn_varlen_non_contiguous_inputs_take_fallback() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(1)
    # [B, H, Q, D] storage viewed as [B, Q, H, D]: the layout the graph
    # declares, but with the strides it does not.
    query = torch.randn(
        4, 8, 64, 64, device=device, dtype=torch.bfloat16, generator=generator
    ).transpose(1, 2)
    key = torch.randn(
        4, 8, 130, 64, device=device, dtype=torch.bfloat16, generator=generator
    ).transpose(1, 2)
    value = torch.randn(
        4, 8, 130, 64, device=device, dtype=torch.bfloat16, generator=generator
    ).transpose(1, 2)
    assert not query.is_contiguous()
    counts = torch.tensor([130, 77, 1, 100], dtype=torch.int32, device=device)
    _cudnn_varlen.enable_cudnn_varlen(False)
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        with torch.no_grad():
            out = _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, counts)
            expected = _cudnn_varlen._masked_fallback(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                counts,
            )
        torch.testing.assert_close(out, expected)
        # Routed to the fallback before any graph was built.
        assert _cudnn_varlen.cudnn_varlen_stats() == (0, 0)
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


@onlyCUDA
def test_cudnn_varlen_compiled_non_contiguous_query() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(2)
    query = torch.randn(
        4, 8, 64, 64, device=device, dtype=torch.bfloat16, generator=generator
    ).transpose(1, 2)
    key = torch.randn(
        4, 130, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        4, 130, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    counts = torch.tensor([130, 77, 1, 100], dtype=torch.int32, device=device)
    _cudnn_varlen.enable_cudnn_varlen(False)
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        # The fake must describe the contiguous output the fallback
        # returns for a non-contiguous query, or the compiled graph would
        # be laid out for the query's strides instead.
        compiled = torch.compile(
            _cudnn_varlen.cudnn_varlen_sdpa,
            fullgraph=True,
            backend="aot_eager",
        )
        with torch.no_grad():
            out = compiled(query, key, value, counts)
            expected = _cudnn_varlen.cudnn_varlen_sdpa(
                query, key, value, counts
            )
        assert out.is_contiguous()
        torch.testing.assert_close(out, expected)
    finally:
        torch._dynamo.reset()
        _cudnn_varlen.enable_cudnn_varlen(False)


@onlyCUDA
def test_cudnn_varlen_over_range_count_saturates() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(3)
    module = SDPA(num_query_heads=8)
    query = torch.randn(
        3, 4, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        3, 33, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        3, 33, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    exact = torch.full((3,), 33, dtype=torch.int32, device=device)
    _cudnn_varlen.enable_cudnn_varlen(False)
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        with torch.no_grad():
            expected = module(
                query=query, key=key, value=value, seqused_key_value=exact
            )
            # cuDNN binds ``seq_len_kv`` as a raw length: a count past the
            # key length reads beyond the valid keys and returns garbage or
            # NaN instead of saturating like the boolean mask does. The op
            # bounds the count, so over-range counts must reproduce the
            # exact-count output bit for bit, through `SDPA` and directly.
            for count in (34, 40, 1000, 2**31 - 1):
                over = exact.new_full((3,), count)
                out = module(
                    query=query, key=key, value=value, seqused_key_value=over
                )
                assert torch.equal(out, expected), count
                out = _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, over)
                assert torch.equal(out, expected), count
        # The comparison is only meaningful on the cuDNN graph, not on the
        # masked fallback a failed build would degrade to.
        assert _cudnn_varlen.cudnn_varlen_stats() == (1, 0)
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


@onlyCUDA
def test_cudnn_varlen_reduce_overhead_compile() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(4)
    channels = 64
    block = TransformerBlock(
        channels=channels,
        num_query_heads=4,
        mlp=torch.nn.Linear(channels, channels, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        # The residual exit is zero-initialized; randomize it so attention
        # (and thus the masking) reaches the output.
        block.attn.out_lin.weight.normal_(std=0.5, generator=generator)
    eager = copy.deepcopy(block)

    def inputs(kv_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.randn(
            4, 32, channels, device=device, dtype=dtype, generator=generator
        )
        key_value = torch.randn(
            4,
            kv_len,
            channels,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        counts = torch.tensor(
            [kv_len, kv_len // 2, 1, kv_len - 3],
            dtype=torch.int32,
            device=device,
        )
        return query, key_value, counts

    # Three calls at one shape walk CUDA graph trees through warmup,
    # recording, and replay; the second shape arrives after warmup.
    plan = [inputs(kv_len) for kv_len in (48, 48, 48, 80, 80)]
    _cudnn_varlen.enable_cudnn_varlen(False)
    with torch.inference_mode():
        # Boolean-mask references, taken before the path is enabled.
        expected = [
            eager(query, key_value, seqused_key_value=counts)
            for query, key_value, counts in plan
        ]
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        block.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
        with (
            torch.inference_mode(),
            warnings.catch_warnings(record=True) as record,
        ):
            warnings.simplefilter("always")
            for (query, key_value, counts), reference in zip(
                plan, expected, strict=True
            ):
                torch.compiler.cudagraph_mark_step_begin()
                out = block(query, key_value, seqused_key_value=counts)
                torch.testing.assert_close(
                    out.clone(), reference, atol=1e-2, rtol=1e-2
                )
        # The first shape ran on a cuDNN graph built during warmup; no
        # shape degraded to the fallback or was negatively cached, whether
        # the second shape was warmed up or first met while recording.
        built, degraded = _cudnn_varlen.cudnn_varlen_stats()
        assert built >= 1
        assert degraded == 0
        assert not [w for w in record if "variable-length" in str(w.message)]
    finally:
        torch._dynamo.reset()
        _cudnn_varlen.enable_cudnn_varlen(False)


@pytest.mark.cuda
def test_cudnn_varlen_masked_fallback_matches_sdpa() -> None:
    # The degrade tests above assert the op against `_masked_fallback`
    # itself, so a masking bug in the fallback would pass them. Pin the
    # fallback to independent references: the boolean-mask path of
    # `SDPA` with the same counts, and a per-batch slice of the valid
    # keys. Counts sit at both ends of the range so an off-by-one in
    # either direction is visible.
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(4, 3, 2, 8, generator=generator)
    key = torch.randn(4, 9, 2, 8, generator=generator)
    value = torch.randn(4, 9, 2, 8, generator=generator)
    counts = torch.tensor([9, 1, 8, 5], dtype=torch.int32)
    module = SDPA(num_query_heads=2)
    _cudnn_varlen.enable_cudnn_varlen(False)
    with sdpa_kernel([SDPBackend.MATH]):
        out = _cudnn_varlen._masked_fallback(query, key, value, counts)
        expected = module(
            query=query, key=key, value=value, seqused_key_value=counts
        )
        torch.testing.assert_close(out, expected)
        for i, count in enumerate(counts.tolist()):
            sliced = module(
                query=query[i], key=key[i, :count], value=value[i, :count]
            )
            torch.testing.assert_close(out[i], sliced)


@onlyCUDA
def test_cudnn_varlen_custom_scale_stays_on_mask_path() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(6)
    query = torch.randn(
        3, 4, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        3, 33, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        3, 33, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    counts = torch.tensor([33, 7, 20], dtype=torch.int32, device=device)
    # The op bakes in the default `1 / sqrt(C)` scale; a module with a
    # custom scale must stay on the boolean-mask path instead of silently
    # attending with the wrong temperature.
    module = SDPA(num_query_heads=8, scale=1.0)
    _cudnn_varlen.enable_cudnn_varlen(False)
    with torch.no_grad():
        expected = module(
            query=query, key=key, value=value, seqused_key_value=counts
        )
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        with torch.no_grad():
            out = module(
                query=query, key=key, value=value, seqused_key_value=counts
            )
        assert _cudnn_varlen.cudnn_varlen_stats() == (0, 0)
        assert torch.equal(out, expected)
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)


@onlyCUDA
def test_cudnn_varlen_unprobed_shape_inside_capture_takes_fallback() -> None:
    if not _cudnn_varlen.is_available():
        pytest.skip("requires nvidia-cudnn-frontend")
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(7)
    query = torch.randn(
        2, 3, 4, 32, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        2, 17, 4, 32, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        2, 17, 4, 32, device=device, dtype=torch.bfloat16, generator=generator
    )
    counts = torch.tensor([17, 5], dtype=torch.int32, device=device)
    # Disabling clears the graph cache, so this shape is unprobed.
    _cudnn_varlen.enable_cudnn_varlen(False)
    _cudnn_varlen.enable_cudnn_varlen(True)
    try:
        # Building a graph (handle creation, plan building, the build
        # sync) under stream capture would invalidate the capture: a shape
        # first met while capturing takes the masked fallback without
        # touching the cache, and builds on its next eager call.
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            out = _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, counts)
        torch.cuda.synchronize()
        assert _cudnn_varlen.cudnn_varlen_stats() == (0, 0)
        assert not _cudnn_varlen._graph_cache
        # The replay is the captured fallback on the new data and counts.
        query.copy_(torch.randn_like(query))
        counts.copy_(torch.tensor([9, 1], dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize()
        expected = _cudnn_varlen._masked_fallback(query, key, value, counts)
        assert torch.equal(out, expected)
        with torch.no_grad():
            eager = _cudnn_varlen.cudnn_varlen_sdpa(query, key, value, counts)
        assert _cudnn_varlen.cudnn_varlen_stats() == (1, 0)
        torch.testing.assert_close(eager, expected, atol=1e-2, rtol=1e-2)
    finally:
        _cudnn_varlen.enable_cudnn_varlen(False)
