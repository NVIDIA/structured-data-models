"""Experiment-only dynamic RMSNorm/RoPE regions; preparation is synthetic only."""
import gc
from contextlib import contextmanager
from contextvars import ContextVar
import time

import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend

ENABLED = False
_NATIVE_ONLY = ContextVar("kumo_compiler_native_only", default=False)
READY = False
PREPARING = False
VALIDATING = False
EVENTS = []
_COMPILED = {}


def _norm(x, weight, eps, dtype):
    shape = x.shape
    c = shape[-1]
    f = x.float().reshape(-1, c)
    group = f.reshape(-1, c // 4, 4)
    square_sum = group[..., 0] * group[..., 0]
    for i in (1, 2, 3):
        square_sum = torch.addcmul(square_sum, group[..., i], group[..., i])
    if c > 128:
        square_sum = square_sum.reshape(-1, c // 128, 32)
        width = 32
    else:
        width = c // 4
    while width > 1:
        half = width // 2
        square_sum = square_sum[..., :half] + square_sum[..., half:]
        width = half
    square_sum = square_sum.reshape(-1, max(1, c // 128))
    width = max(1, c // 128)
    while width > 1:
        half = width // 2
        square_sum = square_sum[..., :half] + square_sum[..., half:]
        width = half
    inv = (square_sum / c + eps).rsqrt()
    out = f * inv
    if weight is not None:
        out = out * weight
    return out.to(dtype).reshape(shape)

def _rope_norm(x, inv_freq, eps, dtype):
    # Match eager work: one sequence-frequency table, broadcast across batches/heads.
    position = torch.arange(x.shape[-3], device=x.device).float()
    frequency = position[:, None] * inv_freq[None, :]
    sine = frequency.sin().to(x.dtype)[None, :, None, :]
    cosine = frequency.cos().to(x.dtype)[None, :, None, :]
    first, second = x.chunk(2, dim=-1)
    left = first * cosine - second * sine
    right = second * cosine + first * sine
    rotated = torch.cat((left, right), dim=-1)
    return _norm(rotated, None, eps, dtype)


def _backend(graph, inputs):
    if not PREPARING:
        raise RuntimeError("Compiler backend entered outside synthetic warmup")
    EVENTS.append({"phase": "warmup", "nodes": len(list(graph.graph.nodes))})
    return lookup_backend("inductor")(graph, inputs)


def rms_norm(x, weight, eps, dtype, rope=None):
    if not READY and not PREPARING and not VALIDATING:
        raise RuntimeError("Compiled regions require successful adapter warmup")
    if x.numel() == 0:
        # Deliberate empty-query path, outside positive-length compiled regions.
        return F.rms_norm(x.float(), (x.shape[-1],), weight, eps).to(dtype)
    if x.dtype != torch.float16 or (rope is None and (x.shape[-1] != 128 or weight is None)) or (rope is not None and x.shape[-1] != 32):
        raise RuntimeError("Unprepared normalization configuration")
    shape = x.shape
    if rope is None:
        # Detach removes view-base guards while preserving storage and arithmetic.
        flat = x.reshape(-1, shape[-1]).contiguous().detach()
        if PREPARING:
            torch._dynamo.mark_static(flat, 1)
            if flat.shape[0] > 1:
                torch._dynamo.mark_dynamic(flat, 0)
        result = _COMPILED["norm"](flat, weight, eps, dtype)
    else:
        if rope.layout != "split_half" or rope.rotary_channels != rope.channels:
            raise RuntimeError("Unprepared rotary configuration")
        canonical = x.reshape(-1, *shape[-3:]).contiguous().detach()
        if PREPARING:
            for dim in (2, 3):
                torch._dynamo.mark_static(canonical, dim)
            for dim in (0, 1):
                if canonical.shape[dim] > 1:
                    torch._dynamo.mark_dynamic(canonical, dim)
        result = _COMPILED["rope"](canonical, rope.inv_freq, eps, dtype)
    return result.reshape(shape)


def _prepare():
    global READY, PREPARING, VALIDATING
    if READY:
        return {"already_ready": True, "variants": len(EVENTS)}
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config
    from sdm.nn import RotaryEmbedding

    start = time.perf_counter()
    READY = False
    PREPARING = True
    dynamo_config.recompile_limit = 256
    dynamo_config.accumulated_recompile_limit = 2048
    inductor_config.compile_threads = 4
    inductor_config.emulate_precision_casts = True
    inductor_config.triton.cudagraphs = False
    inductor_config.triton.autotune_pointwise = False
    _COMPILED["norm"] = torch.compile(_norm, backend=_backend, fullgraph=True, dynamic=True)
    _COMPILED["rope"] = torch.compile(_rope_norm, backend=_backend, fullgraph=True, dynamic=True)
    cases = 0
    weights = {c: torch.nn.Parameter(torch.ones(c, device="cuda")) for c in (128,)}
    rope = RotaryEmbedding(32, layout="split_half", requires_grad=False, device="cuda")
    # Compile only FP16 weighted cell width128 and FP16 RoPE32; other norms remain native eager.
    for mode in (torch.no_grad, torch.inference_mode):
        with mode(), torch.autocast("cuda", dtype=torch.float16):
            for c, input_dtype in ((128, torch.float16),):
                for n in (257, 1):
                    x = torch.randn(n, c, device="cuda", dtype=input_dtype)
                    rms_norm(x, weights[c], torch.finfo(torch.float32).eps, torch.float16)
                    cases += 1
            # Sequence is readout4 or readout4+features; batch can be a tail of1.
            for batch in (3, 1):
                x = torch.randn(batch, 17, 4, 32, device="cuda", dtype=torch.float16)
                rms_norm(x, None, 1e-6, torch.float32, rope)
                cases += 1
    torch.cuda.synchronize()
    PREPARING = False
    del weights, rope, x
    gc.collect()
    before = len(EVENTS)
    VALIDATING = True
    strict_cases = 0
    # New tensors, strides, sequence lengths, and parameter instances.
    with torch.compiler.set_stance("fail_on_recompile"):
        for mode in (torch.no_grad, torch.inference_mode):
            weights = {c: torch.nn.Parameter(torch.ones(c, device="cuda")) for c in (128,)}
            rope = RotaryEmbedding(32, layout="split_half", requires_grad=False, device="cuda")
            with mode(), torch.autocast("cuda", dtype=torch.float16):
                for c, input_dtype in ((128, torch.float16),):
                    for n in (1, 1001, 10003, 100001):
                        x = torch.randn(n, 2 * c, device="cuda", dtype=input_dtype)[:, :c]
                        rms_norm(x, weights[c], torch.finfo(torch.float32).eps, torch.float16)
                        strict_cases += 1
                for batch in (1, 2):
                    for sequence in (4, 73, 1001):
                        x = torch.randn(batch, sequence, 3 * 4 * 32, device="cuda", dtype=torch.float16)[..., 128:256].unflatten(-1, (4, 32))
                        rms_norm(x, None, 1e-6, torch.float32, rope)
                        strict_cases += 1
    torch.cuda.synchronize()
    assert len(EVENTS) == before
    VALIDATING = False
    READY = True
    return {"seconds": time.perf_counter() - start, "warmup_cases": cases,
            "strict_cases": strict_cases, "variants": len(EVENTS),
            "cuda_graphs": False, "emulate_precision_casts": True,
            "layout_policy": "Flatten prefix dimensions and materialize non-contiguous layouts; copy cost remains in fit/predict"}


def is_enabled():
    return ENABLED and not _NATIVE_ONLY.get()


@contextmanager
def native_only():
    token = _NATIVE_ONLY.set(True)
    try:
        yield
    finally:
        _NATIVE_ONLY.reset(token)


def enable():
    """Opt into prepared regions; eligible calls require successful preparation."""
    global ENABLED
    ENABLED = True


def prepare():
    from tabarena.models._weights import rng_guard

    enable()
    global PREPARING, VALIDATING, READY
    try:
        with rng_guard(cuda=True):
            return _prepare()
    except BaseException:
        PREPARING = VALIDATING = READY = False
        raise
