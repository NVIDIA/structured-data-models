"""cuDNN-frontend variable-length SDPA for padded key/value streams."""

import math
import threading
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm._warnings import warn_once

try:
    import cudnn as _cudnn_fe  # ty: ignore[unresolved-import]
except Exception:  # noqa: BLE001 - the frontend dlopens libcudnn at import
    _cudnn_fe = None  # pragma: no cover - optional dependency

_enabled = False
# Serializes handle/stream binding and graph-cache access: the module
# targets single-stream serving; concurrent callers are correct but
# serialized (the lock only orders host-side enqueue).
_lock = threading.Lock()
# Execution graphs are cached per (batch, heads, lengths, head-dim,
# dtype, device) shape, matching the bucketed-serving design where the
# shape set is finite; the valid lengths remain runtime tensor data, so
# varying them never rebuilds a graph.
_graph_cache: dict[tuple[Any, ...], Any] = {}
_handles: dict[int, Any] = {}
_UNPROBED = object()


def enable_cudnn_varlen(enabled: bool = True) -> bool:
    """Toggle the cuDNN variable-length attention path.

    When enabled, eligible attention calls that provide
    ``seqused_key_value`` replace the boolean-mask path in
    :class:`~sdm.nn.SDPA`: instead of materializing a mask and routing
    to the masked kernels, attention runs on cuDNN's native
    padding-mask support with per-batch valid lengths (``seq_len_kv``),
    which bounds the computation to the valid region.

    When cuDNN cannot build a plan for a shape, the op degrades to an
    equivalent masked fallback at runtime (probed once per shape,
    negatively cached, warned once), so ineligible shapes, a missing
    ``nvidia-cudnn-frontend``, or graph-build failures keep today's
    behavior; :func:`cudnn_varlen_stats` reports how many probed
    shapes built an execution graph versus degraded. The path targets
    inference (gradient-enabled calls are ineligible). The op runs
    outside :func:`torch.nn.functional.scaled_dot_product_attention`,
    so :func:`torch.nn.attention.sdpa_kernel` and
    :func:`torch.backends.cuda.enable_cudnn_sdp` do not reach it;
    disable this path to return to PyTorch's backend selection. Under
    CUDA graphs (``mode="reduce-overhead"`` or a manual capture) a shape
    must have run once eagerly - the compiler's warmup run suffices -
    before it is recorded; a shape first met inside a capture replays
    the masked fallback and is probed on its next eager call.

    One execution graph is cached per distinct eligible shape (batch,
    heads, query/key lengths, head dim, dtype, device) and reused for
    every call at that shape; disabling the path drops the cache entries.
    This is the same per-shape policy as PyTorch's own cuDNN SDPA graph
    cache. Each graph costs roughly 2 MB of host memory and a few tens
    of milliseconds to build, so pair the path with shape bucketing
    (``seqused_train`` padding, as the serving recipe does); a stream of
    unbucketed table shapes pays that cost once per new shape. Treat the
    toggle as a setup-time switch: the frontend keeps about 1 MB of host
    memory per dropped graph, so disabling and re-enabling around every
    request rebuilds the graphs and slowly grows the process. Compiled
    callers guard on the toggle and recompile when it changes; a
    ``mode="reduce-overhead"`` graph recorded while the path was enabled
    keeps replaying its recorded cuDNN kernels after such a cycle without
    probing again, so :func:`cudnn_varlen_stats` no longer counts those
    shapes.

    Args:
        enabled: Whether eligible ``seqused_key_value`` attention calls
            should use cuDNN's native padding-mask support instead of a
            boolean mask. Enabling has no effect when the optional
            ``nvidia-cudnn-frontend`` package (installable via the
            ``cudnn`` extra: ``pip install 'structured-data-models[cudnn]'``)
            is unavailable; disabling clears the cached execution
            graphs.

    Returns:
        Whether the path is active after the call.
    """
    global _enabled
    _enabled = bool(enabled) and _cudnn_fe is not None
    if not _enabled:
        with _lock:
            _graph_cache.clear()
    return _enabled


def cudnn_varlen_stats() -> tuple[int, int]:
    """Return ``(built, degraded)`` engagement counts for probed shapes.

    ``built`` counts shapes serving on a cuDNN execution graph;
    ``degraded`` counts shapes whose graph build failed and were
    negatively cached, so their calls take the masked fallback. Both
    reset to zero when the path is disabled via
    :func:`enable_cudnn_varlen`. The counts reflect host-side probes:
    a CUDA graph replay (``mode="reduce-overhead"`` or a manual capture)
    that was recorded before such a reset re-executes the recorded
    kernels without probing and is not counted again.

    Returns:
        The ``(built, degraded)`` pair.
    """
    with _lock:
        graphs = list(_graph_cache.values())
    built = sum(graph is not None for graph in graphs)
    return built, len(graphs) - built


def is_available() -> bool:
    """Return whether the optional cuDNN frontend is importable."""
    return _cudnn_fe is not None


def _shape_eligible(
    query: Tensor,  # [B, Q, H, D] (flattened batch, pre-transpose layout)
    num_query_heads: int,
    num_key_value_heads: int,
) -> bool:
    """The shape/dtype half of :func:`eligible`, independent of device/grad."""
    return (
        query.dtype in (torch.bfloat16, torch.float16)
        and num_query_heads == num_key_value_heads
        and query.size(-1) % 8 == 0
        and query.size(-1) <= 128
        and query.size(0) <= 65535
    )


def eligible(
    query: Tensor,  # [B, Q, H, D] (flattened batch, pre-transpose layout)
    key: Tensor,  # [B, KV, H, D]
    num_query_heads: int,
    num_key_value_heads: int,
) -> bool:
    """Return whether this call can take the variable-length path."""
    # The graph derives its dtype and device from ``query`` alone and then
    # binds ``key``/``value`` to whatever arrives, so a mismatched key
    # would be reinterpreted instead of rejected. The boolean-mask path
    # raises in that case; keep the two consistent. ``value`` and the
    # count tensor are not visible here; the op itself routes their
    # mismatches to the masked fallback, which raises the same way.
    return (
        _enabled
        and _cudnn_fe is not None
        and query.is_cuda
        and key.dtype == query.dtype
        and key.device == query.device
        and not torch.is_grad_enabled()
        and _shape_eligible(query, num_query_heads, num_key_value_heads)
    )


class _Graph:
    """One built cuDNN execution graph for a fixed shape."""

    def __init__(
        self,
        B: int,
        H: int,
        Q: int,
        KV: int,
        D: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        assert _cudnn_fe is not None
        fe_dtype = (
            _cudnn_fe.data_type.BFLOAT16
            if dtype == torch.bfloat16
            else _cudnn_fe.data_type.HALF
        )
        index = device.index or 0
        with torch.cuda.device(device):
            if index not in _handles:
                _handles[index] = _cudnn_fe.create_handle()
            handle = _handles[index]
            self.handle = handle
            graph = _cudnn_fe.pygraph(
                io_data_type=fe_dtype,
                intermediate_data_type=_cudnn_fe.data_type.FLOAT,
                compute_data_type=_cudnn_fe.data_type.FLOAT,
                handle=handle,
            )

            # Logical BHSD dims over the caller's [B, S, H, D] contiguous
            # storage (BSHD physical layout - cuDNN's preferred layout).
            def _tensor(name: str, S: int) -> Any:
                return graph.tensor(
                    name=name,
                    dim=(B, H, S, D),
                    stride=(S * H * D, D, H * D, 1),
                    data_type=fe_dtype,
                )

            self.q = _tensor("q", Q)
            self.k = _tensor("k", KV)
            self.v = _tensor("v", KV)
            self.seq_q = graph.tensor(
                name="seq_q",
                dim=(B, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=_cudnn_fe.data_type.INT32,
            )
            self.seq_kv = graph.tensor(
                name="seq_kv",
                dim=(B, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=_cudnn_fe.data_type.INT32,
            )
            out, _ = graph.sdpa(
                name="sdpa",
                q=self.q,
                k=self.k,
                v=self.v,
                is_inference=True,
                attn_scale=1.0 / math.sqrt(D),
                use_padding_mask=True,
                seq_len_q=self.seq_q,
                seq_len_kv=self.seq_kv,
            )
            out.set_output(True).set_dim((B, H, Q, D)).set_stride(
                (Q * H * D, D, H * D, 1)
            ).set_data_type(fe_dtype)
            self.out = out
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([_cudnn_fe.heur_mode.A])
            graph.check_support()
            graph.build_plans()
            self.graph = graph
            self.workspace_size = max(graph.get_workspace_size(), 1)


def _get_graph_or_none(
    B: int,
    H: int,
    Q: int,
    KV: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """Build (or fetch) the execution graph; ``None`` when unsupported.

    A failed build is negatively cached and warned once per shape, so an
    unsupported shape degrades to the masked fallback instead of raising
    mid-serving. Callers must hold the module lock.
    """
    graph_key = (B, H, Q, KV, D, str(dtype), device.index or 0)
    cached = _graph_cache.get(graph_key, _UNPROBED)
    if cached is not _UNPROBED:
        return cached
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        # Handle creation and plan building under CUDA graph capture would
        # invalidate the capture; a shape first met inside a capture takes
        # the masked fallback and is probed on its next eager call.
        return None
    try:
        graph = _Graph(B, H, Q, KV, D, dtype, device)
    except Exception as exc:  # noqa: BLE001
        _graph_cache[graph_key] = None
        warn_once(
            key=f"cudnn-varlen-unsupported-{graph_key}",
            message=(
                f"cuDNN variable-length attention unavailable for shape "
                f"B={B} H={H} Q={Q} KV={KV} D={D} ({exc}); using the "
                f"masked fallback for this shape."
            ),
            stacklevel=2,
        )
        return None
    _graph_cache[graph_key] = graph
    return graph


def _masked_fallback(
    query: Tensor,  # [B, Q, H, D]
    key: Tensor,  # [B, KV, H, D]
    value: Tensor,  # [B, KV, H, D]
    seqused_key_value: Tensor,  # [B]
) -> Tensor:
    """Boolean-mask attention identical to the SDPA seqused path."""
    kv_index = torch.arange(key.size(1), device=key.device)
    mask = kv_index.view(1, 1, 1, -1) < seqused_key_value.view(-1, 1, 1, 1)
    out = F.scaled_dot_product_attention(
        query=query.transpose(-3, -2),
        key=key.transpose(-3, -2),
        value=value.transpose(-3, -2),
        attn_mask=mask,
    ).transpose(-3, -2)
    # The op's fake registration promises a contiguous [B, Q, H, D]
    # result; a transposed view here would violate that stride contract
    # and crash compiled callers on the degrade path.
    return out.contiguous()


@torch.library.custom_op("sdm::cudnn_varlen_sdpa", mutates_args=())
def cudnn_varlen_sdpa(
    query: Tensor,  # [B, Q, H, D] contiguous
    key: Tensor,  # [B, KV, H, D] contiguous
    value: Tensor,  # [B, KV, H, D] contiguous
    seqused_key_value: Tensor,  # [B] int32
) -> Tensor:  # [B, Q, H, D]
    """Variable-length SDPA on cuDNN's native padding-mask support.

    Counts are bounded to ``[0, KV]`` before they reach cuDNN, which
    binds them as raw sequence lengths: an over-range count saturates
    like the boolean mask instead of reading past the valid keys. A count
    of 0 follows cuDNN's zero-valid-length semantics.

    Shape notation: ``D`` here is the per-head channel dimension,
    written ``C`` in :mod:`sdm.nn.attention`.
    """
    assert _cudnn_fe is not None
    B, Q, H, D = query.shape
    KV = key.size(1)
    # The graph derives its geometry from ``query`` (plus ``key``'s
    # length) and binds the remaining tensors raw, so a mismatched
    # value/key geometry, batch, count dtype/length, or a
    # foreign-device count would be reinterpreted instead of rejected.
    # :func:`eligible` cannot see ``value`` or the count; route their
    # mismatches to the masked fallback, which raises the same errors
    # as the boolean-mask path. The graph also declares fixed BSHD
    # strides and binds raw storage, so non-contiguous inputs take the
    # fallback too. All checks are host-side metadata (no device sync).
    if (
        not query.is_contiguous()
        or not key.is_contiguous()
        or not value.is_contiguous()
        or value.dtype != query.dtype
        or value.device != query.device
        or value.size() != key.size()
        or key.size(0) != B
        or key.size(-1) != D
        or key.size(-2) != H
        or seqused_key_value.device != query.device
        or seqused_key_value.dtype != torch.int32
        or seqused_key_value.numel() != B
    ):
        return _masked_fallback(query, key, value, seqused_key_value)
    # The lock serializes handle/stream binding and graph-cache access:
    # the module targets single-stream serving; concurrent callers are
    # correct but serialized. The support probe lives inside the op
    # (opaque to compilation), so unsupported shapes degrade to the
    # masked fallback at runtime without touching the traced graph.
    with _lock:
        graph = _get_graph_or_none(B, H, Q, KV, D, query.dtype, query.device)
        if graph is None:
            return _masked_fallback(query, key, value, seqused_key_value)
        out = torch.empty_like(query)
        # Allocated per call: the caching allocator makes this cheap and
        # stream-ordered, so callers on distinct streams never share
        # device-side scratch (the lock only orders host-side enqueue).
        workspace = torch.empty(
            graph.workspace_size, device=query.device, dtype=torch.uint8
        )
        # The query-length constant is built per call rather than cached
        # with the graph: a cached device tensor allocated during a CUDA
        # graph warmup run would live in the graph's private pool.
        seq_q = torch.full(
            (B, 1, 1, 1), Q, dtype=torch.int32, device=query.device
        )
        seq_kv = seqused_key_value.clamp(min=0, max=KV).view(B, 1, 1, 1)
        # cuDNN requires the handle's device to be current at call time;
        # guard it like the build in :class:`_Graph` already does.
        with torch.cuda.device(query.device):
            stream = torch.cuda.current_stream(query.device)
            _cudnn_fe.set_stream(
                handle=graph.handle, stream=stream.cuda_stream
            )
            graph.graph.execute(
                {
                    graph.q: query,
                    graph.k: key,
                    graph.v: value,
                    graph.seq_q: seq_q,
                    graph.seq_kv: seq_kv,
                    graph.out: out,
                },
                workspace,
                handle=graph.handle,
            )
    return out


# The fake implementation keeps compiled callers on ``fullgraph=True``;
# the runtime support probe lives inside the op body (opaque to
# compilation), so one compiled graph stays valid whether a shape
# builds an execution graph or degrades to the masked fallback. Both
# real paths return a contiguous tensor, so the fake promises the same
# layout regardless of the input strides.
@cudnn_varlen_sdpa.register_fake
def _(
    query: Tensor, key: Tensor, value: Tensor, seqused_key_value: Tensor
) -> Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
