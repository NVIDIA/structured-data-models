"""cuDNN-frontend variable-length SDPA for padded key/value streams.

Replaces the boolean-mask path in :class:`~sdm.nn.SDPA` for calls that
provide ``seqused_key_value``: instead of materializing a mask and
routing to the masked kernels, the attention runs on cuDNN's native
padding-mask support with per-batch valid lengths (``seq_len_kv``),
which bounds the computation to the valid region. Measured on GB200:
3.1-5.5x over the boolean-mask path at TabICLv2's D=64 ICL shapes, and
~2.0-2.4x at the D=16 column/row-embedding shapes the gate also serves,
with fp32-reference deviation within 2x of the boolean-mask path's at
every bake-off shape.

The dependency (``nvidia-cudnn-frontend``, the ``cudnn`` extra) is
optional: when it is missing, eligibility fails and callers keep the
boolean-mask path. When cuDNN cannot build a plan for a shape, the op
itself degrades to an equivalent masked fallback at runtime (probed
once per shape, negatively cached, warned once) - the probe lives
inside the op because the op body is opaque to compilation, so compiled
graphs stay valid for both outcomes. Execution graphs are cached per
(batch, heads, lengths, head-dim, dtype) shape, matching the
bucketed-serving design where the shape set is finite - disabling the
path clears the cache; the valid lengths remain runtime tensor data, so
varying them never rebuilds a graph.

The op is registered through :mod:`torch.library` with a fake
implementation so compiled callers keep ``fullgraph=True``. Execution
is serialized by a module lock (single-stream serving is the target;
concurrent callers are correct but serialized). The path targets
inference (gradient-enabled calls are ineligible) and is validated
under eager and ``torch.compile``; combining it with
``mode="reduce-overhead"`` CUDA graphs is untested.
"""

import math
import threading
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import cudnn as _cudnn_fe  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - optional dependency
    _cudnn_fe = None

_enabled = False
_lock = threading.Lock()
_graph_cache: dict[tuple[Any, ...], Any] = {}
_handles: dict[int, Any] = {}
_UNPROBED = object()


def enable_cudnn_varlen(enabled: bool = True) -> bool:
    """Toggle the cuDNN variable-length attention path.

    Args:
        enabled: Whether eligible ``seqused_key_value`` attention calls
            should use cuDNN's native padding-mask support instead of a
            boolean mask. Enabling has no effect when the optional
            ``nvidia-cudnn-frontend`` package is unavailable; disabling
            clears the cached execution graphs.

    Returns:
        Whether the path is active after the call.
    """
    global _enabled
    _enabled = bool(enabled) and _cudnn_fe is not None
    if not _enabled:
        with _lock:
            _graph_cache.clear()
    return _enabled


def is_available() -> bool:
    """Return whether the optional cuDNN frontend is importable."""
    return _cudnn_fe is not None


def _shape_eligible(
    query: Tensor,  # [B, Q, H, D] (flattened batch, pre-transpose layout)
    num_query_heads: int,
    num_key_value_heads: int,
) -> bool:
    """The shape/dtype half of :func:`eligible`, testable without CUDA."""
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
    # raises in that case; keep the two consistent.
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
            self.workspace = torch.empty(
                max(graph.get_workspace_size(), 1),
                device=device,
                dtype=torch.uint8,
            )
            self.seq_q_value = torch.full(
                (B, 1, 1, 1), Q, dtype=torch.int32, device=device
            )


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
    try:
        graph = _Graph(B, H, Q, KV, D, dtype, device)
    except Exception as exc:  # noqa: BLE001
        _graph_cache[graph_key] = None
        warnings.warn(
            f"cuDNN variable-length attention unavailable for shape "
            f"B={B} H={H} Q={Q} KV={KV} D={D} ({exc}); using the masked "
            f"fallback for this shape.",
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
    """Variable-length SDPA on cuDNN's native padding-mask support."""
    assert _cudnn_fe is not None
    B, Q, H, D = query.shape
    KV = key.size(1)
    # The lock serializes handle/stream binding and the shared per-graph
    # workspace: the module targets single-stream serving; concurrent
    # callers are correct but serialized. The support probe lives inside
    # the op (opaque to compilation), so unsupported shapes degrade to
    # the masked fallback at runtime without touching the traced graph.
    with _lock:
        graph = _get_graph_or_none(B, H, Q, KV, D, query.dtype, query.device)
        if graph is None:
            return _masked_fallback(query, key, value, seqused_key_value)
        out = torch.empty_like(query)
        _cudnn_fe.set_stream(
            handle=graph.handle,
            stream=torch.cuda.current_stream(query.device).cuda_stream,
        )
        graph.graph.execute(
            {
                graph.q: query,
                graph.k: key,
                graph.v: value,
                graph.seq_q: graph.seq_q_value,
                graph.seq_kv: seqused_key_value.view(B, 1, 1, 1).contiguous(),
                graph.out: out,
            },
            graph.workspace,
            handle=graph.handle,
        )
    return out


@cudnn_varlen_sdpa.register_fake
def _(query, key, value, seqused_key_value):
    return torch.empty_like(query)
