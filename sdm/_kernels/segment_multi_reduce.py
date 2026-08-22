from collections.abc import Callable

import torch
from torch import Tensor

_SegmentMultiReduce = Callable[
    [Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
]

_triton_segment_multi_reduce: _SegmentMultiReduce | None = None
try:
    from sdm._kernels.triton.segment_multi_reduce import (
        segment_multi_reduce as _triton_segment_multi_reduce_impl,
    )
except ImportError:
    pass
else:
    _triton_segment_multi_reduce = _triton_segment_multi_reduce_impl


def _torch_segment_multi_reduce(
    src: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    values = (
        src.float() if src.dtype in {torch.float16, torch.bfloat16} else src
    )
    total = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="sum",
        unsafe=True,
        initial=0,
    )
    mean = total / offsets.diff().clamp(min=1).unsqueeze(1)
    variance = (
        torch.segment_reduce(
            values.square(),
            offsets=offsets,
            reduce="mean",
            unsafe=True,
            initial=0,
        )
        - mean.square()
    )
    std = torch.where(
        variance <= 1e-5,
        0.0,
        variance.clamp(min=1e-5).sqrt(),
    )
    minimum = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="min",
        unsafe=True,
    )
    maximum = torch.segment_reduce(
        values,
        offsets=offsets,
        reduce="max",
        unsafe=True,
    )
    minimum = torch.where(minimum.isinf(), 0.0, minimum)
    maximum = torch.where(maximum.isinf(), 0.0, maximum)
    return (
        total.to(src.dtype),
        mean.to(src.dtype),
        std.to(src.dtype),
        minimum.to(src.dtype),
        maximum.to(src.dtype),
    )


def segment_multi_reduce(
    src: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if (
        _triton_segment_multi_reduce is not None
        and src.is_cuda
        and src.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and src.is_contiguous()
        and offsets.is_contiguous()
        and offsets.device == src.device
        and offsets.dtype in {torch.int32, torch.int64}
    ):
        return _triton_segment_multi_reduce(src, offsets)
    return _torch_segment_multi_reduce(src, offsets)
