from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _segment_multi_reduce_kernel(
    src_ptr,
    offsets_ptr,
    sum_ptr,
    mean_ptr,
    std_ptr,
    min_ptr,
    max_ptr,
    num_channels: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    segment = tl.program_id(0)
    channels = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    channel_mask = channels < num_channels
    start = tl.load(offsets_ptr + segment)
    end = tl.load(offsets_ptr + segment + 1)

    total = tl.zeros((block_channels,), tl.float32)
    square_total = tl.zeros((block_channels,), tl.float32)
    minimum = tl.full((block_channels,), float("inf"), tl.float32)
    maximum = tl.full((block_channels,), -float("inf"), tl.float32)
    has_nan = tl.zeros((block_channels,), tl.int1)

    edge = start
    while edge < end:
        value = tl.load(
            src_ptr + edge.to(tl.int64) * num_channels + channels,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        total += value
        square_total += value * value
        minimum = tl.minimum(minimum, value)
        maximum = tl.maximum(maximum, value)
        has_nan |= value != value
        edge += 1

    count = end - start
    divisor = tl.maximum(count, 1)
    mean = total / divisor
    variance = square_total / divisor - mean * mean
    clamped_variance = tl.where(variance < 1e-5, 1e-5, variance)
    std = tl.where(
        variance <= 1e-5,
        0.0,
        tl.sqrt(clamped_variance),
    )
    minimum_is_inf = (minimum == float("inf")) | (minimum == -float("inf"))
    maximum_is_inf = (maximum == float("inf")) | (maximum == -float("inf"))
    minimum = tl.where(minimum_is_inf, 0.0, minimum)
    maximum = tl.where(maximum_is_inf, 0.0, maximum)
    minimum = tl.where(has_nan, float("nan"), minimum)
    maximum = tl.where(has_nan, float("nan"), maximum)

    output_offsets = segment.to(tl.int64) * num_channels + channels
    tl.store(sum_ptr + output_offsets, total, mask=channel_mask)
    tl.store(mean_ptr + output_offsets, mean, mask=channel_mask)
    tl.store(std_ptr + output_offsets, std, mask=channel_mask)
    tl.store(min_ptr + output_offsets, minimum, mask=channel_mask)
    tl.store(max_ptr + output_offsets, maximum, mask=channel_mask)


def segment_multi_reduce(
    src: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not src.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("src and offsets must be contiguous")

    shape = (offsets.numel() - 1, src.size(1))
    outputs = (
        src.new_empty(shape),
        src.new_empty(shape),
        src.new_empty(shape),
        src.new_empty(shape),
        src.new_empty(shape),
    )
    if shape[0] == 0 or shape[1] == 0:
        return outputs

    block_channels = min(triton.next_power_of_2(shape[1]), 512)
    if block_channels >= 256:
        num_warps = 8
    elif block_channels >= 128:
        num_warps = 4
    else:
        num_warps = 1
    grid = (shape[0], triton.cdiv(shape[1], block_channels))
    with torch.cuda.device(src.device):
        cast(Any, _segment_multi_reduce_kernel)[grid](
            src,
            offsets,
            *outputs,
            num_channels=shape[1],
            block_channels=block_channels,
            num_warps=num_warps,
        )
    return outputs
