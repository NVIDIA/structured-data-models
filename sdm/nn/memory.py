"""CUDA inference memory heuristics.

Reserve 20% (at least 512 MiB) of live headroom. Attention chunks use at
most 5% of the current process's PyTorch CUDA allocator limit, and callers
choose the largest work chunk that fits the resulting budget.
"""

from math import prod

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry

_CUDA_MEMORY_RESERVE_BYTES = 512 * 1024**2
_CUDA_MEMORY_RESERVE_FRACTION = 0.2
_ATTENTION_WORK_FACTOR = 12


def cuda_memory_limits(device: torch.device) -> tuple[int, int]:
    """Return safe headroom and the current process's allocator limit.

    Headroom is CUDA-free memory plus unused PyTorch cache, capped by the
    allocator limit. A margin is retained for fragmentation and unmodeled
    CUDA work.
    """
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    allocator_limit = int(
        total * torch.cuda.get_per_process_memory_fraction(device)
    )
    headroom = max(
        min(free + reserved - allocated, allocator_limit - allocated),
        0,
    )
    margin = max(
        _CUDA_MEMORY_RESERVE_BYTES,
        int(headroom * _CUDA_MEMORY_RESERVE_FRACTION),
    )
    return max(headroom - margin, 0), allocator_limit


def cuda_attention_memory_limit(device: torch.device) -> int:
    """Return the safe memory limit for one attention chunk."""
    headroom, allocator_limit = cuda_memory_limits(device)
    return min(headroom, allocator_limit // 20)


def attention_batch_size_limit(
    requested_limit: int | None,
    query: Tensor,
    key_value: Tensor | KVCacheEntry,
    work_byte_limit: int | None,
) -> int | None:
    """Return the smallest explicit or memory-derived attention batch limit."""
    if work_byte_limit is None:
        return requested_limit

    batch_shapes = [query.size()[:-2]]
    sequence_length = query.size(-2)
    if isinstance(key_value, Tensor):
        batch_shapes.append(key_value.size()[:-2])
        sequence_length = max(sequence_length, key_value.size(-2))
    else:
        batch_shapes.extend(
            [key_value.key.size()[:-3], key_value.value.size()[:-3]]
        )
        sequence_length = max(sequence_length, key_value.key.size(-3))

    batch_size = prod(torch.broadcast_shapes(*batch_shapes))
    bytes_per_batch = (
        _ATTENTION_WORK_FACTOR
        * sequence_length
        * query.size(-1)
        * max(query.element_size(), 4)
    )
    automatic_limit = max(1, work_byte_limit // max(bytes_per_batch, 1))
    if batch_size <= automatic_limit:
        return requested_limit
    if requested_limit is None:
        return automatic_limit
    return min(requested_limit, automatic_limit)
