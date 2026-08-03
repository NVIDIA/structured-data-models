"""CUDA inference memory heuristics.

Reserve 20% (at least 512 MiB) of available memory. Attention chunks use at
most 5% of the current process's PyTorch CUDA allocator limit, and callers
choose the largest work chunk that fits the resulting budget.
"""

from math import prod

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry

_CUDA_MEMORY_RESERVE_BYTES = 512 * 1024**2
_CUDA_MEMORY_RESERVE_FRACTION = 0.2
_ATTENTION_MEMORY_FRACTION = 0.05
_ATTENTION_WORK_FACTOR = 12


def cuda_memory_availability(device: torch.device) -> tuple[int, int]:
    """Return safely available CUDA memory and process capacity.

    Available memory includes CUDA-free memory and unused PyTorch cache, capped
    by the current process's allocator limit.
    """
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    capacity = int(total * torch.cuda.get_per_process_memory_fraction(device))
    available = max(
        min(free + reserved - allocated, capacity - allocated),
        0,
    )
    reserve = max(
        _CUDA_MEMORY_RESERVE_BYTES,
        int(available * _CUDA_MEMORY_RESERVE_FRACTION),
    )
    return max(available - reserve, 0), capacity


def cuda_attention_memory_limit(device: torch.device) -> int:
    """Return the safe memory limit for one attention chunk."""
    available, capacity = cuda_memory_availability(device)
    return min(
        available,
        int(capacity * _ATTENTION_MEMORY_FRACTION),
    )


def attention_batch_size_limit(
    requested_limit: int | None,
    query: Tensor,
    key_value: Tensor | KVCacheEntry,
    attention_memory_limit: int | None,
) -> int | None:
    """Return the smallest explicit or memory-derived attention batch limit."""
    if attention_memory_limit is None:
        return requested_limit

    # Attention inputs use [..., tokens, channels].
    input_batch_shapes = [query.size()[:-2]]
    sequence_length = query.size(-2)
    # Use the longer query/KV token count for the memory estimate.
    if isinstance(key_value, Tensor):
        input_batch_shapes.append(key_value.size()[:-2])
        sequence_length = max(sequence_length, key_value.size(-2))
    else:
        # Cached K/V is already split into [..., tokens, heads, head_channels].
        input_batch_shapes.extend(
            [key_value.key.size()[:-3], key_value.value.size()[:-3]]
        )
        sequence_length = max(sequence_length, key_value.key.size(-3))

    # Match attention's broadcasting, then count its independent sequences.
    total_batch_size = prod(torch.broadcast_shapes(*input_batch_shapes))
    # Estimate one sequence's temporary bytes with the empirical work factor.
    estimated_bytes_per_batch = (
        _ATTENTION_WORK_FACTOR
        * sequence_length
        * query.size(-1)
        * max(query.element_size(), 4)
    )
    # Fit as many sequences as the byte budget allows; one is indivisible.
    automatic_batch_size_limit = max(
        1,
        attention_memory_limit // max(estimated_bytes_per_batch, 1),
    )
    # Keep the normal path when every sequence fits in one batch.
    if total_batch_size <= automatic_batch_size_limit:
        return requested_limit
    if requested_limit is None:
        return automatic_batch_size_limit
    # A caller-supplied smaller limit remains authoritative.
    return min(requested_limit, automatic_batch_size_limit)
