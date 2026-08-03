from math import prod

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry

_CUDA_MEMORY_RESERVE_BYTES = 512 * 1024**2
_CUDA_MEMORY_RESERVE_FRACTION = 0.2
_ATTENTION_WORK_FACTOR = 12


def cuda_memory_budget(device: torch.device) -> tuple[int, int]:
    """Return usable CUDA headroom and the process memory limit."""
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    process_limit = int(
        total_bytes * torch.cuda.get_per_process_memory_fraction(device)
    )
    reusable_bytes = max(reserved_bytes - allocated_bytes, 0)
    unreserved_bytes = min(
        free_bytes,
        max(process_limit - reserved_bytes, 0),
    )
    headroom = min(
        unreserved_bytes + reusable_bytes,
        max(process_limit - allocated_bytes, 0),
    )
    reserve = max(
        _CUDA_MEMORY_RESERVE_BYTES,
        int(headroom * _CUDA_MEMORY_RESERVE_FRACTION),
    )
    return max(headroom - reserve, 0), process_limit


def cuda_attention_work_byte_limit(device: torch.device) -> int:
    """Return the target CUDA bytes for one attention chunk."""
    available_bytes, process_limit = cuda_memory_budget(device)
    return min(available_bytes, process_limit // 20)


def attention_batch_size_limit(
    requested_limit: int | None,
    query: Tensor,
    key_value: Tensor | KVCacheEntry,
    work_byte_limit: int | None,
) -> int | None:
    """Bound an attention batch using its sequence and channel dimensions."""
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
