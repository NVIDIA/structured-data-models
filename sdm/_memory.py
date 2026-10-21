# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch


def set_pinned_max_round_threshold(
    mb: int | None,
    force: bool = False,
) -> None:
    r"""Set the process-wide rounding threshold for pinned allocations.

    Args:
        mb: Threshold in MiB for power-of-two rounding of new pinned
            allocations. Requests above this threshold are not rounded up.
            ``None`` restores PyTorch's default threshold.
        force: Whether to replace an existing threshold.
    """
    if torch.__version__ < "2.14":
        return
    if mb is None:
        mb = (2**64 - 1) // 1024**2
    settings = torch._C._accelerator_getAllocatorSettings()
    if not force and "pinned_max_round_threshold_mb" in settings:
        return
    options = [
        option
        for option in settings.split(",")
        if option
        and option.split(":")[0].strip() != "pinned_max_round_threshold_mb"
    ]
    options.append(f"pinned_max_round_threshold_mb:{mb}")
    torch._C._accelerator_setAllocatorSettings(",".join(options))


def chunk_memory_limit(device: torch.device) -> int:
    r"""Bytes one chunk of a chunked operation may occupy on a CUDA device.

    The limit is the ``SDM_CHUNK_MEMORY_FRACTION`` (default ``0.05``) share of
    the device memory available to this process.
    """
    return int(
        torch.cuda.get_device_properties(device).total_memory
        * torch.cuda.get_per_process_memory_fraction(device)
        * float(os.getenv("SDM_CHUNK_MEMORY_FRACTION", "0.05"))
    )


def split_size(num_items: int, item_bytes: int, device: torch.device) -> int:
    r"""Split size of balanced chunks of items of ``item_bytes`` bytes each.

    On CUDA devices, chunks fit :func:`chunk_memory_limit`, unless a single
    item exceeds the limit. Autograd keeps the memory of every chunk, so one
    chunk holds all items elsewhere or while gradients are enabled.
    """
    if device.type != "cuda" or torch.is_grad_enabled() or item_bytes == 0:
        return max(num_items, 1)
    limit = max(chunk_memory_limit(device), 1)
    capacity = max(limit // item_bytes, 1)
    num_chunks = max(-(-num_items // capacity), 1)
    return max(-(-num_items // num_chunks), 1)
