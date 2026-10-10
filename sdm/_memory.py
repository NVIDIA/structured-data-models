# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch.torch_version import TorchVersion


def configure_pinned_memory() -> None:
    r"""Avoid rounding large pinned allocations unless explicitly configured.

    The allocator setting applies process-wide and retains freed blocks for
    reuse. PyTorch versions before 2.13 do not support the rounding threshold.
    """
    if TorchVersion(torch.__version__) < "2.13":
        return
    settings = torch._C._accelerator_getAllocatorSettings()
    if "pinned_max_round_threshold_mb" in settings:
        return
    setting = "pinned_max_round_threshold_mb:1"
    torch._C._accelerator_setAllocatorSettings(
        f"{settings},{setting}" if settings else setting
    )


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
