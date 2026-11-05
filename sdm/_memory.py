# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true


def _split_tensor(
    tensor: torch.Tensor, size: int, dim: int
) -> tuple[torch.Tensor, ...]:
    """Return one chunk when the dimension provably fits the split size."""
    if statically_known_true(tensor.size(dim) <= size):
        return (tensor,)
    return tensor.split(size, dim=dim)


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


def set_pinned_register_threads(num_threads: int | None = None) -> None:
    r"""Set the number of threads that pin new host allocations.

    By default, PyTorch pins host memory from a single thread. Pinning from
    several threads can reduce the allocation time of large pinned buffers,
    such as the activations of ``offload="cpu"``. The setting applies to the
    whole process. Equivalent alternatives are:

    - the ``PYTORCH_ALLOC_CONF`` environment variable, which must be set
      before the process starts, *e.g.*,
      ``PYTORCH_ALLOC_CONF=pinned_use_cuda_host_register:True,pinned_num_register_threads:8``.
    - the private PyTorch call
      ``torch._C._accelerator_setAllocatorSettings("pinned_use_cuda_host_register:True,pinned_num_register_threads:8")``.

    See the ``pinned_num_register_threads`` option in the
    `PyTorch CUDA memory management notes
    <https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management>`__.
    Requires PyTorch 2.14 or newer and does nothing otherwise.

    Args:
        num_threads: The number of threads, which PyTorch requires to be a
            power of two of at most ``128``. ``None`` uses ``8``, the value
            PyTorch recommends, or :func:`torch.get_num_threads` rounded down
            to a power of two if smaller.
    """
    if torch.__version__ < "2.14":
        return
    if num_threads is None:
        num_threads = 1 << (min(torch.get_num_threads(), 8).bit_length() - 1)
    torch._C._accelerator_setAllocatorSettings(
        "pinned_use_cuda_host_register:True,"
        f"pinned_num_register_threads:{num_threads}"
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
