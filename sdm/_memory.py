# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch


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
