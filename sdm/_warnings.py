# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch

_warned_once: set[str] = set()


def warn_once(
    key: str,
    message: str,
    *,
    category: type[Warning] = RuntimeWarning,
    stacklevel: int = 2,
) -> None:
    r"""Issue a warning once per process.

    The warning is not issued from compiled code.
    """
    if torch.compiler.is_compiling():
        return

    if not torch.is_warn_always_enabled() and key in _warned_once:
        return

    if not torch.is_warn_always_enabled():
        message += (
            " This warning will be suppressed for the remainder of this "
            "process."
        )
        _warned_once.add(key)

    warnings.warn(message, category=category, stacklevel=stacklevel)
