# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable

import torch


def onlyCUDA(func: Callable) -> Callable:
    """Mark a CUDA-only test and skip it when CUDA is unavailable.

    Args:
        func: Test function that sets up CUDA inputs or resources.
    """
    import pytest

    func = pytest.mark.cuda(func)
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def withCUDA(func: Callable) -> Callable:
    """Exercise distinct CPU and CUDA paths with the same assertions.

    The test receives a ``device`` argument. Only the CUDA case receives
    the ``cuda`` marker and is skipped when CUDA is unavailable.

    Args:
        func: Test function that accepts a ``device`` argument.
    """
    import pytest

    devices = [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda:0"),
            id="cuda:0",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA not available",
                ),
            ],
        ),
    ]

    return pytest.mark.parametrize("device", devices)(func)


def onlyMPS(func: Callable) -> Callable:
    """Skip the test if MPS is not available."""
    import pytest

    return pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )(func)


def onlyFullTest(func: Callable) -> Callable:
    r"""Skip the test if it is not a full test run."""
    import pytest

    return pytest.mark.skipif(
        os.getenv("FULL_TEST", "0") != "1",
        reason="Fast test run",
    )(func)
