import os
from collections.abc import Callable

import torch


def onlyCUDA(func: Callable) -> Callable:
    """Skip the test if CUDA is not available."""
    import pytest

    func = pytest.mark.cuda(func)
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def withCUDA(func: Callable) -> Callable:
    """Parametrize the test over ``cpu`` and ``cuda:0``."""
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


def onlyFullTest(func: Callable) -> Callable:
    r"""Skip the test if it is not a full test run."""
    import pytest

    return pytest.mark.skipif(
        os.getenv("FULL_TEST", "0") != "1",
        reason="Fast test run",
    )(func)
