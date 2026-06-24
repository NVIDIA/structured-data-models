from collections.abc import Callable

import torch


def onlyCUDA(func: Callable) -> Callable:
    """Skip the test if CUDA is not available."""
    import pytest

    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def withCUDA(func: Callable) -> Callable:
    """Parametrize the test over ``cpu`` and, if available, ``cuda:0``."""
    import pytest

    devices = [pytest.param(torch.device("cpu"), id="cpu")]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device("cuda:0"), id="cuda:0"))

    return pytest.mark.parametrize("device", devices)(func)
