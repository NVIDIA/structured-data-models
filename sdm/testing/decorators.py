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
    """Parametrize the test over ``cpu`` and ``cuda:0``.

    The ``cuda:0`` case is skipped when CUDA is not available.
    """
    import pytest

    devices = [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda:0"),
            id="cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA not available",
            ),
        ),
    ]

    return pytest.mark.parametrize("device", devices)(func)
