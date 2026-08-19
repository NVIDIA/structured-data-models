import contextlib
from collections.abc import Iterator

import torch


@contextlib.contextmanager
def inference_mode(mode: bool = True) -> Iterator[None]:
    r"""Context manager that enables or disables inference mode.

    Uses :func:`torch.no_grad` instead of :func:`torch.inference_mode` inside a
    compiled region.

    Args:
        mode: Whether to enable or disable inference mode.
    """
    # `torch.inference_mode` is not supported inside a compiled region:
    # https://github.com/pytorch/pytorch/issues/180823
    if torch.compiler.is_compiling():
        context = torch.set_grad_enabled(not mode)
    else:
        context = torch.inference_mode(mode)

    with context:
        yield
